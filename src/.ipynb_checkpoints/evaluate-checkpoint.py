import os
import csv
import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
from .scheduler import cosine_scheduler


def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking = True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics

def get_zeroshot_metrics(model, processor, test_dataloader, options):
    logging.info("Started zeroshot testing")
    if options.checkpoint_finetune is not None:
        if(os.path.isfile(options.checkpoint_finetune)):
            checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
            if(not options.distributed and next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
            if(options.distributed and not next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {f'module.{key}': value for key, value in checkpoint.items()}
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint {options.checkpoint_finetune}")
            
    if options.eval_data_type is not None:
        model.eval()
        umodel = model.module if(options.distributed) else model
        config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
        classes, templates = config["classes"], config["templates"]

        with torch.no_grad():
            text_embeddings = []
            if options.asr:
                backdoor_target_index = list(filter(lambda x: 'banana' in classes[x], range(len(classes))))
                backdoor_target_index = torch.tensor(backdoor_target_index[0]).to(options.device)
            for c in tqdm(classes):
                text = [template(c) for template in templates]
                text_tokens = processor.process_text(text)
                text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device)
                text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
                text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
                text_embedding = text_embedding.mean(dim = 0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)

        with torch.no_grad():
            topk = [1, 3, 5, 10]
            correct = {k: 0 for k in topk}
            total = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                image_embedding = umodel.get_image_features(image)
                image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
                logits = (image_embedding @ text_embeddings)
                ranks = logits.topk(max(topk), 1)[1].T
                predictions = ranks == label
                total += predictions.shape[1]
                for k in topk:
                    correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item()

        results = {f"zeroshot_top{k}": correct[k] / total for k in topk}
        with open('results.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([options.name, str(results)])
        logging.info("Finished zeroshot testing")
    else:
        model.eval()
        umodel = model.module if (options.distributed) else model
        image_embeddings = []
        text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch[
                    "pixel_values"]
                text_input_ids, text_attention_mask, image = input_ids.to(options.device) \
                    , attention_mask.to(options.device), \
                                                          pixel_values.to(options.device)

                image_embedding = umodel.get_image_features(image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                text_embedding = umodel.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)


                image_embeddings.append(image_embedding)
                text_embeddings.append(text_embedding)

        image_embeddings = torch.cat(image_embeddings,dim=0)
        text_embeddings = torch.cat(text_embeddings,dim=0)


        (r1, r5, r10, medr, meanr), (r1i, r5i, r10i, medri, meanri) = i2t(image_embeddings.cpu(), text_embeddings.cpu())

        # logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
        #       (r1, r5, r10, medr, meanr))
        # logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
        #       (r1i, r5i, r10i, medri, meanri))

        results = {f"i2t_R@1":r1,f"i2t_R@5":r5,f"i2t_R@10":r10,f"i2t_medr":medr,f"t2i_R@1":r1i,f"t2i_R@5":r5i,f"t2i_R@10":r10i,f"t2i_meadri":medri,}




        with open('results.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([options.name, str(results)])

    return results


def i2t(images, captions, masks=None, npts=None, measure=None, return_ranks=False,
        model=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images #图片特征
    Captions: (5N, K) matrix of captions #文本特征
    """
    if npts is None:
        npts = images.shape[0] // 5  # N
    index_list = []
    gv1_list = []
    gv2_list = []

    ranks = np.zeros(npts)  # N
    top1 = np.zeros(npts)  # N

    score_matrix = np.zeros((images.shape[0] // 5, captions.shape[0]))  # 大小是（N，5N）

    for index in range(npts):

        # Get query image

        im = images[5 * index].reshape(1, images.shape[1])
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])
        score_matrix[index] = d

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # i2t
    stat_num = 0
    minnum_rank_image = np.array([1e7] * npts)
    for i in range(npts):
        cur_rank = np.argsort(score_matrix[i])[::-1]
        for index, j in enumerate(cur_rank):
            if j in range(5 * i, 5 * i + 5):
                stat_num += 1
                minnum_rank_image[i] = index
                break

    i2t_r1 = 100.0 * len(np.where(minnum_rank_image < 1)[0]) / len(minnum_rank_image)
    i2t_r5 = 100.0 * len(np.where(minnum_rank_image < 5)[0]) / len(minnum_rank_image)
    i2t_r10 = 100.0 * len(np.where(minnum_rank_image < 10)[0]) / len(minnum_rank_image)
    i2t_medr = np.floor(np.median(minnum_rank_image)) + 1
    i2t_meanr = minnum_rank_image.mean() + 1

    # print("i2t results:", i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr)

    # t2i

    stat_num = 0
    score_matrix = score_matrix.transpose()
    minnum_rank_caption = np.array([1e7] * npts * 5)
    for i in range(5 * npts):
        img_id = i // 5
        cur_rank = np.argsort(score_matrix[i])[::-1]
        for index, j in enumerate(cur_rank):
            if j == img_id:
                stat_num += 1
                minnum_rank_caption[i] = index
                break


    t2i_r1 = 100.0 * len(np.where(minnum_rank_caption < 1)[0]) / len(minnum_rank_caption)
    t2i_r5 = 100.0 * len(np.where(minnum_rank_caption < 5)[0]) / len(minnum_rank_caption)
    t2i_r10 = 100.0 * len(np.where(minnum_rank_caption < 10)[0]) / len(minnum_rank_caption)
    t2i_medr = np.floor(np.median(minnum_rank_caption)) + 1
    t2i_meanr = minnum_rank_caption.mean() + 1

    # print("t2i results:", t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr)

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr), (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr)

class Finetune(torch.nn.Module):
    def __init__(self, input_dim, output_dim, model):
        super(Finetune, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.model  = model
    def forward(self, x):
        outputs = self.linear(self.model.get_image_features(x))
        return outputs

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def get_odim_metric(options):

    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    return output_dim, metric

def get_finetune_metrics(model, train_dataloader, test_dataloader, options):

    logging.info("Starting finetune testing")
    model.train()
    umodel = model.module if(options.distributed) else model

    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = Finetune(input_dim = input_dim, output_dim = output_dim, model = umodel).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))

    if options.checkpoint_finetune is not None:
        if(os.path.isfile(options.checkpoint_finetune)):
            checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
            if(not options.distributed and next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
            if(options.distributed and not next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {f'module.{key}': value for key, value in checkpoint.items()}
            state_dict = checkpoint["state_dict"]
            classifier.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint {options.checkpoint_finetune}")
    
    if(not options.checkpoint_finetune or not os.path.isfile(options.checkpoint_finetune)):
        for epoch in pbar:
            cbar = tqdm(train_dataloader, leave = False)
            for index, (image, label) in enumerate(cbar):
                step = len(train_dataloader) * epoch + index
                scheduler(step)
                image, label = image.to(options.device), label.to(options.device)
                logit = classifier(image)
                optimizer.zero_grad()
                loss = criterion(logit, label)
                loss.backward()
                optimizer.step()
                if options.wandb:
                    wandb.log({'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
                cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        checkpoint = {'state_dict': classifier.state_dict()}
        checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(checkpoints_dir_path, exist_ok = True)
        torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"finetune.pt"))

    classifier.eval()
    
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(image)
                prediction = torch.argmax(logits, dim = 1)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}

    logging.info("Finished finetune testing")
    return results


def get_linear_probe_metrics(model, train_dataloader, test_dataloader, options):
    logging.info("Started linear probe testing")
    logging.info(f"Number of train examples: {train_dataloader.num_samples}")
    logging.info(f"Number of test examples: {test_dataloader.num_samples}")

    model.eval()
    umodel = model.module if(options.distributed) else model
    
    images = None
    labels = None
    with torch.no_grad():
        for image, label in tqdm(train_dataloader):
            image = umodel.get_image_features(image.to(options.device)).cpu()
            images = torch.cat([images, image], dim = 0) if(images is not None) else image
            labels = torch.cat([labels, label], dim = 0) if(labels is not None) else label

    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = options.batch_size, shuffle = True)
    
    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, 0.005, 0, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for index, (image, label) in enumerate(cbar):
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            image, label = image.to(options.device), label.to(options.device)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                prediction = torch.argmax(logits, dim = 1)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
        else:
            correct = torch.zeros(output_dim).to(options.device)
            total = torch.zeros(output_dim).to(options.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                predictions = torch.argmax(logits, dim = 1)
                
                temp = torch.zeros(output_dim, len(label)).to(options.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
        
    logging.info("Finished linear probe testing")
    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["eval_test"] is not None): 
            if(data["eval_train"] is not None):
                if options.linear_probe:
                    metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
                elif options.finetune:
                    metrics.update(get_finetune_metrics(model, data["eval_train"], data["eval_test"], options))
            else:
                metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options))
        
        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics