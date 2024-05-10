import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import copy
from torch.autograd import Variable
from . import attacks
from . import utils
from transformers import BertTokenizer, BertForMaskedLM
from transformers import logging as logging_t
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import os
os.environ['CURL_CA_BUNDLE'] = ''

logging_t.set_verbosity_error()
def get_loss(umodel, outputs, criterion, options, gather_backdoor_indices):
    if (options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[
                                               :len(outputs.image_embeds) // 2], outputs.image_embeds[
                                                                                 len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[
                                                                                                  len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    if (options.distributed):
        if (options.inmodal):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in
                                               range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in
                                              range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.rank] + [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [
                augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds = torch.cat(augmented_gathered_text_embeds[:options.rank] + [
                augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.rank] + [text_embeds] + gathered_text_embeds[options.rank + 1:])

    constraint = torch.tensor(0).to(options.device)
    if options.unlearn:
        normal_indices = (~gather_backdoor_indices).nonzero().squeeze()
        backdoor_indices = gather_backdoor_indices.nonzero()
        backdoor_indices = backdoor_indices[:, 0] if len(backdoor_indices.shape) == 2 else backdoor_indices
        if len(backdoor_indices):
            backdoor_image_embeds = image_embeds[backdoor_indices]
            backdoor_text_embeds = text_embeds[backdoor_indices]
            similarity_backdoor_embeds = torch.diagonal(backdoor_image_embeds @ backdoor_text_embeds.t())
            constraint = (similarity_backdoor_embeds + options.unlearn_target).square().mean().to(options.device,
                                                                                                  non_blocking=True)
        image_embeds = image_embeds[normal_indices]
        text_embeds = text_embeds[normal_indices]

    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if (options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    batch_size = len(logits_text_per_image)
    target = torch.arange(batch_size).long().to(options.device)

    contrastive_loss = torch.tensor(0).to(options.device)
    if (options.inmodal):
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text,
                                                                                            target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(
            logits_text_per_augmented_text, target)) / 2
        # contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
        contrastive_loss = (options.clip_weight * crossmodal_contrastive_loss) + (
                    options.inmodal_weight * inmodal_contrastive_loss)
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text,
                                                                                            target)) / 2
        contrastive_loss = crossmodal_contrastive_loss

    if options.unlearn:
        contrastive_loss = contrastive_loss + (options.constraint_weight * constraint)

    loss = contrastive_loss
    return loss, contrastive_loss, constraint

def get_loss1(umodel, outputs, criterion, options, gather_backdoor_indices):
    if (options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[
                                               :len(outputs.image_embeds) // 2], outputs.image_embeds[
                                                                                 len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[
                                                                                                  len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    cos_loss = (1 - torch.nn.functional.cosine_similarity(image_embeds,text_embeds,dim=1)).mean()
    constraint = torch.tensor(0).to(options.device)

    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()


    batch_size = len(logits_text_per_image)
    target = torch.arange(batch_size).long().to(options.device)


    crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text,
                                                                                        target)) / 2
    contrastive_loss = crossmodal_contrastive_loss


    loss = contrastive_loss
    return cos_loss, contrastive_loss, constraint


def process_batch(model, batch, options, step):
    input_ids, attention_mask, pixel_values, is_backdoor = batch["input_ids"].to(options.device), \
                                                           batch["attention_mask"].to(options.device,
                                                                                      non_blocking=True), batch[
                                                               "pixel_values"].to(options.device), \
                                                           batch["is_backdoor"].to(options.device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    with torch.no_grad():
        similarity = torch.diagonal(outputs.image_embeds @ outputs.text_embeds.t())
        topmax = int(options.remove_fraction * len(similarity))
        detect_indices = similarity.topk(topmax).indices
    num_backdoor = is_backdoor.sum().item()
    backdoor_indices = is_backdoor.nonzero()
    backdoor_indices = backdoor_indices[:, 0] if len(backdoor_indices.shape) == 2 else backdoor_indices
    count = 0
    if len(backdoor_indices) > 0:
        for backdoor_index in backdoor_indices:
            count += (backdoor_index in detect_indices)
    if options.wandb and options.master:
        wandb.log({f'{options.rank}/total backdoors': num_backdoor, 'step': step})
        wandb.log({f'{options.rank}/correct backdoors detected': count, 'step': step})
    pred_backdoor_indices = torch.zeros_like(similarity).int()
    pred_backdoor_indices[detect_indices] = 1
    return outputs, pred_backdoor_indices


def clamp(X,lower_limit,upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class PerturbationTool():
    def __init__(self, seed, epsilon, num_steps, step_size, lower_limit=None,
                 upper_limit=None,num_trigger_tokens=1, options=None):
        self.epsilon = epsilon.to(options.device)
        self.num_steps = num_steps
        self.step_size = step_size.to(options.device)
        self.seed = seed
        np.random.seed(seed)
        self.lower_limit = lower_limit.to(options.device)
        self.upper_limit = upper_limit.to(options.device)
        self.num_trigger_tokens = num_trigger_tokens

    def image_loss(self,model,input_ids,attention_mask,perturb_img,criterion,pred_text_idx,eta,options):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=perturb_img)
        loss1, _, _ = get_loss(model, outputs, criterion, options, gather_backdoor_indices=None)
        loss2 = attacks.get_linear_noise_csd_loss(eta, torch.tensor(pred_text_idx))
        loss = loss1 + loss2
        return loss1

    def min_min_attack(self, batch, model, criterion, batch_token,random_noise=None, options=None):
        mean = torch.tensor([[0.48145466], [0.4578275], [0.40821073]])
        mean = mean.expand(3, 224 * 224)
        mean = mean.view(3, 224, 224)

        var = torch.tensor([[0.26862954], [0.26130258], [0.27577711]])
        var = var.expand(3, 224 * 224)
        var = var.view(3, 224, 224)

        input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        input_ids[:, 1:1 + self.num_trigger_tokens] = batch_token
        input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                , attention_mask.to(options.device), \
                                                  pixel_values.to(options.device)

        # random_noise = clamp(random_noise, -self.epsilon, self.epsilon)
        perturb_img = Variable(pixel_values.data + random_noise, requires_grad=True)
        perturb_img = Variable(clamp(perturb_img, self.lower_limit, self.upper_limit), requires_grad=True)
        eta = random_noise

        with torch.no_grad():
            input_keyword = copy.deepcopy(input_ids)
            input_keyword[:,options.token_num + 1:] = 0
            input_keyword[:,options.token_num + 1] = 49407
            text_embeds = model.get_text_features(input_ids=input_keyword, attention_mask=attention_mask)
            classifier = KMeans()
            pred_text_idx = classifier.fit_predict(text_embeds.cpu().view(32, -1).numpy())

        print("token idx",pred_text_idx)

        for _ in range(self.num_steps):
            model.zero_grad()
            loss = self.image_loss(model, input_ids, attention_mask, perturb_img, criterion, pred_text_idx, eta, options)
            perturb_img.retain_grad()
            loss.backward()
            logging.info(f"Image Loss: {loss}")
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = clamp(perturb_img.data - pixel_values.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(pixel_values.data + eta, requires_grad=True)
            perturb_img = Variable(clamp(perturb_img, self.lower_limit, self.upper_limit), requires_grad=True)
            # import matplotlib.pyplot as plt
            # plt.imshow(
            #     (perturb_img[0] * var.to(options.device) + mean.to(options.device)).permute(1, 2,
            #                                                                                 0).detach().cpu().numpy())
            # plt.show()

        #
        # with torch.no_grad():
        #     classifier = KMeans()
        #     pred_text_idx = classifier.fit_predict(eta.cpu().view(32, -1).numpy())
        #     print("image idx",pred_text_idx)


        return perturb_img, eta

def poison(epoch, model, data, optimizer, scheduler, scaler, options,processor):

    mean = torch.tensor([[0.48145466], [0.4578275], [0.40821073]])
    mean = mean.expand(3, 224 * 224)
    mean = mean.view(3, 224, 224)

    var = torch.tensor([[0.26862954], [0.26130258], [0.27577711]])
    var = var.expand(3, 224 * 224)
    var = var.view(3, 224, 224)


    dataloader = data["train"]
    if (options.distributed): dataloader.sampler.set_epoch(epoch)

    criterion = nn.CrossEntropyLoss().to(
        options.device)  # if not options.unlearn else nn.CrossEntropyLoss(reduction = 'none').to(options.device)

    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")

    #初始化图像扰动和文本扰动
    num_trigger_tokens = options.token_num
    noise = torch.zeros([30000, 3, 224, 224])
    # noise = torch.load('./noise_shuffle_sample_0910.pt')
    # noise = (torch.randint(-8, 8, [32365, 3, 224, 224], dtype=torch.int)-mean)/var
    # trigger_token_id = processor.process_text('The The The ')['input_ids'][0][1:1 + num_trigger_tokens]
    # trigger_token_ids = trigger_token_id.view(1,-1).expand(32365,3)
    trigger_token_ids = torch.randint(30522,size=(30000,options.token_num))
    trigger_token_ids = torch.LongTensor(trigger_token_ids)

    if options.init_pert is not None:
        init_root = os.path.join(options.logs, options.init_pert)
        init_noise = torch.load(os.path.join(init_root, "noise.pt"))
        # init_token = torch.load(os.path.join(init_root, "token.pt"))
        init_index = torch.load(os.path.join(init_root, "index_list.pt"))

        for i in range(noise.shape[0]):
            noise[i] = init_noise[init_index[i]]
            # trigger_token_ids[i] = init_token[init_index[i]]


    #初始化参数
    data_iter = iter(dataloader)
    # condition = 0

    upper_limit = ((1 - mean) / var)
    lower_limit = ((0 - mean) / var)

    step_size = ((1 / 255) / var)
    epsilon = ((8 / 255) / var)


    noise_generator = PerturbationTool(seed=0, epsilon=epsilon, num_steps=8, step_size=step_size,
                                      lower_limit=lower_limit, upper_limit=upper_limit,num_trigger_tokens=num_trigger_tokens,options=options)
    extracted_grads = []

    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    # add hooks for embeddings
    def add_hooks(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == 49408:  # only add a hook to wordpiece embeddings, not position
                    module.weight.requires_grad = True
                    hook = module.register_backward_hook(extract_grad_hook)
        return hook



    def get_embedding_weight(model):
        """
        Extracts and returns the token embedding weight matrix from the model.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == 49408:
                    return module.weight.cpu().detach()


    def get_embedding_module(model):
        """
        Extracts and returns the token embedding weight matrix from the model.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == 49408:
                    return module


    def text_loss(model,input_ids,attention_mask,pixel_values,pred_idx,criterion,options,balance_para=1):
        global extracted_grads
        extracted_grads = []

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss1, _, _ = get_loss1(model, outputs, criterion, options, gather_backdoor_indices=None)

        #token trigger embedding
        # input_keyword = copy.deepcopy(input_ids)
        # input_keyword[:,options.token_num + 1:] = 0
        # input_keyword[:,options.token_num + 1] = 49407
        #
        # text_embeds = model.get_text_features(input_ids=input_keyword, attention_mask=attention_mask)
        # loss2 = attacks.get_linear_noise_csd_loss(text_embeds, torch.tensor(pred_idx).data)
        # loss = loss1 #+ balance_para*loss2

        # loss.backward()

        # print('loss1:',loss1)
        # print('loss2:',loss2)

        return loss1

    def text_loss1(model,input_ids,origin_input_ids,attention_mask,pixel_values,pred_idx,criterion,options,balance_para=1):
        outputs = model.test(input_ids=input_ids, origin_input_ids=origin_input_ids,attention_mask=attention_mask, pixel_values=pixel_values)
        loss1, _, _ = get_loss1(model, outputs, criterion, options, gather_backdoor_indices=None)

        #token trigger embedding
        # input_keyword = copy.deepcopy(input_ids)
        # input_keyword[:,options.token_num + 1:] = 0
        # input_keyword[:,options.token_num + 1] = 49407

        # text_embeds = model.get_text_features(input_ids=input_keyword, attention_mask=attention_mask)
        # loss2 = attacks.get_linear_noise_csd_loss(text_embeds, torch.tensor(pred_idx).data)
        # loss = loss1 + balance_para*loss2
        # print('loss1:',loss1)
        # print('loss2:',loss2)

        return loss1


    # embedding_module = get_embedding_module(model).to(options.device)

    for _ in range(0,10):  # optimize theta for M steps
        optimizer.zero_grad()
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        # for j in range(0, 10):
        #     try:
        #         batch, index = next(data_iter)
        #     except:
        #         data_iter = iter(train_loader)
        #         batch, index = next(data_iter)
        #
        #     input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        #
        #     with torch.no_grad():
        #         pixel_values = clamp(batch["pixel_values"] + noise[index], lower_limit, upper_limit)
        #         for i, _ in enumerate(batch["pixel_values"]):
        #             input_ids[i, 1:1 + num_trigger_tokens] = trigger_token_ids[index[i]]
        #
        #     input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
        #                                         ,attention_mask.to(options.device), \
        #                                               pixel_values.to(options.device)
        #
        #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        #
        #     with autocast():
        #         loss, _, _ = get_loss(model, outputs, criterion, options,
        #                                                            gather_backdoor_indices= None)
        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)
        #
        #     scaler.update()
        #     # model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)


        idx = 0
        for _, (batch,index) in tqdm(enumerate(dataloader), total=len(dataloader)):

            # hook = add_hooks(model)
            embedding_weight = get_embedding_weight(model).to(options.device)

            # for param in model.parameters():
            #     param.requires_grad = True

            model.eval()
            input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"],batch["pixel_values"]

            with torch.no_grad():
                pixel_values = clamp(batch["pixel_values"] + noise[index], lower_limit, upper_limit)
                for i, _ in enumerate(batch["pixel_values"]):
                    # pixel_values[i] = clamp(batch["pixel_values"][i]+noise[index[i]], lower_limit, upper_limit)
                    input_ids[i, 1:1 + num_trigger_tokens] = trigger_token_ids[index[i]]

            input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                , attention_mask.to(options.device), \
                                                      pixel_values.to(options.device)
            logging.info(attacks.decode_vis(input_ids[0]))
            classifier = KMeans()
            pred_img_idx = classifier.fit_predict(noise[index].view(32, -1).numpy())


            one_hot = torch.zeros(input_ids[:,1:1+num_trigger_tokens].shape[0],input_ids[:,1:1+num_trigger_tokens].shape[1],
                                  embedding_weight.shape[0],device=options.device)

            one_hot.scatter_(2,input_ids[:,1:1+num_trigger_tokens].unsqueeze(2),
                             torch.ones((one_hot.shape[0],one_hot.shape[1],1),device=options.device))

            one_hot.requires_grad_()

            trigger_embed = (one_hot @ embedding_weight)
            input_embed = attacks.get_embeddings(model,input_ids)

            full_embed = torch.cat([input_embed[:,:1,:],
                                    trigger_embed,
                                    input_embed[:,1+num_trigger_tokens:,:]
                                    ],dim=1)

            loss = text_loss1(model, full_embed, input_ids, attention_mask, pixel_values, pred_img_idx, criterion, options)
            loss.backward()
            logging.info("*"*60)
            logging.info(f"Text Loss: {loss}")
            model.zero_grad()

            grad = one_hot.grad.clone()
            print(grad.shape)
            # tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Administrator\.cache\huggingface\hub\models--bert-base-uncased\snapshots\1dbc166cf8765166998eff31ade2eb64c8a40076')
            # bert = BertForMaskedLM.from_pretrained(r'C:\Users\Administrator\.cache\huggingface\hub\models--bert-base-uncased\snapshots\1dbc166cf8765166998eff31ade2eb64c8a40076')
            # bert.eval()
            # bert.to(options.device)

            # decode_text = attacks.decode_text(input_ids,options.token_num)
            # process_text = attacks.process_text(decode_text, options, tokenizer)
            # input_ids_copy = input_ids.clone()
            # loss_ori = loss
            # cache_trigger_token = torch.zeros_like(trigger_token_ids[index,:])

            # for token_index in range(1,1+options.token_num):
            #     # predict_list = attacks.predict_text(process_text,token_index,tokenizer,bert,options,num_cand=2000)
            #     # cand_word = attacks.encode_text(predict_list)
            #     # cand_word = torch.tensor(cand_word).to(options.device)
            #     # embedding_weight = embedding_module(cand_word)
            #     cand_index = attacks.hotflip_attack_new(grad[:,token_index,:], embedding_weight,
            #                                     increase_loss=False, num_trigger_tokens=num_trigger_tokens)
            #     selected_values = cand_index
            #
            #     # selected_values = torch.gather(cand_word, 1, cand_index.to(options.device).unsqueeze(1))
            #     # cand_word = attacks.decode(selected_values)
            #     # process_text = attacks.change_word(process_text, cand_word, tokenizer, token_index)
            #     cache_trigger_token[:, token_index - 1] = selected_values.cpu()#.squeeze(1).cpu()
            #
            # # test loss
            #     with torch.no_grad():
            #         input_ids_copy[:, 1:1 + num_trigger_tokens] = cache_trigger_token
            #         loss_cand = text_loss1(model, input_ids_copy, attention_mask, pixel_values, pred_img_idx, criterion, options)
            #         logging.info(f"Text Loss cand: {loss_cand}")
            #
            #     # if loss_cand < loss_ori:
            #         trigger_token_ids[index] = cache_trigger_token

            # input_ids[:, 1:1 + num_trigger_tokens] = trigger_token_ids[index]
            # print(input_ids[0])

            top_indices = (-grad).topk(20, dim=-1).indices
            print(top_indices.shape)

            for i in range(20):
                input_ids[:, 1:1 + num_trigger_tokens] = top_indices[:,:,i]
                with torch.no_grad():
                    loss = text_loss(model, input_ids, attention_mask, pixel_values, pred_img_idx, criterion,
                                      options)
                    logging.info(f"Text Loss: {loss}")
                    logging.info(attacks.decode_vis(input_ids[0]))
                    logging.info("=" * 60)
            logging.info("aa" * 60)

            # with torch.no_grad():
            #     trigger_embed = (grad @ embedding_weight)
            #     full_embed = torch.cat([input_embed[:, :1, :],
            #                             trigger_embed,
            #                             input_embed[:, 1 + num_trigger_tokens:, :]
            #                             ], dim=1)
            #     loss = text_loss1(model, full_embed, input_ids, attention_mask, pixel_values, pred_img_idx, criterion, options)





            # Perturbation over entire dataset
            for param in model.parameters():
                param.requires_grad = False

            batch_start_idx, batch_noise,batch_token = idx, [],[]

            for i, _ in enumerate(batch["pixel_values"]):
                # Update noise to images
                batch_noise.append(noise[index[i]])
                batch_token.append(trigger_token_ids[index[i]])
                idx += 1

            batch_noise = torch.stack(batch_noise).to(options.device)
            batch_token = torch.stack(batch_token)

            # # # Update sample-wise perturbation
            model.eval()
            perturb_img, eta = noise_generator.min_min_attack(batch,model,criterion,
                                                              batch_token,
                                                              random_noise=batch_noise,
                                                              options=options)
            for i, delta in enumerate(eta):
                noise[index[i]] = delta.clone().detach().cpu()


    return noise,trigger_token_ids

