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
import os 


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

    def min_min_attack(self, batch, model, criterion, batch_token,random_noise=None, options=None):
        mean = torch.tensor([[0.48145466], [0.4578275], [0.40821073]])
        mean = mean.expand(3, 224 * 224)
        mean = mean.view(3, 224, 224)

        var = torch.tensor([[0.26862954], [0.26130258], [0.27577711]])
        var = var.expand(3, 224 * 224)
        var = var.view(3, 224, 224)

        input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        
        # 使用索引对张量进行重新排序
        # shuffled_indices = torch.randperm(input_ids.shape[0])
        # input_ids = input_ids[shuffled_indices]
        
        # input_ids[:, 1:1 + self.num_trigger_tokens] = batch_token
        input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                , attention_mask.to(options.device), \
                                                  pixel_values.to(options.device)
        # random_noise = clamp(random_noise, -self.epsilon, self.epsilon)
        perturb_img = Variable(pixel_values.data + random_noise, requires_grad=True)
        perturb_img = Variable(clamp(perturb_img, self.lower_limit, self.upper_limit), requires_grad=True)
        eta = random_noise
        
        input_ids_copy = input_ids.clone()

        for _ in range(self.num_steps):
            model.zero_grad()
            
            shuffled_indices = torch.randperm(input_ids.shape[0])
            input_ids_copy = input_ids[shuffled_indices]
            input_ids_copy[:, 1:1 + self.num_trigger_tokens] = batch_token
            
            outputs = model(input_ids=input_ids_copy, attention_mask=attention_mask, pixel_values=perturb_img)
            loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
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
    if dataloader.num_samples<=30000:
        data_samples_num = 30000
    else:
        data_samples_num = 145000
        
    
    noise = torch.zeros([data_samples_num, 3, 224, 224])
    trigger_token_ids = torch.randint(30522,size=(data_samples_num,options.token_num))
    trigger_token_ids = torch.LongTensor(trigger_token_ids)

    #初始化参数
    data_iter = iter(dataloader)
    # condition = 0

    upper_limit = ((1 - mean) / var)
    lower_limit = ((0 - mean) / var)

    step_size = ((1 / 255) / var)
    epsilon = ((8 / 255) / var)
    num_steps = 8


    noise_generator = PerturbationTool(seed=0, epsilon=epsilon, num_steps=num_steps, step_size=step_size,
                                      lower_limit=lower_limit,
                                       upper_limit=upper_limit,
                                       num_trigger_tokens=num_trigger_tokens,options=options)
    extracted_grads = []

    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    # add hooks for embeddings
    def add_hooks(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == 49408:  # only add a hook to wordpiece embeddings, not position
                    module.weight.requires_grad = True
                    module.register_backward_hook(extract_grad_hook)

    def remove_hooks(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding) and hasattr(module, "_backward_hooks"):
                if module.weight.shape[0] == 49408:
                    module._backward_hooks.clear()


    def get_embedding_weight(model):
        """
        Extracts and returns the token embedding weight matrix from the model.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == 49408:
                    return module.weight.detach()

    embedding_weight = get_embedding_weight(model)
    # loss = 10
    # times = 0
    
    for _ in range(0,20):
        optimizer.zero_grad()
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        for j in range(0, 200):
            try:
                batch, index = next(data_iter)
            except:
                data_iter = iter(dataloader)
                batch, index = next(data_iter)

            input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch["pixel_values"]

            with torch.no_grad():
                pixel_values = clamp(batch["pixel_values"] + noise[index], lower_limit, upper_limit)
                for i, _ in enumerate(batch["pixel_values"]):
                    input_ids[i, 1:1 + num_trigger_tokens] = trigger_token_ids[index[i]]

            input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                ,attention_mask.to(options.device), \
                                                      pixel_values.to(options.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            with autocast():
                loss, _, _ = get_loss(model, outputs, criterion, options,
                                                                   gather_backdoor_indices= None)
                scaler.scale(loss).backward()
                scaler.step(optimizer)

            scaler.update()
            # model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)


        idx = 0
        for _, (batch,index) in tqdm(enumerate(dataloader), total=len(dataloader)):

            add_hooks(model)

            for param in model.parameters():
                param.requires_grad = True

            model.eval()
            input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"],batch["pixel_values"]
            with torch.no_grad():
                 # 使用索引对张量进行重新排序
                shuffled_indices = torch.randperm(pixel_values.shape[0])
                pixel_values = pixel_values[shuffled_indices]
                pixel_values = clamp(batch["pixel_values"] + noise[index], lower_limit, upper_limit)
                for i, _ in enumerate(batch["pixel_values"]):
                    # pixel_values[i] = clamp(batch["pixel_values"][i]+noise[index[i]], lower_limit, upper_limit)
                    input_ids[i, 1:1 + num_trigger_tokens] = trigger_token_ids[index[i]]

            input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                , attention_mask.to(options.device), \
                                                      pixel_values.to(options.device)

            model.zero_grad()
            # global extracted_grads
            extracted_grads = []
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
            logging.info("*"*60)
            logging.info(f"Text Loss: {loss}")
            loss.backward()
            grad = extracted_grads[0]
            # decode_text = attacks.decode_text(input_ids,options.token_num)
            logging.info(attacks.decode_vis(input_ids[0]))
            candidates = attacks.hotflip_attack(grad, embedding_weight,
                                                increase_loss=False, num_candidates=16,
                                                num_trigger_tokens=num_trigger_tokens)
            optimize_trigger_token = utils.get_best_candidates(model,
                                                                 input_ids, attention_mask, pixel_values,
                                                                 trigger_token_ids[index],
                                                                 candidates, num_trigger_tokens,criterion=criterion,
                                                                 increase_loss=False,options=options)

            trigger_token_ids[index] = optimize_trigger_token
            input_ids[:, 1:1 + num_trigger_tokens] = trigger_token_ids[index]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                loss, _, _ = get_loss(model, outputs, criterion, options, gather_backdoor_indices=None)
            logging.info(f"Text Loss: {loss}")
            logging.info(attacks.decode_vis(input_ids[0]))
            logging.info("="*60)


            remove_hooks(model)

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
                
        torch.save(noise, os.path.join(options.log_dir_path, "noise.pt"))
        torch.save(trigger_token_ids, os.path.join(options.log_dir_path, "token.pt"))


    return noise,trigger_token_ids

