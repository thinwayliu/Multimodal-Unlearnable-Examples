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
                 upper_limit=None, options=None):
        self.epsilon = epsilon.to(options.device)
        self.num_steps = num_steps
        self.step_size = step_size.to(options.device)
        self.seed = seed
        np.random.seed(seed)
        self.lower_limit = lower_limit.to(options.device)
        self.upper_limit = upper_limit.to(options.device)


    def min_min_attack(self, batch, model, criterion,random_noise=None, options=None):
        mean = torch.tensor([[0.48145466], [0.4578275], [0.40821073]])
        mean = mean.expand(3, 224 * 224)
        mean = mean.view(3, 224, 224)

        var = torch.tensor([[0.26862954], [0.26130258], [0.27577711]])
        var = var.expand(3, 224 * 224)
        var = var.view(3, 224, 224)

        input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                , attention_mask.to(options.device), \
                                                  pixel_values.to(options.device)
        # random_noise = clamp(random_noise, -self.epsilon, self.epsilon)
        perturb_img = Variable(pixel_values.data + random_noise, requires_grad=True)
        perturb_img = Variable(clamp(perturb_img, self.lower_limit, self.upper_limit), requires_grad=True)
        eta = random_noise


        for _ in range(self.num_steps):
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=perturb_img)
            loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
            perturb_img.retain_grad()
            loss.backward()
            logging.info(f"Image Loss: {loss}")
            eta = self.step_size * perturb_img.grad.data.sign() * (1)
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
    #初始化图像扰动和文本扰动
    if dataloader.num_samples<=30000:
        data_samples_num = 30000
    else:
        data_samples_num = 145000
    noise = torch.zeros([data_samples_num, 3, 224, 224])

    #初始化参数


    upper_limit = ((1 - mean) / var)
    lower_limit = ((0 - mean) / var)

    step_size = ((1 / 255) / var)
    epsilon = ((8 / 255) / var)


    noise_generator = PerturbationTool(seed=0, epsilon=epsilon, num_steps=8, step_size=step_size,
                                      lower_limit=lower_limit, upper_limit=upper_limit,options=options)


    idx = 0
    for _, (batch,index) in tqdm(enumerate(dataloader), total=len(dataloader)):

        # Perturbation over entire dataset
        for param in model.parameters():
            param.requires_grad = False

        batch_start_idx, batch_noise= idx, []

        for i, _ in enumerate(batch["pixel_values"]):
            # Update noise to images
            batch_noise.append(noise[index[i]])
            idx += 1

        batch_noise = torch.stack(batch_noise).to(options.device)

        # checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
        # state_dict = checkpoint["state_dict"]
        # model.load_state_dict(state_dict)

        # # # Update sample-wise perturbation
        model.eval()
        perturb_img, eta = noise_generator.min_min_attack(batch,model,criterion,
                                                          random_noise=batch_noise,
                                                          options=options)
        for i, delta in enumerate(eta):
            noise[index[i]] = delta.clone().detach().cpu()


    return noise

