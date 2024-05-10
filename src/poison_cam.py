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
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch



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
    criterion = nn.CrossEntropyLoss().to(
        options.device)  # if not options.unlearn else nn.CrossEntropyLoss(reduction = 'none').to(options.device)
    
    model.train()
    grad_block=[]
    feature_block=[]
    extracted_grads = []
    
    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    # add hooks for embeddings
    def add_hooks(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == 49408:  # only add a hook to wordpiece embeddings, not position
                    module.weight.requires_grad = True
                    # print('*^^^^^^^^^^^^^^^^^')
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

    # embedding_weight = get_embedding_weight(model)
    
    ## module选择的模块，grad_in:moudle的输入；grad_out:module的输出
    def backward_hook(module,grad_in,grad_out):
        # print(grad_out)
        grad_block.append(grad_out[0].detach())
        # grad_block.append(grad_out[1].detach())

    def forward_hook(module,input,output):
        # print(output)
        feature_block.append(output)
        # feature_block.append(output[1])
        
    def cam_show_img(img, feature_map, grads):
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加
        print(cam.shape)
        grads = grads.reshape([grads.shape[0], -1])
        # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
        weights = np.mean(grads, axis=1)	
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]	# 特征图加权和
            
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (224, 224))
        
        # cam.dim=2 heatmap.dim=3
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)	# 伪彩色
        cam_img = 0.3 * heatmap + 0.7 * img

        cv2.imwrite("cam_TAP.jpg", cam_img)

        
    model.visual.layer4[-1].register_full_backward_hook(backward_hook)
    ##正向传播时自动执行farward_hook
    model.visual.layer4[-1].register_forward_hook(forward_hook)

    for param in model.parameters():
            param.requires_grad = True
            
    #for _, (batch,index) in tqdm(enumerate(dataloader), total=len(dataloader)):
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        add_hooks(model)
        input_ids, attention_mask, pixel_values = batch["input_ids"], batch["attention_mask"], batch["pixel_values"]
        input_ids, attention_mask, pixel_values = input_ids.to(options.device) \
                                                , attention_mask.to(options.device), \
                                                  pixel_values.to(options.device)
        model.zero_grad() 
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)                         

        
                
        loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
        loss.backward(retain_graph=True)

        grads_val = grad_block[0][0].cpu().data.numpy().squeeze()
        fmap = feature_block[0][0].cpu().data.numpy().squeeze()

    
        raw_img= cv2.cvtColor((pixel_values[0] * var.to(options.device) + mean.to(options.device)).permute(1, 2,0).detach().cpu().numpy()*255, cv2.COLOR_BGR2RGB)
        import matplotlib.pyplot as plt
        plt.imshow(
            (pixel_values[0] * var.to(options.device) + mean.to(options.device)).permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('orig.png')
        
        cam_show_img(raw_img, fmap, grads_val)
        
        # print(extracted_grads)
        grads = extracted_grads[0][0].cpu().data.numpy()
        print(grads.shape)
        # grads = np.mean(grads, axis=0)
        weights = np.sqrt(np.square(grads).sum(axis=1))
        print(weights)
        print(attacks.decode2(input_ids)[0])
        # if scale:
        # weights = (weights - weights.min()) / (weights.max() - weights.min())
        # print(results[1:-1])

        # 取绝对值后再取最大
        # scores = torch.abs(pred_grads).max(axis=1)[0]
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        # scores = scores.round(4)
        # results = [(attacks.decode_text(t.item(),0), s.item()) for t, s in zip(input_ids, scores)]
        # print(results[1:-1])
        # from .textcolor import print_color_text
        # print_color_text('two people are on the side of a glass building .', weights)
        # print(" =>", id_to_classes[y_pred_id])

        # 词堆组成的句子
        # sentence = "<start_of_text>,two ,people ,are ,on ,the ,side ,of ,a ,glass ,building ,. ,<end_of_text>,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!,!"
        # words = sentence.split(',')

        # # 词堆中每个词的重要性
        # importance =  np.array([0.00530903, 0.00695095, 0.00698094, 0.00675391, 0.01102503,
        #                0.00811789, 0.0140195, 0.00656614, 0.0059219, 0.02200382,
        #                0.0131144, 0.00086892, 0.00821449] + [0]*64)
        # # 绘图
        # words = ['two', 'people', 'are', 'on', 'the', 'side', 'of', 'a', 'glass', 'building', '.']
        # importance = np.array([0.00695095, 0.00698094, 0.00675391, 0.01102503, 0.00811789,
        #                     0.0140195, 0.00656614, 0.0059219, 0.02200382, 0.0131144, 0.00086892])
        # plt.figure()
        # plt.axis('off')

        # # 根据重要性绘制颜色条
        # for i, (word, imp) in enumerate(zip(words, importance)):
        #     color_value = imp
        #     plt.text(i, 0.5, word, color=plt.cm.viridis(color_value), fontsize=12, ha='center')

        # # 保存图像
        # plt.savefig('text_importance_image.png', bbox_inches='tight')

        break
        
                                                  
                                             


    