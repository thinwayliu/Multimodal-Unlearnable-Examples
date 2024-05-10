from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
# from allennlp.common.util import lazy_groups_of
# from allennlp.data.iterators import BucketIterator, BasicIterator
# from allennlp.nn.util import move_to_device
# from allennlp.modules.text_field_embedders import TextFieldEmbedder
# from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from torch.autograd import Variable

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
    target = torch.arange(batch_size).long().to(options.device, non_blocking=True)

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

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_embedding_weight(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                embedding_weight = module._token_embedders[embed].weight.cpu().detach()
    return embedding_weight

# hook used in add_hooks()
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def add_hooks(language_model):
    for module in language_model.text_encoder.modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.requires_grad = True
            module.register_backward_hook(extract_grad_hook)

# def evaluate_batch(model, batch, trigger_token_ids=None, snli=False):
#     """
#     Takes a batch of classification examples (SNLI or SST), and runs them through the model.
#     If trigger_token_ids is not None, then it will append the tokens to the input.
#     This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
#     """
#     batch = move_to_device(batch[0], cuda_device=0)
#     if trigger_token_ids is None:
#         if snli:
#             model(batch['premise'], batch['hypothesis'], batch['label'])
#         else:
#             model(batch['tokens'], batch['label'])
#         return None
#     else:
#         trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
#         trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['label']), 1).cuda()
#         if snli:
#             original_tokens = batch['hypothesis']['tokens'].clone()
#             batch['hypothesis']['tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1)
#             output_dict = model(batch['premise'], batch['hypothesis'], batch['label'])
#             batch['hypothesis']['tokens'] = original_tokens
#         else:
#             original_tokens = batch['tokens']['tokens'].clone()
#             batch['tokens']['tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1)
#             output_dict = model(batch['tokens'], batch['label'])
#             batch['tokens']['tokens'] = original_tokens
#         return output_dict

def get_average_grad(model, batch, trigger_token_ids, target_label=None, snli=False):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch[0]['label'].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[0]['label'] = int(target_label) * torch.ones_like(batch[0]['label']).cuda()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss']
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].cpu()
    batch[0]['label'] = original_labels # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[0:len(trigger_token_ids)] # return just trigger grads
    return averaged_grad

# def get_accuracy(model, dev_dataset, vocab, trigger_token_ids=None, snli=False):
#     """
#     When trigger_token_ids is None, gets accuracy on the dev_dataset. Otherwise, gets accuracy with
#     triggers prepended for the whole dev_dataset.
#     """
#     model.get_metrics(reset=True)
#     model.eval() # model should be in eval() already, but just in case
#     if snli:
#         iterator = BucketIterator(batch_size=128, sorting_keys=[("premise", "num_tokens")])
#     else:
#         iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
#     iterator.index_with(vocab)
#     if trigger_token_ids is None:
#         for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
#             evaluate_batch(model, batch, trigger_token_ids, snli)
#         print("Without Triggers: " + str(model.get_metrics()['accuracy']))
#     else:
#         print_string = ""
#         for idx in trigger_token_ids:
#             print_string = print_string + vocab.get_token_from_index(idx) + ', '
#
#         for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
#             evaluate_batch(model, batch, trigger_token_ids, snli)
#         print("Current Triggers: " + print_string + " : " + str(model.get_metrics()['accuracy']))

def get_best_candidates(model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,snli=False,criterion=None, beam_size=1,
                        increase_loss=False,options=None):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(0, model, input_ids, attention_mask, pixel_values,  trigger_token_ids,
                                                cand_trigger_token_ids, num_trigger_tokens,
                                                criterion=criterion,snli=snli,options=options)
    # maximize the loss 从四个cand中选出一个
    if increase_loss == False:
        top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
    else:
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    # top_candidates now contains beam_size trigger sequences, each with a different 0th token

    for idx in range(1,num_trigger_tokens): # for all trigger tokens, skipping the 0th (we did it above)
        # loss_per_candidate = []
        # for cand, _ in top_candidates: # for all the beams, try all the candidates at idx 只有一次循环
        cand, _ = top_candidates[0]
        loss_per_candidate = get_loss_per_candidate(idx, model, input_ids, attention_mask, pixel_values,  cand,
                                                cand_trigger_token_ids, num_trigger_tokens,
                                                criterion=criterion,snli=snli,options=options)
        if increase_loss == False:
            top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
        else:
            top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate(index, model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,criterion, snli=False,options=None):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    '修改单词'

    input_ids[:,1:1+num_trigger_tokens] = trigger_token_ids.to(options.device, non_blocking=True)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
    curr_loss = loss.cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(cand_trigger_token_ids.shape[-1]):
        input_ids_cand = deepcopy(input_ids)
        trigger_token_test = deepcopy(trigger_token_ids)
        input_ids_cand[:,index+1] = cand_trigger_token_ids[:,index,cand_id]
        trigger_token_test[:,index] = cand_trigger_token_ids[:,index,cand_id]
        with torch.no_grad():
            outputs = model(input_ids=input_ids_cand, attention_mask=attention_mask, pixel_values=pixel_values)
            loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
        loss_per_candidate.append((deepcopy(trigger_token_test), loss.cpu().detach().numpy()))

    return loss_per_candidate


def get_loss_per_candidate_bert(index, model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,criterion, snli=False,options=None):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    # cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    '修改单词'

    input_ids[:,1:1+num_trigger_tokens] = trigger_token_ids.to(options.device, non_blocking=True)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
    curr_loss = loss.cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(cand_trigger_token_ids.shape[-1]):
        input_ids_cand = deepcopy(input_ids)
        trigger_token_test = deepcopy(trigger_token_ids)
        input_ids_cand[:,index+1] = cand_trigger_token_ids[:,index,cand_id]
        trigger_token_test[:,index] = cand_trigger_token_ids[:,index,cand_id]
        with torch.no_grad():
            outputs = model(input_ids=input_ids_cand, attention_mask=attention_mask, pixel_values=pixel_values)
            loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
        loss_per_candidate.append((deepcopy(trigger_token_test), loss.cpu().detach().numpy()))

    return loss_per_candidate




# def get_loss_per_candidate_batch(index, model, batch, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens, snli=False):
#     """
#     For a particular index, the function tries all of the candidate tokens for that index.
#     The function returns a list containing the candidate triggers it tried, along with their loss.
#     """
#     # if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
#     #     print("Only 1 candidate for index detected, not searching")
#     #     return trigger_token_ids
#     # model.get_metrics(reset=True)
#     cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
#     loss_per_candidate = []
#     # loss for the trigger without trying the candidates
#     '修改单词'
#     batch_base = deepcopy(batch)
#     batch_base["input_ids"][:,1:1+num_trigger_tokens] = trigger_token_ids
#     curr_loss = model(batch_base).cpu().detach().numpy()
#     loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
#     for cand_id in range(cand_trigger_token_ids.shape[-1]):
#         batch_test = deepcopy(batch_base)
#         trigger_token_test = deepcopy(trigger_token_ids)
#         batch_test["input_ids"][:,index+1] = cand_trigger_token_ids[:,index,cand_id]
#         trigger_token_test[:,index] = cand_trigger_token_ids[:,index,cand_id]
#         loss = model(batch_test).cpu().detach().numpy()
#         loss_per_candidate.append((deepcopy(trigger_token_test), loss))

#     return loss_per_candidate


# def get_best_candidates__(model, batch, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,snli=False, beam_size=1,
#                         increase_loss=False):
#     """"
#     Given the list of candidate trigger token ids (of number of trigger words by number of candidates
#     per word), it finds the best new candidate trigger.
#     This performs beam search in a left to right fashion.
#     """
#     # first round, no beams, just get the loss for each of the candidates in index 0.
#     # (indices 1-end are just the old trigger)
#     top_candidates_list = []
#     for i in range(batch['image'].shape[0]):
#         sample = {}
#         sample['image'] = batch['image'][i].unsqueeze(0)
#         sample["input_ids"] = batch["input_ids"][i].unsqueeze(0)
#         sample["attention_mask"] = batch["attention_mask"][i].unsqueeze(0)
#         loss_per_candidate = get_loss_per_candidate__(0, model, sample, trigger_token_ids[i],
#                                                     cand_trigger_token_ids[i], num_trigger_tokens,snli)
#         # maximize the loss 从四个cand中选出一个
#         if increase_loss == False:
#             top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
#         else:
#             top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
#         # top_candidates now contains beam_size trigger sequences, each with a different 0th token

#         for idx in range(1,num_trigger_tokens): # for all trigger tokens, skipping the 0th (we did it above)
#             loss_per_candidate = []
#             # for cand, _ in top_candidates: # for all the beams, try all the candidates at idx 只有一次循环
#             cand, _ = top_candidates[0]
#             loss_per_candidate.extend(get_loss_per_candidate__(idx, model, sample, cand,
#                                                              cand_trigger_token_ids[i], num_trigger_tokens,snli))
#             if increase_loss == False:
#                 top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
#             else:
#                 top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
#             # print(loss_per_candidate)
#         # print(max(top_candidates, key=itemgetter(1))[0].shape)
#         top_candidates_list.append(max(top_candidates, key=itemgetter(1))[0])

#     return torch.stack(top_candidates_list)

# from torch import nn
# import torch.nn.functional as F
# KLDloss = nn.KLDivLoss(reduction='sum')


# def get_loss_per_candidate__(index, model, batch, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens, snli=False):
#     """
#     For a particular index, the function tries all of the candidate tokens for that index.
#     The function returns a list containing the candidate triggers it tried, along with their loss.
#     """
#     # if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
#     #     print("Only 1 candidate for index detected, not searching")
#     #     return trigger_token_ids
#     # model.get_metrics(reset=True)
#     cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).cuda()
#     loss_per_candidate = []
#     # loss for the trigger without trying the candidates
#     '修改单词'
#     batch_base = deepcopy(batch)
#     batch_base["input_ids"][:,1:1+num_trigger_tokens] = trigger_token_ids
#     # curr_loss = model(batch_base).cpu().detach().numpy()
#     im_e = model.image_projection(model.image_encoder(batch_base['image']))
#     tx_e = model.text_projection(model.text_encoder(batch_base['input_ids'], batch_base["attention_mask"]))
#     curr_loss = KLDloss(F.log_softmax(tx_e, dim=-1), F.softmax(im_e.data, dim=-1))
#     loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

#     for cand_id in range(cand_trigger_token_ids.shape[-1]):
#         batch_test = deepcopy(batch_base)
#         trigger_token_test = deepcopy(trigger_token_ids)
#         batch_test["input_ids"][:,index+1] = cand_trigger_token_ids[index,cand_id]
#         trigger_token_test[index] = cand_trigger_token_ids[index,cand_id]
#         tx_e = model.text_projection(model.text_encoder(batch_test['input_ids'], batch_test["attention_mask"]))
#         loss = KLDloss(F.log_softmax(tx_e, dim=-1), F.softmax(im_e.data, dim=-1))
#         #loss = model(batch_test).cpu().detach().numpy()
#         loss_per_candidate.append((deepcopy(trigger_token_test), loss))

#     return loss_per_candidate



# def get_loss_per_candidate_batch__(index, model, batch, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens, snli=False,options=None):
#     """
#     For a particular index, the function tries all of the candidate tokens for that index.
#     The function returns a list containing the candidate triggers it tried, along with their loss.
#     """
#     # if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
#     #     print("Only 1 candidate for index detected, not searching")
#     #     return trigger_token_ids
#     # model.get_metrics(reset=True)
#     cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
#     loss_per_candidate = []
#     # loss for the trigger without trying the candidates
#     '修改单词'
#     batch_base = deepcopy(batch)
#     batch_base["input_ids"][:,1:1+num_trigger_tokens] = trigger_token_ids



#     curr_loss = model(batch_base).cpu().detach().numpy()
#     loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
#     for cand_id in range(cand_trigger_token_ids.shape[-1]):
#         batch_test = deepcopy(batch_base)
#         trigger_token_test = deepcopy(trigger_token_ids)
#         print(cand_trigger_token_ids.shape)
#         print(batch_test["input_ids"].shape)
#         print(trigger_token_test.shape)
#         batch_test["input_ids"][:,index+1] = cand_trigger_token_ids[index,cand_id]
#         trigger_token_test[index] = cand_trigger_token_ids[index,cand_id]
#         loss = model(batch_test).cpu().detach().numpy()
#         loss_per_candidate.append((deepcopy(trigger_token_test), loss))

#     return loss_per_candidate





def get_best_candidates_batch(model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,snli=False,criterion=None, beam_size=1,
                        increase_loss=False,options=None):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate_batch(0, model, input_ids, attention_mask, pixel_values,  trigger_token_ids,
                                                cand_trigger_token_ids, num_trigger_tokens,
                                                criterion=criterion,snli=snli,options=options)
    # maximize the loss 从四个cand中选出一个
    for sample_index in range(input_ids.shape[0]):
        if increase_loss == False:
            top_candidates = heapq.nsmallest(beam_size, loss_per_candidate[sample_index], key=itemgetter(1))
        else:
            top_candidates = heapq.nlargest(beam_size, loss_per_candidate[sample_index], key=itemgetter(1))
        print(top_candidates.shape)
    # top_candidates now contains beam_size trigger sequences, each with a different 0th token

    for idx in range(1,num_trigger_tokens): # for all trigger tokens, skipping the 0th (we did it above)
        # loss_per_candidate = []
        # for cand, _ in top_candidates: # for all the beams, try all the candidates at idx 只有一次循环
        cand, _ = top_candidates[0]
        loss_per_candidate = get_loss_per_candidate_batch(idx, model, input_ids, attention_mask, pixel_values,  cand,
                                                cand_trigger_token_ids, num_trigger_tokens,
                                                criterion=criterion,snli=snli,options=options)
        if increase_loss == False:
            top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
        else:
            top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate_batch(index, model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,criterion, snli=False,options=None):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    '修改单词'
    
    for sample_index in range(input_ids.shape[0]):
        sample_candidate = []
        input_ids[sample_index,1:1+num_trigger_tokens] = trigger_token_ids[sample_index].to(options.device, non_blocking=True)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
        curr_loss = loss.cpu().detach().numpy()
        sample_candidate.append((deepcopy(trigger_token_ids[sample_index]), curr_loss))

        for cand_id in range(cand_trigger_token_ids.shape[-1]):
            input_ids_cand = deepcopy(input_ids)
            trigger_token_test = deepcopy(trigger_token_ids)
            input_ids_cand[sample_index,index+1] = cand_trigger_token_ids[sample_index,index,cand_id]
            trigger_token_test[sample_index,index] = cand_trigger_token_ids[sample_index,index,cand_id]
            with torch.no_grad():
                outputs = model(input_ids=input_ids_cand, attention_mask=attention_mask, pixel_values=pixel_values)
                loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
            sample_candidate.append((deepcopy(trigger_token_test[sample_index]), loss.cpu().detach().numpy()))
        loss_per_candidate.append(sample_candidate)
    return loss_per_candidate



# def get_best_candidates_attention(model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,snli=False,criterion=None, beam_size=1,
#                         increase_loss=False,options=None,images_embeds = None):
#     """"
#     Given the list of candidate trigger token ids (of number of trigger words by number of candidates
#     per word), it finds the best new candidate trigger.
#     This performs beam search in a left to right fashion.
#     """
#     # first round, no beams, just get the loss for each of the candidates in index 0.
#     # (indices 1-end are just the old trigger)
#     loss_per_candidate = get_loss_per_candidate_attention(0, model, input_ids, 
#                                                 attention_mask, pixel_values,  trigger_token_ids,
#                                                 cand_trigger_token_ids, num_trigger_tokens,
#                                                 criterion=criterion,snli=snli,
#                                                 options=options,images_embeds = images_embeds)
#     # maximize the loss 从四个cand中选出一个
#     if increase_loss == False:
#         top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
#     else:
#         top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
#     # top_candidates now contains beam_size trigger sequences, each with a different 0th token

#     for idx in range(1,num_trigger_tokens): # for all trigger tokens, skipping the 0th (we did it above)
#         # loss_per_candidate = []
#         # for cand, _ in top_candidates: # for all the beams, try all the candidates at idx 只有一次循环
#         cand, _ = top_candidates[0]
#         loss_per_candidate = get_loss_per_candidate_attention(idx, model, input_ids, attention_mask, pixel_values,  cand,
#                                                 cand_trigger_token_ids, num_trigger_tokens,
#                                                 criterion=criterion,snli=snli,options=options,
#                                                 images_embeds = images_embeds)
#         if increase_loss == False:
#             top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
#         else:
#             top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

#     return max(top_candidates, key=itemgetter(1))[0]


# def get_loss_per_candidate_attention(index, model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,criterion, snli=False,options=None,images_embeds = None):
#     """
#     For a particular index, the function tries all of the candidate tokens for that index.
#     The function returns a list containing the candidate triggers it tried, along with their loss.
#     """
#     cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
#     loss_per_candidate = []
#     # loss for the trigger without trying the candidates
#     '修改单词'

#     input_ids[:,1:1+num_trigger_tokens] = trigger_token_ids.to(options.device, non_blocking=True)
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
#         loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
#         input_ids_copy = input_ids.clone()
#         input_ids_copy[:,num_trigger_tokens + 1:] = 0
#         input_ids_copy[:,num_trigger_tokens + 1] = 49407
#         text_embeds = model.get_text_features(input_ids=input_ids_copy, attention_mask=attention_mask)
#         loss += 0.5*attacks.sim_loss(images_embeds,text_embeds.data)
        
#     curr_loss = loss.cpu().detach().numpy()
#     loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

#     for cand_id in range(cand_trigger_token_ids.shape[-1]):
#         input_ids_cand = deepcopy(input_ids)
#         trigger_token_test = deepcopy(trigger_token_ids)
#         input_ids_cand[:,index+1] = cand_trigger_token_ids[:,index,cand_id]
#         trigger_token_test[:,index] = cand_trigger_token_ids[:,index,cand_id]
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids_cand, attention_mask=attention_mask, pixel_values=pixel_values)
#             loss, _, _ = get_loss(model, outputs, criterion, options,gather_backdoor_indices=None)
#             input_ids_copy = input_ids.clone()
#             input_ids_copy[:,num_trigger_tokens + 1:] = 0
#             input_ids_copy[:,num_trigger_tokens + 1] = 49407
#             text_embeds = model.get_text_features(input_ids=input_ids_copy, attention_mask=attention_mask)
#             loss += 0.5*attacks.sim_loss(images_embeds,text_embeds.data)
#         loss_per_candidate.append((deepcopy(trigger_token_test), loss.cpu().detach().numpy()))

#     return loss_per_candidate



def get_best_candidates_max(model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,snli=False,criterion=None, beam_size=1,
                        increase_loss=False,options=None,normal_embeds=None):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate_max(0, model, input_ids, attention_mask, pixel_values,  trigger_token_ids,
                                                cand_trigger_token_ids, num_trigger_tokens,
                                                criterion=criterion,snli=snli,options=options,normal_embeds=normal_embeds)
    # maximize the loss 从四个cand中选出一个
    if increase_loss == False:
        top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
    else:
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    # top_candidates now contains beam_size trigger sequences, each with a different 0th token

    for idx in range(1,num_trigger_tokens): # for all trigger tokens, skipping the 0th (we did it above)
        # loss_per_candidate = []
        # for cand, _ in top_candidates: # for all the beams, try all the candidates at idx 只有一次循环
        cand, _ = top_candidates[0]
        loss_per_candidate = get_loss_per_candidate_max(idx, model, input_ids, attention_mask, pixel_values,  cand,
                                                cand_trigger_token_ids, num_trigger_tokens,
                                                criterion=criterion,snli=snli,options=options,normal_embeds=normal_embeds)
        if increase_loss == False:
            top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
        else:
            top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate_max(index, model, input_ids, attention_mask, pixel_values, trigger_token_ids, cand_trigger_token_ids,num_trigger_tokens,criterion, snli=False,options=None,normal_embeds=None):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    cand_trigger_token_ids = torch.LongTensor(cand_trigger_token_ids).to(options.device, non_blocking=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    '修改单词'

    input_ids[:,1:1+num_trigger_tokens] = trigger_token_ids.to(options.device, non_blocking=True)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss, _, _ = attacks.max_loss(model, outputs, criterion,
                                      options,gather_backdoor_indices=None,normal_embeds=normal_embeds)
    curr_loss = loss.cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(cand_trigger_token_ids.shape[-1]):
        input_ids_cand = deepcopy(input_ids)
        trigger_token_test = deepcopy(trigger_token_ids)
        input_ids_cand[:,index+1] = cand_trigger_token_ids[:,index,cand_id]
        trigger_token_test[:,index] = cand_trigger_token_ids[:,index,cand_id]
        with torch.no_grad():
            outputs = model(input_ids=input_ids_cand, attention_mask=attention_mask, pixel_values=pixel_values)
            loss, _, _ = attacks.max_loss(model, outputs, criterion,
                                      options,gather_backdoor_indices=None,normal_embeds=normal_embeds)
        loss_per_candidate.append((deepcopy(trigger_token_test), loss.cpu().detach().numpy()))

    return loss_per_candidate