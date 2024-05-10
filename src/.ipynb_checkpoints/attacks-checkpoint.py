"""
Contains different methods for attacking models. In particular, given the gradients for token
embeddings, it computes the optimal token replacements. This code runs on CPU.
"""
import torch
import numpy
from torch.autograd import Variable
from pkgs.openai.tokenizer import SimpleTokenizer as Tokenizer
import torch
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import copy
import torch.nn.functional as F

def get_embeddings(model,input_ids):
    return model.token_embedding(input_ids).type(model.dtype)


def pred_text_idx(model,input_ids,attention_mask,options):
    classifier = KMeans(32)
    input_keyword = copy.deepcopy(input_ids).to(options.device)
    input_keyword[:,options.token_num + 1:] = 0
    input_keyword[:,options.token_num + 1] = 49407
    text_embeds = model.get_text_features(input_ids=input_keyword, attention_mask=attention_mask).cpu()
    pred_text_idx = classifier.fit_predict(text_embeds.view(input_ids.shape[0], -1).numpy())
    return pred_text_idx 

def linear_loss(noise,pred_text_idx):
    loss = get_linear_noise_csd_loss(noise.view(noise.shape[0], -1), torch.tensor(pred_text_idx).data)
    # if is_test == True:
    #     pred_text_idx = classifier.fit_predict(text_embeds.view(32, -1).detach().cpu().numpy())
    #     print('text_idx',pred_text_idx)
    return loss

def get_linear_noise_csd_loss(x, labels):
    sample = x.reshape(x.shape[0], -1)
    cluster_label = labels

    class_center = []
    intra_class_dis = []
    c = torch.max(cluster_label) + 1

    for i in range(c):

        idx_i = torch.where(cluster_label == i)[0]
        if idx_i.shape[0] == 0:
            continue

        class_i = sample[idx_i, :]
        class_i_center = class_i.mean(dim=0)
        class_center.append(class_i_center)

        point_dis_to_center = torch.sqrt(torch.sum((class_i - class_i_center) ** 2, dim=1))
        intra_class_dis.append(torch.mean(point_dis_to_center))

    # print("time3: {}".format(time3 - time2))
    if len(class_center) <= 1:
        return 0
    class_center = torch.stack(class_center, dim=0)

    c = len(intra_class_dis)

    class_dis = torch.cdist(class_center, class_center,
                            p=2)  # TODO: this can be done for only one time in the whole set

    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

    cluster_DB_loss = ((intra_class_dis_pair_sum + 1e-5) / (class_dis + 1e-5)).mean()

    loss = cluster_DB_loss

    # print('get_linear_noise_csd_loss:', cluster_DB_loss.item())

    return loss

def encode_text(predict_list):
    encode_list = []
    transform = Tokenizer()
    for text in predict_list:
        sentence = []
        for word in text:
            e_text = transform.encode(word)
            if len(e_text)>1:
                sentence.append(transform.encode('the')[0])
            else:
                sentence.append(e_text[0])
        encode_list.append(sentence)
    return encode_list

def decode_text(input_ids,token_num):
    decode_list = []
    transform = Tokenizer()
    for text in input_ids:
        last_nonzero_index = torch.max(torch.nonzero(text)).item()
        d_text = transform.decode(text[1+token_num:last_nonzero_index].cpu().tolist())
        decode_list.append(d_text)
    return decode_list

def decode(input_ids):
    decode_list = []
    transform = Tokenizer()
    for text in input_ids:
        d_text = transform.decode(text.cpu().tolist())
        decode_list.append(d_text)
    return decode_list

def decode_vis(input_ids):
    transform = Tokenizer()
    d_text = transform.decode(input_ids.cpu().tolist())
    return d_text

def process_text(decode_text,options,tokenizer):
    mask = '[MASK] ' * options.token_num
    # print([tokenizer.tokenize('[CLS] '+ mask + text + '[SEP]') for text in decode_text])
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS] '
                                                                    + mask + text + '[SEP]'))
                      for text in decode_text]


    max_length = max(len(x) for x in indexed_tokens)

    padded_list = [x + [0] * (max_length - len(x)) for x in indexed_tokens]
    indexed_tokens = torch.tensor(padded_list)

    tokens_tensor = torch.tensor(indexed_tokens)
    tokens_tensor = tokens_tensor.to(options.device)

    return tokens_tensor

def predict_text(token_tensor,token_index,tokenizer,model,options,num_cand):

    predicted_index = []

    with torch.no_grad():
        outputs = model(token_tensor)
        predictions = outputs[0]

    for batch_index in range(token_tensor.shape[0]):
        _, index = torch.topk(predictions[batch_index,token_index],num_cand) # [32,50]
        predicted_index.append(index)

    predict_list = []

    for row in predicted_index:
        predicted_token = tokenizer.convert_ids_to_tokens(row)
        # print('The word is predicted as :', predicted_token)
        predict_list.append(predicted_token)

    return predict_list


def change_word(process_text,cand_word,tokenizer,token_index):
    for i,word in enumerate(cand_word):
        process_text[i][token_index] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0]
    return process_text

def hotflip_attack_new(averaged_grad, embedding_matrix,
                   increase_loss=False, num_candidates = 100,num_trigger_tokens=1,no_zero_index = None):
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()

    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",(averaged_grad,embedding_matrix))

    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.

    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=-1)
        return best_k_ids.detach()
    _, best_at_each_step = gradient_dot_embedding_matrix.max(1)
    return best_at_each_step.detach()


def hotflip_attack(averaged_grad, embedding_matrix,
                   increase_loss=False, num_candidates=1,num_trigger_tokens=1,no_zero_index = None):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """

    averaged_grad = averaged_grad.cpu()[:,1:1+num_trigger_tokens,:]
    embedding_matrix = embedding_matrix.cpu()
    embedding_matrix= torch.transpose(embedding_matrix, dim0=0, dim1=1)
    gradient_dot_embedding_matrix = torch.matmul(averaged_grad,embedding_matrix)


    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.

    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

def random_attack(embedding_matrix, trigger_token_ids, num_candidates=1):
    """
    Randomly search over the vocabulary. Gets num_candidates random samples and returns all of them.
    """
    embedding_matrix = embedding_matrix.cpu()
    new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
    for trigger_token_id in range(len(trigger_token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = numpy.random.randint(embedding_matrix.shape[0])
            new_trigger_token_ids[trigger_token_id][candidate_number] = rand_token
    return new_trigger_token_ids

# steps in the direction of grad and gets the nearest neighbor vector.
def nearest_neighbor_grad(averaged_grad, embedding_matrix, trigger_token_ids,
                          tree, step_size, increase_loss=False, num_candidates=1):
    """
    Takes a small step in the direction of the averaged_grad and finds the nearest
    vector in the embedding matrix using a kd-tree.
    """
    new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    if increase_loss: # reverse the sign
        step_size *= -1
    for token_pos, trigger_token_id in enumerate(trigger_token_ids):
        # take a step in the direction of the gradient
        trigger_token_embed = torch.nn.functional.embedding(torch.LongTensor([trigger_token_id]),
                                                            embedding_matrix).detach().cpu().numpy()[0]
        stepped_trigger_token_embed = trigger_token_embed + \
            averaged_grad[token_pos].detach().cpu().numpy() * step_size
        # look in the k-d tree for the nearest embedding
        _, neighbors = tree.query([stepped_trigger_token_embed], k=num_candidates)
        for candidate_number, neighbor in enumerate(neighbors[0]):
            new_trigger_token_ids[token_pos][candidate_number] = neighbor
    return new_trigger_token_ids

def sim_loss(batch1,batch2):
    batch1_normalized = F.normalize(batch1, p=2, dim=1)  # 归一化第一个批次
    batch2_normalized = F.normalize(batch2, p=2, dim=1)  # 归一化第二个批次

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(batch1_normalized, batch2_normalized, dim=1)

    # 计算余弦相似度损失
    cosine_loss = 1 - cosine_sim.mean() 
    return cosine_loss

def normal_embeds(input_ids,attention_mask,model,options):
    input_ids_copy = input_ids.clone()
    input_ids_copy[:,1:input_ids.shape[1]-options.token_num] = input_ids[:,1+options.token_num:]
    # input_ids_copy[:,self.num_trigger_tokens + 1] = 49407
    return model.get_text_features(input_ids_copy,attention_mask)


def max_loss(umodel, outputs, criterion, options, gather_backdoor_indices,normal_embeds):
    if (options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[
                                               :len(outputs.image_embeds) // 2], outputs.image_embeds[
                                                                                 len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[
                                                                                                  len(outputs.text_embeds) // 2:]
    else:
        image_embeds = normal_embeds
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