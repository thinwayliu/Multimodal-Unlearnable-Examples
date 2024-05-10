import os
import csv
import torch
import random
import logging
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image
from backdoor.utils import apply_trigger

ImageFile.LOAD_TRUNCATED_IMAGES = True
    
class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal = False, defense = False, crop_size = 150):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep = delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor
        
        self.inmodal = inmodal
        if(inmodal):
            self.augment_captions = processor.process_text([_augment_text(caption) for caption in df[caption_key].tolist()])
        
        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]
        
        if(self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx])))
        else:  
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)
        
        return item

class ImageCaptionDataset_Poison(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, options, inmodal=False, defense=False, crop_size=150):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep=delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        text_init = 'The ' * options.token_num
        self.captions_text = [text_init + i for i in self.captions_text]
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor

        self.inmodal = inmodal
        if (inmodal):
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in df[caption_key].tolist()])

        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]

        if (self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(
                _augment_image(os.path.join(self.root, self.images[idx])))
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)
        return item, idx

class ImageCaptionDataset_Poisoned(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, noise, token,options, inmodal=False, defense=False, crop_size=150,):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep=delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.token = None
        self.noise = None

        # order = torch.randperm(noise.size(0))
        
        if token is not None:
            text_init = 'The '*options.token_num
            self.captions_text = [text_init+i for i in self.captions_text]
            self.num_trigger_tokens = token.shape[1]
            self.token = token
            print(self.token.shape)
            
            
        if noise is not None:
            self.noise = noise
            print(self.noise.shape)
            
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor



        self.inmodal = inmodal
        if (inmodal):
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in df[caption_key].tolist()])

        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]

        if (self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(
                _augment_image(os.path.join(self.root, self.images[idx])))
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            if self.token is not None:
                item["input_ids"][1:1 + self.num_trigger_tokens] = self.token[idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image) 
            if self.noise is not None:
                item["pixel_values"] += self.noise[idx]
            

        return item
    
    
class ImageCaptionDataset_RN(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor,options, inmodal=False, defense=False, crop_size=150,):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep=delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.token = None

        # if token is not None:
        #     text_init = 'The '*options.token_num
        #     self.captions_text = [text_init+i for i in self.captions_text]
        #     self.num_trigger_tokens = token.shape[1]
        #     self.token = token
        #     print(self.token.shape)
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor
        
        
        
        mean = torch.tensor([[0.48145466], [0.4578275], [0.40821073]])
        mean = mean.expand(3, 224 * 224)
        mean = mean.view(3, 224, 224)

        var = torch.tensor([[0.26862954], [0.26130258], [0.27577711]])
        var = var.expand(3, 224 * 224)
        var = var.view(3, 224, 224)

        self.mean = mean
        self.var = var 
        
        self.upper_limit = ((1 - mean) / var)
        self.lower_limit = ((0 - mean) / var)
        random_noise = (torch.randint(-8, 8, [145000 ,3, 224, 224])/255.)/var
        
        
        self.noise = random_noise


        # self.token = torch.randint(30522, size=(32365,5))             
        print((self.noise[0]*var)*255)


        self.inmodal = inmodal
        if (inmodal):
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in df[caption_key].tolist()])

        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None

        logging.debug("Loaded data")
        
    def clamp(self,X,lower_limit,upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def add_gaussian_noise(self,image, mean=0, std=25):
        """
        给图像添加高斯噪声
        :param image: PIL Image对象
        :param mean: 噪声的平均值，默认为0
        :param std: 噪声的标准差，默认为25
        :return: 添加了噪声的图像
        """
        np_image = np.array(image)
        h, w, c = np_image.shape
        noise = np.random.normal(mean, std, (h, w, c))
        noisy_image = np.clip(np_image + noise, 0, 255)
        noisy_image = Image.fromarray(np.uint8(noisy_image))
        return noisy_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx])) 
        # image = self.add_gaussian_noise(image)
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]

        if (self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(
                _augment_image(os.path.join(self.root, self.images[idx])))
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            if self.token is not None:
                item["input_ids"][1:1 + self.num_trigger_tokens] = self.token[idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image) + self.noise[idx]
            item["pixel_values"] = self.clamp(item["pixel_values"], self.lower_limit, self.upper_limit)
            
#             import matplotlib.pyplot as plt

#             pic = (item["pixel_values"] * self.var + self.mean).permute(1, 2,0).detach().cpu().numpy()
#             plt.imshow(pic)
#             print(pic*255)
#             plt.savefig('./picture1.png')


        return item
    
    

class ImageCaptionDataset_Poisoned_class(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, noise, token,options,index, inmodal=False, defense=False, crop_size=150,):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep=delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.token = None

        if token is not None:
            text_init = 'The '*options.token_num
            self.captions_text = [text_init+i for i in self.captions_text]
            self.num_trigger_tokens = token.shape[1]
            self.token = token
            print(self.token.shape)
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor
        self.noise = noise
        self.index = index

        # self.token = torch.randint(30522, size=(32365,5))
        print(self.noise.shape)



        self.inmodal = inmodal
        if (inmodal):
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in df[caption_key].tolist()])

        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]

        if (self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(
                _augment_image(os.path.join(self.root, self.images[idx])))
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            if self.token is not None:
                item["input_ids"][1:1 + self.num_trigger_tokens] = self.token[self.index[idx]]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image) + self.noise[self.index[idx]]

        return item
def calculate_scores(options, model, dataloader, epoch):

    if options.distributed:
        model = model.module  
    model.eval()

    dirname = os.path.dirname(options.train_data)
    filename = f'{options.name}_{epoch}.csv'
    path = os.path.join(dirname, filename)

    csvfile = open(path, 'a')
    csvwriter = csv.writer(csvfile)

    with torch.no_grad():
        logging.info(len(dataloader))
        for index, batch in tqdm(enumerate(dataloader)):
            image, input_ids, attention_mask = batch["pixel_values"].to(options.device), batch["input_ids"].to(options.device),  batch["attention_mask"].to(options.device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = image)
            scores  = model.logit_scale.exp() * torch.diagonal(outputs.image_embeds @ outputs.text_embeds.t())
            for j in range(len(scores)):
                csvwriter.writerow([batch['image_path'][j], batch['caption'][j], batch['is_backdoor'][j].item(), scores[j].item()])
    return path

def get_clean_train_dataloader(options, processor, path):

    logging.info(f'Creating a clean train dataloader with path {path}')

    if options.master:
        df = pd.read_csv(path, names = ['image', 'caption', 'is_backdoor', 'score'], header = None)
        df = df.sort_values(by=['score'], ascending = False)
        df_clean = df.iloc[int(options.remove_fraction * len(df)) :]
        df_dirty = df.iloc[: int(options.remove_fraction * len(df))]
        total_backdoors = sum(df['is_backdoor'].tolist())
        backdoor_detected = sum(df_dirty['is_backdoor'].tolist())
        if options.wandb:
            wandb.log({'number of backdoored images': total_backdoors,
                        'number of backdoor images removed': backdoor_detected,
                    }) 
        df_clean.to_csv(path, index = False)
        # backdoor_detected = sum(df.iloc[:5000]['is_backdoor'].tolist())
        # logging.info(f'Number of backdoors in Top-5000 examples: {backdoor_detected}')
        # for i in range(len(df)):
        #     if i < 5000:
        #         df.loc[i, 'is_backdoor'] = 1
        #     else:
        #         df.loc[i, 'is_backdoor'] = 0
        # df.to_csv(path, index = False)

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    sampler = DistributedSampler(dataset) if(options.distributed) else None
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * options.batch_size
    dataloader.num_batches = len(dataloader)
    return dataloader
    
def get_train_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
        
    sampler = DistributedSampler(dataset) if(options.distributed) else None

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * batch_size 
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_poison_train_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset_Poison(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, inmodal=options.inmodal,options=options)

    sampler = DistributedSampler(dataset) if (options.distributed) else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=options.num_workers,
                            pin_memory=True, sampler=sampler, drop_last=True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, options = None):
        self.root = root
        # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
        # print(filename)
        # df = pd.read_csv(os.path.join(root, filename))
        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.options = options
        self.add_backdoor = options.add_backdoor
        self.backdoor_sufi = options.backdoor_sufi
        if self.backdoor_sufi:
            self.backdoor_indices = list(range(50000))
            shuffle(self.backdoor_indices)
            self.backdoor_indices = self.backdoor_indices[:1000]

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):
        return apply_trigger(image, patch_size, patch_type, patch_location)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

        if self.backdoor_sufi:
            if idx in self.backdoor_indices:
                image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location)
            label = 954
            return image, label

        if self.add_backdoor:
            image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location)

        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        print(f'Test: {options.add_backdoor}')
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    else:
        path = options.eval_test_data_dir
        if (path is None): return
        dataset = ImageCaptionDataset(path, image_key=options.image_key, caption_key=options.caption_key,
                                      delimiter=options.delimiter, processor=processor, inmodal=options.inmodal)

        #raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_train_dataloader(options, processor):
    # if(not options.linear_probe or not options.finetune or options.eval_train_data_dir is None): return
    if(options.eval_train_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_train_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image), torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "val", transform = processor.process_image)])
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        options.add_backdoor = False
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None, shuffle = True)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def load(options, processor):
    data = {}
    
    data["train"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data

def poison_load(options, processor):
    data = {}

    data["train"] = get_poison_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data

def get_poison_test_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size
    options.save_path = os.path.join(options.logs, options.save_pert)
    
    try:
        noise = torch.load(os.path.join(options.save_path, "noise.pt"))
    except:
        noise = None
    try:
        token = torch.load(os.path.join(options.save_path, "token.pt"))
    except:
        token = None
        
    dataset = ImageCaptionDataset_Poisoned(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, inmodal=options.inmodal,noise=noise,token=token,options=options)

    sampler = DistributedSampler(dataset) if (options.distributed) else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=options.num_workers,
                            pin_memory=True, sampler=sampler, drop_last=True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_poison_class_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size
    options.save_path = os.path.join(options.logs, options.save_pert)

    noise = torch.load(os.path.join(options.save_path, "noise.pt"))
    try:
        token = torch.load(os.path.join(options.save_path, "token.pt"))
    except:
        token = None

    index = torch.load(os.path.join(options.save_path, "index_list.pt"))

    dataset = ImageCaptionDataset_Poisoned_class(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, inmodal=options.inmodal,noise=noise,token=token,index=index,options=options)

    sampler = DistributedSampler(dataset) if (options.distributed) else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=options.num_workers,
                            pin_memory=True, sampler=sampler, drop_last=True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_poison_RNclass_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size
    # options.save_path = os.path.join(options.logs, options.save_pert)

    mean = torch.tensor([[0.48145466], [0.4578275], [0.40821073]])
    mean = mean.expand(3, 224 * 224)
    mean = mean.view(3, 224, 224)

    var = torch.tensor([[0.26862954], [0.26130258], [0.27577711]])
    var = var.expand(3, 224 * 224)
    var = var.view(3, 224, 224)

    upper_limit = ((1 - mean) / var)
    lower_limit = ((0 - mean) / var)
    
    noise = (torch.randint(-8, 8, [500 ,3, 224, 224])/255.)/var
    #设定伪标签，为了减少shortcut
    index = torch.randint(500,size=(30000,))
    token = torch.randint(30522,size=(500,options.token_num))
    token = torch.LongTensor(token)


    dataset = ImageCaptionDataset_Poisoned_class(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, inmodal=options.inmodal,noise=noise,token=token,index=index,options=options)

    sampler = DistributedSampler(dataset) if (options.distributed) else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=options.num_workers,
                            pin_memory=True, sampler=sampler, drop_last=True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader





def poison_test_load(options, processor):
    data = {}

    data["train"] = get_poison_test_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data

def poison_class_test_load(options, processor):
    data = {}

    data["train"] = get_poison_class_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data

def poison_class_RN_load(options, processor):
    data = {}

    data["train"] = get_poison_RNclass_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data


def RN_load(options, processor):
    data = {}

    data["train"] = get_RN_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data


def get_RN_train_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset_RN(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, inmodal=options.inmodal,options=options)

    sampler = DistributedSampler(dataset) if (options.distributed) else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=options.num_workers,
                            pin_memory=True, sampler=sampler, drop_last=True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader