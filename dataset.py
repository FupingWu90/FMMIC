# -*- coding:utf-8 -*-

import random, math
import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import datasets, transforms
import PIL
from PIL import Image
import medmnist
from medmnist import INFO, Evaluator,OrganAMNIST,OrganCMNIST,OrganSMNIST,PathMNIST,DermaMNIST,BreastMNIST,OCTMNIST,PneumoniaMNIST,ChestMNIST,RetinaMNIST,BloodMNIST,TissueMNIST
import math
from Backbones.SAM.utils.transforms import ResizeLongestSide

info = {"pathmnist": {
        "python_class": "PathMNIST",
        "description": "The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "url_64": "https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1",
        "MD5_64": "55aa9c1e0525abe5a6b9d8343a507616",
        "url_128": "https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1",
        "MD5_128": "ac42d08fb904d92c244187169d1fd1d9",
        "url_224": "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1",
        "MD5_224": "2c51a510bcdc9cf8ddb2af93af1eadec",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium",
        },
        "n_channels": 3,
        "n_samples": {"train": 89996, "val": 10004, "test": 7180},
        "license": "CC BY 4.0",
    },
    'url':'https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py'
}

MedMNIST_INFO = {'OrganAMNIST':11,'OrganCMNIST':11,'OrganSMNIST':11,'PathMNIST':9,'DermaMNIST':7,'BreastMNIST':2,'OCTMNIST':4,'PneumoniaMNIST':2,'ChestMNIST':14,'RetinaMNIST':5,'BloodMNIST':8,'TissueMNIST':8}

class MedMNIST_Embedding_Labels(Dataset):
    def __init__(self,configs,train,sample_num_per_cls=10000,seed=None):
        root_path = '/gpfs3/well/papiez/users/cub991/Datasets/OrganCMNIST'
        # if train:
        #
        #     self.pre_process = transforms.Compose([
        #                                 transforms.RandomCrop(32, padding=4),
        #                                 transforms.RandomHorizontalFlip(),
        #                                 transforms.ToTensor(),
        #                             ])
        # else:

        self.pre_process = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[.5], std=[.5])
        ])
        OrganCMNIST_dataset = None
        if configs.dataname == 'OrganCMNIST':
            OrganCMNIST_dataset = OrganCMNIST(split=train, transform=self.pre_process, download=True,
                                                   root=root_path, size=configs.img_size)
        elif configs.dataname == 'OrganSMNIST':
            OrganCMNIST_dataset = OrganSMNIST(split=train, transform=self.pre_process, download=True,
                                                   root=root_path, size=configs.img_size)
        elif configs.dataname == 'PathMNIST':
            OrganCMNIST_dataset = PathMNIST(split=train, transform=self.pre_process, download=True,
                                                 root=root_path, size=configs.img_size)
        elif configs.dataname == 'DermaMNIST':
            OrganCMNIST_dataset = DermaMNIST(split=train, transform=self.pre_process, download=True,
                                                  root=root_path, size=configs.img_size)
        elif configs.dataname == 'BreastMNIST':
            OrganCMNIST_dataset = BreastMNIST(split=train, transform=self.pre_process, download=True,
                                                   root=root_path, size=configs.img_size)
        elif configs.dataname == 'OrganAMNIST':
            OrganCMNIST_dataset = OrganAMNIST(split=train, transform=self.pre_process, download=True,
                                                   root=root_path, size=configs.img_size)
        elif configs.dataname == 'OCTMNIST':
            OrganCMNIST_dataset = OCTMNIST(split=train, transform=self.pre_process, download=True,
                                                root=root_path, size=configs.img_size)
        elif configs.dataname == 'PneumoniaMNIST':
            OrganCMNIST_dataset = PneumoniaMNIST(split=train, transform=self.pre_process, download=True,
                                                      root=root_path, size=configs.img_size)
        elif configs.dataname == 'ChestMNIST':
            OrganCMNIST_dataset = ChestMNIST(split=train, transform=self.pre_process, download=True,
                                                  root=root_path, size=configs.img_size)
        elif configs.dataname == 'RetinaMNIST':
            OrganCMNIST_dataset = RetinaMNIST(split=train, transform=self.pre_process, download=True,
                                                   root=root_path, size=configs.img_size)
        elif configs.dataname == 'BloodMNIST':
            OrganCMNIST_dataset = BloodMNIST(split=train, transform=self.pre_process, download=True,
                                                  root=root_path, size=configs.img_size)
        elif configs.dataname == 'TissueMNIST':
            OrganCMNIST_dataset = TissueMNIST(split=train, transform=self.pre_process, download=True,
                                                   root=root_path, size=configs.img_size)

        self.original_imgs = OrganCMNIST_dataset.imgs
        dims = len(self.original_imgs.shape)
        if dims == 3:
            self.original_imgs = np.expand_dims(self.original_imgs, axis=-1)
        self.original_targets = OrganCMNIST_dataset.labels


        # task2_labels ----  list:[5,6,7,8,9]
        save_path = '/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundEmbeddings/{}_{}/{}_{}/'.format(configs.dataname, configs.img_size, configs.backbone,configs.img_resize)
        embedding_file_name = save_path + '{}_img_embedding.npy'.format(train)
        gt_file_name = save_path + '{}_gt.npy'.format(train)

        self.imgs = np.load(embedding_file_name)
        self.targets = np.load(gt_file_name)
        self.num_cls = MedMNIST_INFO[configs.dataname]
        self.backbone = configs.backbone
        # transform input shape to (B,N,C)
        if 'SAM' in configs.backbone:
            N,C,W,H = self.imgs.shape
            self.embedding_size = C
            self.map_size = W
            self.patch_size = int(configs.img_resize/self.map_size)
            self.imgs = self.imgs.reshape(N,C,-1).transpose(0,2,1) # (N,W*H,C)
        elif 'DINO' in configs.backbone or 'dinov2' in configs.backbone:
            B, N, C = self.imgs.shape
            self.embedding_size = C # (B,1+N,C)
            self.map_size = int(math.sqrt(N-1))
            self.patch_size = int(configs.img_resize / self.map_size)

        if sample_num_per_cls < 1001:
            np.random.seed(seed)
            select_ids = []
            for label_id in range(self.num_cls):
                select_id = np.where(np.array(self.targets.squeeze(1)) == label_id)[0].tolist()
                select_ids = select_ids + select_id

                # indices = np.random.choice(self.imgs.shape[0], size=sample_num, replace=False)
            self.imgs = self.imgs[select_ids]
            self.targets = self.targets[select_ids]




    def __len__(self):

        return self.imgs.shape[0]

    def __getitem__(self, index):

        data, label = self.imgs[index], self.targets[index].astype(int)
        data = torch.from_numpy(data)

        return data,torch.LongTensor(label)



class MedMNIST_Labels(Dataset):
    def __init__(self,dataset_name,train,img_size=224,img_resize=224,download=False,resize=False,sample_num_per_cls=10000,seed=None,mean = None, std=None):
        # task1_labels ----  list:[0,1,2,3,4]
        # task2_labels ----  list:[5,6,7,8,9]
        root_path = '/gpfs3/well/papiez/users/cub991/Datasets/OrganCMNIST'
        # if train:
        #
        #     self.pre_process = transforms.Compose([
        #                                 transforms.RandomCrop(32, padding=4),
        #                                 transforms.RandomHorizontalFlip(),
        #                                 transforms.ToTensor(),
        #                             ])
        # else:
        if resize:
            self.pre_process = transforms.Compose(
                [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            self.pre_process =transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                        ])

        OrganCMNIST_dataset = None
        if dataset_name == 'OrganCMNIST':
            OrganCMNIST_dataset = OrganCMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'OrganSMNIST':
            OrganCMNIST_dataset = OrganSMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'PathMNIST':
            OrganCMNIST_dataset = PathMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'DermaMNIST':
            OrganCMNIST_dataset = DermaMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'BreastMNIST':
            OrganCMNIST_dataset = BreastMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'OrganAMNIST':
            OrganCMNIST_dataset = OrganAMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'OCTMNIST':
            OrganCMNIST_dataset = OCTMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'PneumoniaMNIST':
            OrganCMNIST_dataset = PneumoniaMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'ChestMNIST':
            OrganCMNIST_dataset = ChestMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'RetinaMNIST':
            OrganCMNIST_dataset = RetinaMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'BloodMNIST':
            OrganCMNIST_dataset = BloodMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)
        elif dataset_name == 'TissueMNIST':
            OrganCMNIST_dataset = TissueMNIST(split=train, transform=self.pre_process, download=download,
                                                   root=root_path, size=img_size)

        # extract task1 dataset
        # task_id_list = []
        # for id in task_labels:
        #     index = np.where(np.array(self.CIFAR10_dataset.targets) == id)[0].tolist()
        #     task_id_list = task_id_list + index
        self.dataset_name = dataset_name
        self.imgs = OrganCMNIST_dataset.imgs
        self.targets = OrganCMNIST_dataset.labels
        self.num_cls = MedMNIST_INFO[dataset_name]

        del OrganCMNIST_dataset

        if sample_num_per_cls < 1001:
            np.random.seed(seed)
            if dataset_name == 'ChestMNIST':
                indices = np.random.choice(self.imgs.shape[0], size=sample_num_per_cls*self.num_cls, replace=False)
                self.imgs = self.imgs[indices]
                self.targets = self.targets[indices]
            else:
                select_ids = []
                for label_id in range(self.num_cls):
                    select_id_all = np.where(np.array(self.targets.squeeze(1)) == label_id)[0].tolist()
                    select_id = select_id_all[:sample_num_per_cls]
                    select_ids = select_ids + select_id

                    # indices = np.random.choice(self.imgs.shape[0], size=sample_num, replace=False)
                self.imgs = self.imgs[select_ids]
                self.targets = self.targets[select_ids]



        # dims = len(self.imgs.shape)
        # if dims == 3:
        #     self.img_channel = 1
        #     self.imgs = np.expand_dims(self.imgs, axis= -1)
        # else:
        #     self.img_channel = self.imgs.shape[3]

        self.sam_pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.sam_pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1,1,1)
        self.img_size = img_size
        self.img_resize = img_resize





    def __len__(self):

        return self.imgs.shape[0]

    def __getitem__(self, index):

        data, label = self.imgs[index], self.targets[index].astype(int) # data shape: w*h*3
        # if self.img_channel != 3:
        #     data = np.repeat(data, 3, axis=2)
        # data = Image.fromarray(data)
        # data = self.pre_process(data)


        if self.img_resize == self.img_size:
            data = Image.fromarray(data)
            data = data.convert("RGB")
            resize_img_tensor = self.pre_process(data)
        else:
            sam_transform = ResizeLongestSide(self.img_resize)
            resize_img = sam_transform.apply_image(data)
            resize_img = Image.fromarray(resize_img)
            resize_img = resize_img.convert("RGB")
            resize_img_tensor = self.pre_process(resize_img)

        return resize_img_tensor,label