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


class MedMNIST_Labels(Dataset):
    def __init__(self,dataset_name,train,img_size=224,img_resize=224,download=False,resize=False,sample_num_per_cls=10000,semi = None,seed=None,mean = None, std=None):

        root_path = '/gpfs3/well/papiez/users/cub991/Datasets/OrganCMNIST'

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

        self.dataset_name = dataset_name
        self.imgs = OrganCMNIST_dataset.imgs
        self.targets = OrganCMNIST_dataset.labels
        self.num_cls = MedMNIST_INFO[dataset_name]

        del OrganCMNIST_dataset

        self.semi = semi

        if sample_num_per_cls < 1001:
            np.random.seed(seed)
            if dataset_name == 'ChestMNIST':
                indices = np.random.choice(self.imgs.shape[0], size=sample_num_per_cls*self.num_cls, replace=False)
                self.imgs_l = self.imgs[indices]
                self.targets_l = self.targets[indices]
                all_indices = np.arange(self.imgs.shape[0])  # [0, 1, 2, ..., 99]
                unselected_ids = np.setdiff1d(all_indices, indices)
                self.imgs_u = self.imgs[unselected_ids]
                self.targets_u = self.targets[unselected_ids]
            else:
                select_ids = []
                unselected_ids = []
                for label_id in range(self.num_cls):
                    select_id_all = np.where(np.array(self.targets.squeeze(1)) == label_id)[0].tolist()
                    select_id = select_id_all[:sample_num_per_cls]
                    select_ids = select_ids + select_id
                    unselected_ids = unselected_ids + select_id_all[sample_num_per_cls:]

                    # indices = np.random.choice(self.imgs.shape[0], size=sample_num, replace=False)
                self.imgs_l = self.imgs[select_ids]
                self.targets_l = self.targets[select_ids]

                self.imgs_u = self.imgs[unselected_ids]
                self.targets_u = self.targets[unselected_ids]


        self.sam_pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.sam_pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1,1,1)
        self.img_size = img_size
        self.img_resize = img_resize





    def __len__(self):
        if self.semi is None:
            return self.imgs.shape[0]
        else:
            if 'semi' in self.semi:
                return self.imgs_u.shape[0]
            else:

                return self.imgs_l.shape[0]

    def __getitem__(self, index):
        if self.semi is None:
            data, label = self.imgs[index], self.targets[index].astype(int)  # data shape: w*h*3

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

            return resize_img_tensor, label

        else:
            if 'semi' in self.semi:

                ## labeled
                index_l = index % self.imgs_l.shape[0]

                data, label = self.imgs_l[index_l], self.targets_l[index_l].astype(int) # data shape: w*h*3


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


                ## unlabeled
                data_u, label_u = self.imgs_u[index], self.targets_u[index].astype(int)  # data shape: w*h*3

                if self.img_resize == self.img_size:
                    data_u = Image.fromarray(data_u)
                    data_u = data_u.convert("RGB")
                    resize_img_tensor_u = self.pre_process(data_u)
                else:
                    sam_transform = ResizeLongestSide(self.img_resize)
                    resize_img_u = sam_transform.apply_image(data_u)
                    resize_img_u = Image.fromarray(resize_img_u)
                    resize_img_u = resize_img_u.convert("RGB")
                    resize_img_tensor_u = self.pre_process(resize_img_u)

                return resize_img_tensor,label,resize_img_tensor_u

            else:
                data, label = self.imgs_l[index], self.targets_l[index].astype(int)  # data shape: w*h*3

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

                return resize_img_tensor, label