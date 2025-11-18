# -*- coding:utf-8 -*-

import random, math
import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
# import PIL
from PIL import Image


Indice_Column_NumCls_Dict = {'Age':{'left_image_file_path':0,'right_image_file_path':6,'colum':0,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                             'Sex':{'left_image_file_path':0,'right_image_file_path':6,'colum':1,'num_cls':2,'label_names':['male','female'],'metric':'roc','task':'cls'},
                            'HbA1c':{'left_image_file_path':0,'right_image_file_path':6,'colum':2,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'glucose':{'left_image_file_path':0,'right_image_file_path':6,'colum':3,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'bmi':{'left_image_file_path':0,'right_image_file_path':6,'colum':4,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'icd':{'left_image_file_path':0,'right_image_file_path':6,'colum':5,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'ckd_binary':{'left_image_file_path':0,'right_image_file_path':6,'colum':6,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'egfr_6stage':{'left_image_file_path':0,'right_image_file_path':6,'colum':7,'num_cls':6,'label_names':['G1','G2','G3a','G3b','G4','G5'],'metric':'macro_f1','task':'cls'},
                            'ckd_6stage':{'left_image_file_path':0,'right_image_file_path':6,'colum':8,'num_cls':7,'label_names':['Normal','G1','G2','G3a','G3b','G4','G5'],'metric':'macro_f1','task':'cls'},
                            'diabete':{'left_image_file_path':0,'right_image_file_path':6,'colum':9,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'SBP':{'left_image_file_path':0,'right_image_file_path':6,'colum':10,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},

                             }
Task_List = ['Sex','icd', 'ckd_binary','diabete', 'Age', 'HbA1c', 'glucose', 'bmi','SBP']

class UKBBRetinal_Labels(Dataset):
    def __init__(self,data_file,task_id_list,seed=None):
        # super(UKBBRetinal_Labels, self).__init__()
        # task1_labels ----  list:[0,1,2,3,4]
        # task2_labels ----  list:[5,6,7,8,9]  (data_file=train_file,task=configs.task_name, task_info = task_info_,seed=sample_random_seed,mode='train')

        self.data_file = data_file
        self.data_file_list = glob.glob(data_file+'/*pth.tar')

        self.task_id_list = task_id_list

        self.seed  =seed


    def __len__(self):

        return len(self.data_file_list)

    def __getitem__(self, item):
        data_dict = {}

        imgs = torch.load(self.data_file_list[item], weights_only=False)

        #### left eye
        left15_img = imgs['left_img']
        right16_img = imgs['right_img']

        if self.seed is not None:
            if random.random() > 0.5:
                rotate_degree = random.randint(1, 4) * 90
                left15_img = TF.rotate(left15_img, angle=rotate_degree)
                # left15_img = left15_img.rotate(rotate_degree, Image.BILINEAR, expand=0)
            if random.random() > 0.5:
                left15_img = TF.hflip(left15_img)
            if random.random() > 0.5:
                left15_img = TF.vflip(left15_img)


        #### right eye

            if random.random() > 0.5:
                rotate_degree = random.randint(1, 4) * 90
                right16_img = TF.rotate(right16_img, angle=rotate_degree)
            if random.random() > 0.5:
                right16_img = TF.hflip(right16_img)
            if random.random() > 0.5:
                right16_img = TF.vflip(right16_img)

        data_dict['left_img'] = left15_img
        data_dict['right_img'] = right16_img

        for task_id in self.task_id_list:
            data_dict[Task_List[task_id]] = imgs[Task_List[task_id]]

        return data_dict

