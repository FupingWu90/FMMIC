# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial
import os
import torch
import torch.nn as nn
import warnings
from collections import namedtuple
InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits", "aux_logits"}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

from retizero.Finetuning import Model_Finetuing
import models_vit
from util.pos_embed import interpolate_pos_embed
import timm
os.environ['HF_HOME'] = "/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/TIMM"
import numpy as np


Backbon_Dict = {'vgg16':'vgg16.tv_in1k',
                'resnet18':'resnet18.a1_in1k',
                'densenet121':'densenet121.ra_in1k',
                'effi_b4':'efficientnet_b4.ra2_in1k',

                'Incept_v3':'inception_v3.tv_in1k', # classification

                'InceptResnet_v2':'inception_resnet_v2.tf_in1k', # classification
                #'Mobile_v3':'mobilenetv3_small_100.lamb_in1k',# classification


                'vit_b16':'vit_base_patch16_224.augreg2_in21k_ft_in1k',
                'clip_b16':'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
                'eva_clip':  'eva02_base_patch14_224.mim_in22k', #'eva02_base_patch16_clip_224.merged2b_s8b_b131k',
                'clip_opai':'vit_base_patch16_clip_224.openai_ft_in12k_in1k',

                'clip_opaif':'vit_base_patch16_clip_224.openai',
                'dino_base':'vit_base_patch16_224.dino',
                'dino_small':'vit_small_patch16_224.dino',
                'dino2_small':'vit_small_patch14_dinov2.lvd142m',
                'dino2_base':'vit_base_patch14_dinov2.lvd142m',
                'sam_base':'samvit_base_patch16.sa1b',
                'sam_cls':'vit_base_patch16_224.sam_in1k'}


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


class Multiple_Classifier(nn.Module):
    def __init__(self, embed_dim, task_id_list):
        super(Multiple_Classifier, self).__init__()

        self.task_id_list = task_id_list

        self.decoders = nn.ModuleList()

        for task_id in task_id_list:
            num_cls = Indice_Column_NumCls_Dict[Task_List[task_id]]['num_cls']
            if num_cls ==2:
                num_cls = 1
            self.decoders.append(nn.Linear(embed_dim, num_cls))

    def forward(self, z):
        out = []
        for i in range(len(self.task_id_list)):
            out_i = self.decoders[i](z)
            out.append(out_i)

        return out




class MT_Net(nn.Module):
    def __init__(self, configs):
        super(MT_Net, self).__init__()
        self.backbone = None
        self.backbone_name = configs.backbone

        self.task_id_list = configs.task_id_list

        # self.classifiers = None

        if configs.backbone == 'retfound':

            self.backbone = models_vit.__dict__['RETFound_mae'](
                num_classes=0,
                drop_path_rate=0.2,
                # global_pool=True,
            )

            # load RETFound weights
            if configs.backbone_update != 'scratch':
                checkpoint = torch.load(
                    '/gpfs3/well/papiez/users/cub991/PJ2022/UKBB_Project/RETFound/RETFound_MAE/RETFound_cfp_weights.pth',
                    map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = self.backbone.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                interpolate_pos_embed(self.backbone, checkpoint_model)

                # load pre-trained model
                msg = self.backbone.load_state_dict(checkpoint_model, strict=False)

                # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

            # manually initialize fc layer
            # trunc_normal_(self.backbone.head.weight, std=2e-5)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

            # self.classifiers = Multiple_Classifier(self.backbone.embed_dim,configs.task_id_list)
            self.backbone.head = Multiple_Classifier(self.backbone.embed_dim,configs.task_id_list)


        elif configs.backbone == 'dinov3':
            REPO_DIR = '/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/DINOv3/dinov3'
            weight_file = '/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/DINOv3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
            if configs.backbone_update == 'scratch':
                self.backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', pretrained=False)
            else:
                self.backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=weight_file)

            self.backbone.head = Multiple_Classifier(self.backbone.embed_dim,configs.task_id_list) #torch.nn.Linear(self.backbone.embed_dim, out_ch)

        elif configs.backbone == 'retizero':
            weight_path = "/gpfs3/well/papiez/users/cub991/PJ2022/UKBB_Project/RetiZero/pretrained_model/RetiZero.pth"
            self.backbone = Model_Finetuing(model_name="lora", class_num=1, weight_path=weight_path)

            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

            feature_dim = 1024

            # self.classifiers = Multiple_Classifier(feature_dim,configs.task_id_list)
            self.backbone.classifier = Multiple_Classifier(feature_dim,configs.task_id_list)

        else:
            if configs.backbone_update == 'scratch':
                self.backbone = timm.create_model(Backbon_Dict[configs.backbone], pretrained=False, num_classes=0)
            else:

                self.backbone = timm.create_model(Backbon_Dict[configs.backbone], pretrained=True, num_classes=0)
            mean, std = self.backbone.default_cfg['mean'], self.backbone.default_cfg['std']

            if configs.backbone in ['densenet121']:
                # self.classifiers = Multiple_Classifier(self.backbone.num_features,configs.task_id_list)
                self.backbone.classifier = Multiple_Classifier(self.backbone.num_features,configs.task_id_list)

            elif configs.backbone in ['dino2_base']:
                # self.classifiers = Multiple_Classifier(self.backbone.embed_dim,configs.task_id_list)
                self.backbone.head = Multiple_Classifier(self.backbone.embed_dim,configs.task_id_list)

    def forward(self, img):
        out = self.backbone(img)

        return out



    def zero_grad_shared_modules(self):
        base_params = self.get_base_params()
        for p in base_params:
            # p.grad.detach_()
            p.grad.zero_()


    def get_base_params(self,):
        # logits_params_id = list(map(id, self.classifiers.parameters()))
        # for i in range(len(self.task_id_list)):
        #     logits_params_id += list(map(id, self.classifiers[i].parameters()))
        base_params = []
        for name,param in self.named_parameters():
            if 'classifier' not in name and 'head' not in name:
                if param.requires_grad:
                    base_params.append(param)
        # base_params = filter(lambda p: id(p) not in logits_params_id, params)

        return base_params







