import torch
from torch.backends import cudnn
import argparse
import pandas as pd
from utils import *
from dataset import *
from Network import *
from CAGrad_Trainer import Trainer
import random
import sys

os.environ['HF_HOME'] = "/FoundCheckpoints/TIMM"
import datetime


Task_List = ['Sex','icd', 'ckd_binary','diabete', 'Age', 'HbA1c', 'glucose', 'bmi','SBP'] # 'ckd_6stage', 'egfr_6stage',

Indice_Column_NumCls_Dict_SeparateFiles = {'Age':{'colum':4,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                             'Sex':{'colum':5,'num_cls':2,'label_names':['male','female'],'metric':'roc','task':'cls'},
                            'HbA1c':{'colum':6,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'glucose':{'colum':7,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'bmi':{'colum':8,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'icd':{'colum':4,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'ckd_binary':{'colum':7,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'ckd_6stage':{'colum':10,'num_cls':7,'label_names':['Normal','G1','G2','G3a','G3b','G4','G5'],'metric':'macro_f1','task':'cls'},
                            'egfr_6stage':{'colum':8,'num_cls':6,'label_names':['G1','G2','G3a','G3b','G4','G5'],'metric':'macro_f1','task':'cls'},
                            'diabete':{'colum':10,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'}

                             }

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

LR_List1 = [1e-3,1e-4,1e-5,1e-6]
LR_List2 = [1e-4,1e-5,1e-6,1e-7]

BatchSize_List = [64,128]


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def index_map(x):
    y = x.split('/')[-1].split('_')
    return y[0] + '_' + y[2] + '_' + y[3]

def reset_index_name(dataframe):
    names_list = dataframe.iloc[:,0].to_list()
    index_new = [index_map(name_new) for name_new in names_list]
    dataframe.index = index_new

    return dataframe


def training_validation(configs):
    #######################################  log prepare #########################################
    Save_Dir = './Save_Train{}/'.format(configs.Train_Data)+'Task{}/{}_{}/{}_Iter{}/{}/{}_lr{}_{}_bs{}/'.format(configs.task_id_list,configs.backbone_update, configs.bb_lr,configs.backbone, configs.Iter_Num,configs.method,configs.Optim, configs.lr,
                                                                                     configs.lr_decay,
                                                                                     configs.BatchSize)
    if not os.path.exists(Save_Dir):
        # os.mkdir(SAVE_DIR)
        os.makedirs(Save_Dir)


    # data list 21015 21016
    for sample_random_seed in configs.random_seeds:
        save_result_file = Save_Dir + 'seed{}/test_ndarray.npy'.format(sample_random_seed)
        if os.path.exists(save_result_file):
            continue

        control_seed(sample_random_seed)

        train_starttime = datetime.datetime.now()

        SAVE_DIR_Seed = Save_Dir + 'seed{}/'.format(sample_random_seed)
        if not os.path.exists(SAVE_DIR_Seed):
            # os.mkdir(SAVE_DIR)
            os.makedirs(SAVE_DIR_Seed)

        model = MT_Net(configs)

        ## dataloader
        train_file = '/UKBBRetinal/tensor_data_v1/224/train'
        vali_file = '/UKBBRetinal/tensor_data_v1/224/vali'
        test_file = '/UKBBRetinal/tensor_data_v1/224/test'
        if 'dino2' in configs.backbone:
            configs.img_resize = 518
            train_file = '/UKBBRetinal/tensor_data_v1/518/train'
            vali_file = '/UKBBRetinal/tensor_data_v1/518/vali'
            test_file = '/UKBBRetinal/tensor_data_v1/518/test'


        train_dataset = UKBBRetinal_Labels(data_file=train_file, task_id_list=configs.task_id_list, seed=sample_random_seed)
        val_dataset = UKBBRetinal_Labels(data_file=vali_file, task_id_list=configs.task_id_list)
        test_dataset = UKBBRetinal_Labels(data_file=test_file, task_id_list=configs.task_id_list)

        if len(train_dataset) < configs.BatchSize:
            configs.BatchSize = len(train_dataset)

        train_loaders = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.BatchSize,
                                                    shuffle=True, sampler=None,
                                                    num_workers=0, drop_last=True)
        val_loaders = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, num_workers=0)
        test_loaders = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, num_workers=0)

        configs.iter_per_epoch = len(train_loaders)
        configs.total_epoch = configs.Iter_Num // configs.iter_per_epoch + 1



    #######################################  start train #########################################

        checkpoint_file = SAVE_DIR_Seed + 'checkpoint.pth.tar'
        checkpoint = None
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)


        # training
        trainer = Trainer(model, checkpoint, configs, train_loaders,val_loaders, test_loaders,SAVE_DIR_Seed)
        trainer.train()

        train_endtime = datetime.datetime.now()
        print('train time:', (train_endtime - train_starttime).seconds)



def main(configs):


    training_validation(configs)


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    cudnn.benchmark = True

    ## set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='densenet121')
    parser.add_argument('--Train_Data', type=str, default='L') # 'L', 'R', 'LR_Mix', 'LR_Combine'
    # network param

    parser.add_argument('--task_id_list', nargs='+', type=int, default=[0, 1,2,3,4,5,6,7,8])


    # dataset param
    parser.add_argument('--Data_Path', type=str,
                        default='/UKBB/AutoMorph_segmentation/')
    parser.add_argument('--Excel_Path_meta', type=str,
                        default='/UKBB_Project/Index_Preparing')
    parser.add_argument('--Excel_Path_ckd', type=str,
                        default='/UKBB_Project/Index_Preparing/CKD')
    parser.add_argument('--Excel_Path_diabete', type=str,
                        default='/UKBB_Project/Index_Preparing/Diabete')


    # parser.add_argument('--Img_Size', nargs='+', type=int, default=[299,299])
    parser.add_argument('--img_resize', type=int, default=224)  # DINO: 224 ,  SAM:1024

    parser.add_argument('--train_num_per_cls', type=int, default=10000)  #
    parser.add_argument('--random_seeds', nargs='+', type=int, default=[0])


    parser.add_argument('--BatchSize', type=int, default=32)
    parser.add_argument('--iter_per_epoch', type=int, default=100)

    # train param
    # parser.add_argument('--Finetune', action='store_true', help='Boolean argument')

    parser.add_argument('--Optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_iters', type=int, default=200)

    parser.add_argument('--backbone_update', type=str, default='freeze')  # 'freeze', 'finetune' lossb
    parser.add_argument('--bb_lr', type=float, default=1e-6)

    parser.add_argument('--Iter_Num', type=int, default=1) #150
    parser.add_argument('--total_epoch', type=int, default=1)  # 150



    ## loss param
    parser.add_argument('--weight', type=str, default='equal')  #
    parser.add_argument('--method', type=str, default='cagrad')  #"graddrop", "pcgrad", "mgd", "cagrad"
    parser.add_argument('--alpha', default=0.2, type=float, help='the alpha')


    # parser.add_argument('--edge_loss_type', type=str, default='L1Loss')  # 'L1Loss', balanced, 'BCE'
    parser.add_argument('--loss_weight', type=float, default=0.95) # 0.95, None

    parser.add_argument('--job_array_id', type=int, default=-10)





    CONFIGs = parser.parse_args()

    if CONFIGs.backbone_update == 'ft':

        if CONFIGs.backbone in ['vgg16', 'resnet18', 'densenet121', 'effi_b4', 'Incept_v3', 'InceptResnet_v2']:
            CONFIGs.bb_lr = LR_List1[CONFIGs.job_array_id]
        else:
            CONFIGs.bb_lr = LR_List2[CONFIGs.job_array_id]


    if CONFIGs.backbone == 'dino2_base' and CONFIGs.backbone_update == 'ft':
        CONFIGs.BatchSize = 32
    elif CONFIGs.backbone == 'dino2_base' and CONFIGs.backbone_update == 'scratch':
        CONFIGs.BatchSize = 32
    elif CONFIGs.backbone == 'dino2_base' and CONFIGs.backbone_update == 'lora':
        CONFIGs.BatchSize = 32
    elif CONFIGs.backbone == 'retfound' and CONFIGs.backbone_update == 'ft':
        CONFIGs.BatchSize = 64
    elif CONFIGs.backbone == 'retfound' and CONFIGs.backbone_update == 'scratch':
        CONFIGs.BatchSize = 64
    elif CONFIGs.backbone == 'retfound' and CONFIGs.backbone_update == 'lora' and CONFIGs.method == 'pcgrad':
        CONFIGs.BatchSize = 32
    elif CONFIGs.backbone == 'retizero':
        CONFIGs.BatchSize = 64
    else:
        CONFIGs.BatchSize = 64

    main(CONFIGs)



