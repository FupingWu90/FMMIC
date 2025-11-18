# -*- coding:utf-8 -*-

import os
os.environ['HF_HOME'] = "/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/TIMM"
import sys
import argparse
import torch
from torch.backends import cudnn
from Common_Trainer import Trainer
from dataset import *
import numpy as np
import datetime
from utils_useless import *
import timm

torch.multiprocessing.set_sharing_strategy('file_system')

print(os.environ.get('HF_HOME'))

""" Path Config """

embedding_dim_info = {'SAM_vit_b':256, 'SAM_vit_l':256, 'SAM_vit_h':256, 'MedSAM_vit_b':256,'DINO_vit_small':384, 'DINO_vit_base':768, 'DINO_vit_large':1024,'dinov2_vits14':384, 'dinov2_vitb14':768, 'dinov2_vitl14':1024, 'dinov2_vitg14':1536}
embedding_patchsize_info = {'SAM_vit_b':16, 'SAM_vit_l':16, 'SAM_vit_h':16, 'MedSAM_vit_b':16,'DINO_vit_small':16, 'DINO_vit_base':16, 'DINO_vit_large':16,'dinov2_vits14':14, 'dinov2_vitb14':14, 'dinov2_vitl14':14, 'dinov2_vitg14':14}
Dataset_Name = ['DermaMNIST','PneumoniaMNIST','OrganAMNIST','PathMNIST','BreastMNIST','OCTMNIST','RetinaMNIST','BloodMNIST','TissueMNIST','OrganCMNIST','OrganSMNIST','ChestMNIST']
# LR_List1 = [1e-3,1e-4,1e-5,1e-6]
LR_List2 = [1e-4,1e-5,1e-6]
train_num_per_cls_List = [10,20,30,40,50]

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



def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


#add_path('/home/wfp/2020-Semi-PU/TPAMI_Rebuttal_R2/REFUGE/CPS/Uniform_Framework/furnace')

def training_validation(configs):
    # pretrain
    cur_path = os.path.abspath(os.curdir)
    SAVE_DIR_Prefix = cur_path + '/' + '{}_{}_{}/'.format(configs.dataname,configs.img_size,configs.img_resize) \
                + '{}/{}_{}/'.format(configs.train_num_per_cls,configs.backbone,configs.backbone_update)+ '{}_Iter{}/'.format(configs.bb_lr,configs.total_Iter_Num)+ '{}_lr{}_{}_bs{}/'.format(configs.Optim, configs.lr,configs.lr_decay,configs.BatchSize)

    if not os.path.exists(SAVE_DIR_Prefix):
        # os.mkdir(SAVE_DIR)
        os.makedirs(SAVE_DIR_Prefix)



    for sample_random_seed in configs.random_seeds:
        save_result_file = SAVE_DIR_Prefix + 'seed{}/test_ndarray.npy'.format(sample_random_seed)
        if os.path.exists(save_result_file):
            continue

        if sample_random_seed < 100:

            control_seed(sample_random_seed)

            train_starttime = datetime.datetime.now()


            SAVE_DIR_Seed = SAVE_DIR_Prefix + 'seed{}/'.format(sample_random_seed)
            if not os.path.exists(SAVE_DIR_Seed):
                # os.mkdir(SAVE_DIR)
                os.makedirs(SAVE_DIR_Seed)

            ## model
            configs.num_cls = MedMNIST_INFO[configs.dataname]
            out_ch = configs.num_cls
            if configs.num_cls == 2:
                out_ch = 1


            Model_Main = timm.create_model(Backbon_Dict[configs.backbone], pretrained=True, num_classes=out_ch)


            ## dataloader
            if 'dino2' in configs.backbone:
                configs.img_resize = 518
            mean, std = Model_Main.default_cfg['mean'], Model_Main.default_cfg['std']
            train_dataset = MedMNIST_Labels(dataset_name=configs.dataname, train='train', img_size=configs.img_size,
                                            img_resize=configs.img_resize, download=True,
                                            sample_num_per_cls=configs.train_num_per_cls,semi = configs.backbone_update, seed=sample_random_seed,mean = mean, std=std)
            val_dataset = MedMNIST_Labels(dataset_name=configs.dataname, train='val', img_size=configs.img_size,
                                          img_resize=configs.img_resize, download=True,mean = mean, std=std)
            test_dataset = MedMNIST_Labels(dataset_name=configs.dataname, train='test', img_size=configs.img_size,
                                           img_resize=configs.img_resize, download=True,mean = mean, std=std)


            if len(train_dataset) < configs.BatchSize :
                configs.BatchSize =len(train_dataset)

            train_loaders = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.BatchSize,
                                                        shuffle=True,
                                                        num_workers=0, drop_last=True)
            val_loaders = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, num_workers=0)
            test_loaders = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, num_workers=0)

            configs.iter_per_epoch = len(train_loaders)
            configs.total_epoch = configs.total_Iter_Num // configs.iter_per_epoch+10
            if configs.total_epoch == 0:
                configs.total_epoch = 1


            # training
            checkpoint_file = SAVE_DIR_Seed + 'checkpoint.pth.tar'
            checkpoint = None
            if os.path.exists(checkpoint_file):
                try:
                    checkpoint = torch.load(checkpoint_file)
                    print(f"Successfully loaded {checkpoint_file}")

                except Exception as e:
                    checkpoint = None
                    print(f"Error loading {checkpoint_file}: {e}")

            trainer = Trainer(Model_Main,checkpoint,configs,train_loaders,val_loaders, test_loaders,SAVE_DIR_Seed)
            trainer.train()

            train_endtime = datetime.datetime.now()
            print('train time:', (train_endtime - train_starttime).seconds)


    total_metrics = np.zeros((1, 2))
    for cv in range(3):
        SAVE_DIR_Seed = SAVE_DIR_Prefix + 'seed{}/'.format(cv)

        metric_ndarray = np.load(SAVE_DIR_Seed + 'test_ndarray.npy', allow_pickle=True)

        total_metrics = np.concatenate((total_metrics, metric_ndarray), axis=0)

    average_accuracy = np.mean(total_metrics[1:], axis=0)
    std_accuracy = np.std(total_metrics[1:], axis=0)

    with open("%s/fivef_testout_index.txt" % SAVE_DIR_Prefix, "a") as f:
        f.writelines(
            ["average_accuracy of acc,auc:", "", str(average_accuracy.tolist()), "std_accuracy:", "",
             str(std_accuracy.tolist())
             ])



def main(configs):


    training_validation(configs)


if __name__ == '__main__':


    ## set parameters
    parser = argparse.ArgumentParser()

    # network param
    parser.add_argument('--backbone', type=str, default='vgg16')


    # dataset param
    parser.add_argument('--job_array_id', type=int, default=-10)
    parser.add_argument('--dataname', type=str, default='PathMNIST')  # 'OrganCMNIST', 'OrganSMNIST', 'PathMNIST'
    parser.add_argument('--img_size', type=int, default=224) # 28, 62, 128, 224
    parser.add_argument('--img_resize', type=int, default=224)  # DINO: 224 ,  SAM:1024
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_cls', type=int, default=3)

    parser.add_argument('--train_num_per_cls',type=int, default=10000) #

    parser.add_argument('--random_seeds', nargs='+', type=int, default=[0, 1, 2])


    # train param
    parser.add_argument('--BatchSize', type=int, default=128)
    parser.add_argument('--iter_per_epoch', type=int, default=100)
    parser.add_argument('--GPU_ID', type=str, default='0')

    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--Optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_iters', type=int, default=200)

    parser.add_argument('--backbone_update', type=str, default='sft') # 'sft', 'lp','sft_lora','semi_sft', 'semi_lp','semi_sft_lora'
    parser.add_argument('--bb_lr', type=float, default=0)




    parser.add_argument('--total_Iter_Num', type=int, default=15000) #150
    parser.add_argument('--total_epoch', type=int, default=100)  # 200 LR_IRER


    parser.add_argument('--TRAIN_USE_CHECKPOINT', type=bool, default=False)


    CONFIGs = parser.parse_args()

    if CONFIGs.job_array_id > -1:
        if 'lp' in CONFIGs.backbone_update:
            data_id, train_num_id = divmod(CONFIGs.job_array_id, 5)
            CONFIGs.dataname = Dataset_Name[data_id]
            CONFIGs.train_num_per_cls = train_num_per_cls_List[train_num_id]
        elif 'scratch' in CONFIGs.backbone_update:
            data_id, train_num_id = divmod(CONFIGs.job_array_id, 5)
            CONFIGs.dataname = Dataset_Name[data_id]
            CONFIGs.train_num_per_cls = train_num_per_cls_List[train_num_id]
        else:
            data_id, lr_train_num_id = divmod(CONFIGs.job_array_id , 15)
            train_num_id, lr_id = divmod(lr_train_num_id,3)
            CONFIGs.dataname = Dataset_Name[data_id]
            CONFIGs.train_num_per_cls = train_num_per_cls_List[train_num_id]

            CONFIGs.bb_lr = LR_List2[lr_id]


    if 'semi' in CONFIGs.backbone_update:
        if CONFIGs.backbone == 'dino2_base' and 'sft' in CONFIGs.backbone_update:
            CONFIGs.BatchSize = 16
        elif CONFIGs.backbone == 'dino2_base' and 'scratch' in CONFIGs.backbone_update:
            CONFIGs.BatchSize = 16
        else:
            CONFIGs.BatchSize = 32


    else:
        if CONFIGs.backbone == 'dino2_base' and 'sft' in CONFIGs.backbone_update:
            CONFIGs.BatchSize = 32
        elif CONFIGs.backbone == 'dino2_base' and 'scratch' in CONFIGs.backbone_update:
            CONFIGs.BatchSize = 32
        else:
            CONFIGs.BatchSize = 64

    cudnn.benchmark = True



    main(CONFIGs)