#!/bin/bash


#SBATCH -D /MedMNIST/SemiSFT
#SBATCH -A ssl.prj
#SBATCH -J s_s_lor0
#SBATCH -o /MedMNIST/SemiSFT/train_out_error/train-semi_sft_lora0.out
#SBATCH -e /MedMNIST/SemiSFT/train_out_error/train-semi_sft_lora0.err



#ml use -a /apps/eb/2020b/skylake/modules/all
# note that you must load whichever main Python module you used to create your virtual environments before activating the virtual environment
module load Python/3.11.3-GCCcore-12.3.0
#module load Python/3.8.6-GCCcore-10.2.0

source /python/mypython311-skylake/bin/activate

cd /MedMNIST/SemiSFT

python Common_Main.py --job_array_id 0 --train_num_per_cls 10 --total_Iter_Num 10000 --BatchSize 64 --Optim 'AdamW' --lr 0.001 --lr_decay 0.9 --random_seeds 0 1 2 --GPU_ID '3' --backbone 'densenet121' --backbone_update 'semi_sft_lora' --bb_lr 0 --dataname 'PathMNIST' --img_size 224 --img_resize 224
