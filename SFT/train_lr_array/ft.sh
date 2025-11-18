#!/bin/bash


#SBATCH -D FMMIC/SFT
#SBATCH -A sft.prj
#SBATCH -J ft0_lr3
#SBATCH -o FMMIC/SFT/train_out_error/train-ft0_lr3.out
#SBATCH -e FMMIC/SFT/train_out_error/train-ft0_lr3.err

module load Python/3.11.3-GCCcore-12.3.0

source /python/mypython311-skylake/bin/activate

cd FMMIC/SFT

python Common_Main.py --job_array_id 0 --total_Iter_Num 15000 --BatchSize 64 --Optim 'AdamW' --lr 0.001 --lr_decay 0.9 --random_seeds 0 1 2 --GPU_ID '3' --backbone 'vgg16' --backbone_update 'ft' --bb_lr 1e-3 --dataname 'PathMNIST' --img_size 224 --img_resize 224 --train_num_per_cls 10000
