#!/bin/bash


#SBATCH -D /UKBBRetinal/CAGrad
#SBATCH -A mtl.prj
#SBATCH -J di_caLoR
#SBATCH -o /UKBBRetinal/CAGrad/train_out_error/train-di_caLoR.out
#SBATCH -e /UKBBRetinal/CAGrad/train_out_error/train-di_caLoR.error

module load Python/3.11.3-GCCcore-12.3.0

source /python/mypython311-skylake/bin/activate

cd /UKBBRetinal/CAGrad

python CAGrad_Main.py --task_id_list 0 4 5 6 7 8 --method 'cagrad' --job_array_id 0 --Iter_Num 15000 --Train_Data 'L' --Optim 'AdamW' --lr 0.001 --lr_decay 0.9 --random_seeds 0 --backbone 'dino2_base' --backbone_update 'lora' --bb_lr 1e-3 --lr_decay_iters 200 --BatchSize 64 --img_resize 224
