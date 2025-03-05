#!/bin/bash


#SBATCH -D /gpfs3/well/papiez/users/cub991/PJ2022/EPLF/MedMNIST/CommonNet
#SBATCH -A papiez.prj
#SBATCH -J ft0_lr3
#SBATCH --array 0
#SBATCH -o /gpfs3/well/papiez/users/cub991/PJ2022/EPLF/MedMNIST/CommonNet/train_out_error/train-ft0_lr6.out
#SBATCH -e /gpfs3/well/papiez/users/cub991/PJ2022/EPLF/MedMNIST/CommonNet/train_out_error/train-ft0_lr6.err



#ml use -a /apps/eb/2020b/skylake/modules/all
# note that you must load whichever main Python module you used to create your virtual environments before activating the virtual environment
module load Python/3.11.3-GCCcore-12.3.0
#module load Python/3.8.6-GCCcore-10.2.0


# Activate the ivybridge or skylake version of your python virtual environment
# NB The environment variable MODULE_CPU_TYPE will evaluate to ivybridge or skylake as appropriate
#source /well/papiez/users/cub991/python/mypython38-${MODULE_CPU_TYPE}/bin/activate
source /well/papiez/users/cub991/python/mypython311-skylake/bin/activate

cd /gpfs3/well/papiez/users/cub991/PJ2022/EPLF/MedMNIST/CommonNet

python Common_Main.py --job_array_id ${SLURM_ARRAY_TASK_ID} --total_Iter_Num 15000 --BatchSize 64 --Optim 'AdamW' --lr 0.001 --lr_decay 0.9 --random_seeds 0 1 2 --GPU_ID '3' --backbone 'vgg16' --backbone_update 'ft' --bb_lr 1e-3 --dataname 'PathMNIST' --img_size 224 --img_resize 224 --train_num_per_cls 10000
#python BLOMTL_FlattenNoise_Main.py --block_type 'concat' --task1_ratio 0.5 --temperature 1.0 --sample_num_ratio 1.0 --task1_labels 0 1 2 3 4 --task2_labels 5 6 7 8 9 --warm_task1_Iter_Num 1000 --warm_task2_Iter_Num 1000 --Task1_BatchSize 256 --sample_random_seeds 1 2 3 4 5 --task_lr_end 1e-6 --task1_Iter_Num 100 --task2_Iter_Num 100 --OUTER_ITER 1000 --random_times 8 --random_type 'Uniform' --bound_value 0.1 --proto_loss True --proto_loss_w 0.0 &&
#
#python BLOMTL_FlattenNoise_Main.py --block_type 'concat' --task1_ratio 0.5 --temperature 1.0 --sample_num_ratio 1.0 --task1_labels 0 1 2 3 4 --task2_labels 5 6 7 8 9 --warm_task1_Iter_Num 1000 --warm_task2_Iter_Num 1000 --Task1_BatchSize 256 --sample_random_seeds 1 2 3 4 5 --task_lr_end 1e-6 --task1_Iter_Num 100 --task2_Iter_Num 100 --OUTER_ITER 1000 --random_times 8 --random_type 'Uniform' --bound_value 0.1 --proto_loss True --proto_loss_w 1e-1 &&
#
#python BLOMTL_FlattenNoise_Main.py --block_type 'concat' --task1_ratio 0.5 --temperature 1.0 --sample_num_ratio 1.0 --task1_labels 0 1 2 3 4 --task2_labels 5 6 7 8 9 --warm_task1_Iter_Num 1000 --warm_task2_Iter_Num 1000 --Task1_BatchSize 256 --sample_random_seeds 1 2 3 4 5 --task_lr_end 1e-6 --task1_Iter_Num 100 --task2_Iter_Num 100 --OUTER_ITER 1000 --random_times 8 --random_type 'Uniform' --bound_value 0.1 --proto_loss True --proto_loss_w 1e1 &&
#
#python BLOMTL_FlattenNoise_Main.py --block_type 'concat' --task1_ratio 0.5 --temperature 1.0 --sample_num_ratio 1.0 --task1_labels 0 1 2 3 4 --task2_labels 5 6 7 8 9 --warm_task1_Iter_Num 1000 --warm_task2_Iter_Num 1000 --Task1_BatchSize 256 --sample_random_seeds 1 2 3 4 5 --task_lr_end 1e-6 --task1_Iter_Num 100 --task2_Iter_Num 100 --OUTER_ITER 1000 --random_times 8 --random_type 'Uniform' --bound_value 0.1 --proto_loss True --proto_loss_w 1e-2 &&
#
#python BLOMTL_FlattenNoise_Main.py --block_type 'concat' --task1_ratio 0.5 --temperature 1.0 --sample_num_ratio 1.0 --task1_labels 0 1 2 3 4 --task2_labels 5 6 7 8 9 --warm_task1_Iter_Num 1000 --warm_task2_Iter_Num 1000 --Task1_BatchSize 256 --sample_random_seeds 1 2 3 4 5 --task_lr_end 1e-6 --task1_Iter_Num 100 --task2_Iter_Num 100 --OUTER_ITER 1000 --random_times 8 --random_type 'Uniform' --bound_value 0.1 --proto_loss True --proto_loss_w 1e2

# python Independant_Edge_Main.py --block_type 'paral_se_add' --feats 128 64 32 16 8 3 --task1_ratio 0.5 --temperature 1.0 --edge_loss_type 'balanced' --edge_loss_pos_w 0.95 --Edge_Iter_Num 8000 --Edge_BatchSize 36 --sample_num 400 --data_aug_times 1 --sample_random_seeds 1 --vali_Num 5 &&

# python Independant_Edge_Main.py --block_type 'paral_se_add' --feats 128 64 32 16 8 3 --task1_ratio 0.5 --temperature 1.0 --edge_loss_type 'balanced' --edge_loss_pos_w -1 --Edge_Iter_Num 8000 --Edge_BatchSize 36 --sample_num 400 --data_aug_times 1 --sample_random_seeds 1 --vali_Num 5 

# continue to use your python venv as normal