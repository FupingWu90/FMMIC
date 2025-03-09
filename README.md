# FMMIC
[Rethinking Non-Medical Foundation Models for Medical Image Classification](paper link: https://arxiv.org/abs/2501.14685)


## run
1. In train_lr_array/ft.sh, change the --job_array_id to set the learning rate and dataset, change --backbone to set the backbone you choose;
2. revise the save or root path in Common_Main.py and dataset.py
3. run ``sh ft.sh'' in Slurm system


## Acknowledgement
This work is inspired by [Rethinking model prototyping through the MedMNIST+ dataset collection]([Rethinking model prototyping through the MedMNIST+ dataset collection](https://www.nature.com/articles/s41598-025-92156-9)).
