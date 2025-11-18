# Using CAGrad for multi-task learning

1. prepare the retinal images from UKBB (or your own datasets), and preprocess them with AutoMorph into standadized images.
2. resizing them into 224*224 or 518*518 (for Dino-v2)
3. set the arguments in train_array_sh/dino_cagrad_lora.sh, like --job_array_id , --backbone, --backbone_update for backbone and updating strategy selection.
4. run: sh dino_cagrad_lora.sh
