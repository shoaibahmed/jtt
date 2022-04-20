#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python generate_downstream.py --exp_name CUB_probes_exp --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --include_probes --no_wandb

bash results/CUB/CUB_probes_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

# python process_training.py --exp_name CUB_sample_exp --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 60 --deploy

# for upweight in 20 50 100; do
#     bash results/CUB/CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# done

# python analysis.py --exp_name CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/ --dataset CUB
