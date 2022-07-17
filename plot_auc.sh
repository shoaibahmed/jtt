#!/bin/bash

seed=2
final_epoch=1  # For CelebA
python plot_auc.py --exp_name CelebA_trajectories_seed_${seed} \
    --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1 \
    --lr 1e-05 --weight_decay 0.1 --final_epoch ${final_epoch}

seed=2
final_epoch=60  # For Waterbirds
python plot_auc.py --exp_name CUB_trajectories_seed_${seed} \
    --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 \
    --lr 1e-05 --weight_decay 1.0 --final_epoch ${final_epoch}
