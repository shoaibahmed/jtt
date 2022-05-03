#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python generate_downstream.py --exp_name CelebA_probes_10_exp --dataset CelebA --n_epochs 50 --lr 1e-5 --weight_decay 0.1 --method ERM --include_probes

# bash results/CelebA/CelebA_probes_10_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

final_epoch=43  # TBD

python process_training.py --exp_name CelebA_probes_10_exp --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1 --lr 1e-05 --weight_decay 0.1 --final_epoch ${final_epoch} --deploy

bash results/CelebA/CelebA_probes_10_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch${final_epoch}/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python analysis.py --exp_name CelebA_probes_10_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch${final_epoch}/ --dataset CelebA
