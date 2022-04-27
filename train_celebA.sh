#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 50 --lr 1e-5 --weight_decay 0.1 --method ERM

bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python process_training.py --exp_name CelebA_sample_exp --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1 --lr 1e-05 --weight_decay 0.1 --final_epoch 1 --deploy

sbatch results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python analysis.py --exp_name CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/ --dataset CelebA
