#!/bin/bash

python generate_downstream.py --exp_name CUB_sample_exp --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM

bash results/CUB/CUB_sample_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python process_training.py --exp_name CUB_sample_exp --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 60 --deploy

bash results/CUB/CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/JTT_upweight_100_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python analysis.py --exp_name CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/ --dataset CUB
