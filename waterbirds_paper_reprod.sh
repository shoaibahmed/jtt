#!/bin/bash


# optimized for up to 300 epochs with batch size 64, using batch normalization and no data augmentation
# ERM, this yields learning rate 1e-3 and l2 regularization 1e-4
# JTT, this yields learning rate 1e-5 and l2 regularization 1.
# Grid search: JTT yields T = 60 epochs and λup = 100


python generate_downstream.py --exp_name CUB_reproduce_paper --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM

bash results/CUB/CUB_reproduce_paper/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python process_training.py --exp_name CUB_reproduce_paper --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 60 --deploy

for upweight in 20 50 100; do
    bash results/CUB/CUB_reproduce_paper/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
done

python analysis.py --exp_name CUB_reproduce_paper/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/ --dataset CUB
