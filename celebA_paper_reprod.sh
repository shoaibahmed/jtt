#!/bin/bash


# We train all approaches for up to 50 epochs with batch size 128
# ERM, this yields learning rate 1e-4 and l2 regularization 1e-4
# JTT, this yields learning rate 1e-5 and l2 regularization 1e-1
# Grid search: JTT yields T = 1 epoch and λup = 50

# job_name=celebA-paper-reprod
# srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=50G \
#     --kill-on-bad-exit --job-name ${job_name} --nice=0 --time 3-00:00:00 \
#     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#     ./celebA_paper_reprod.sh > ./logs/${job_name}.log 2>&1 &

export WANDB_MODE=offline

python generate_downstream.py --exp_name CelebA_reproduce_paper --dataset CelebA --n_epochs 50 --lr 1e-5 --weight_decay 0.1 --method ERM

bash results/CelebA/CelebA_reproduce_paper/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python process_training.py --exp_name CelebA_reproduce_paper --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1 --lr 1e-05 --weight_decay 0.1 --final_epoch 1 --deploy

bash results/CelebA/CelebA_reproduce_paper/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python analysis.py --exp_name CelebA_reproduce_paper/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/ --dataset CelebA
