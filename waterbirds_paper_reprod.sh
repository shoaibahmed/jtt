#!/bin/bash


# optimized for up to 300 epochs with batch size 64, using batch normalization and no data augmentation
# ERM, this yields learning rate 1e-3 and l2 regularization 1e-4
# JTT, this yields learning rate 1e-5 and l2 regularization 1.
# Grid search: JTT yields T = 60 epochs and λup = 100

# job_name=waterbirds-paper-reprod
# srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=50G \
#     --kill-on-bad-exit --job-name ${job_name} --nice=0 --time 3-00:00:00 \
#     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#     ./waterbirds_paper_reprod.sh > ./logs/${job_name}.log 2>&1 &

export WANDB_MODE=offline

python generate_downstream.py --exp_name CUB_reproduce_paper --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM

bash results/CUB/CUB_reproduce_paper/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python process_training.py --exp_name CUB_reproduce_paper --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 60 --deploy

for upweight in 20 50 100; do
    bash results/CUB/CUB_reproduce_paper/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
done

python analysis.py --exp_name CUB_reproduce_paper/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/ --dataset CUB
