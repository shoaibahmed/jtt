#!/bin/bash


# optimized for up to 300 epochs with batch size 64, using batch normalization and no data augmentation
# ERM, this yields learning rate 1e-3 and l2 regularization 1e-4
# JTT, this yields learning rate 1e-5 and l2 regularization 1.
# Grid search: JTT yields T = 60 epochs and λup = 100

# job_name=waterbirds-trajectories-eval
# srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=50G \
#     --kill-on-bad-exit --job-name ${job_name} --nice=0 --time 3-00:00:00 \
#     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#     ./waterbirds_trajectories.sh > ./logs/${job_name}.log 2>&1 &

export WANDB_MODE=offline

python generate_downstream.py --exp_name CUB_trajectories \
    --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --seed 1

bash results/CUB/CUB_trajectories/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python get_final_epoch.py ./results/CUB/CUB_trajectories/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/
final_epoch=$?
echo "Using final epoch to be: "$final_epoch

python process_training_trajectories.py --exp_name CUB_trajectories --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch ${final_epoch} --deploy

for upweight in 20 50 100; do
    bash results/CUB/CUB_trajectories/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
done

python analysis.py --exp_name CUB_trajectories/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/ --dataset CUB
