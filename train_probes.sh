#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# # python generate_downstream.py --exp_name CUB_probes_10_exp --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --include_probes --no_wandb

# # # python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 128 --root_dir ./cub --n_epochs 300 --aug_col None --log_dir results/CUB/CUB_probes_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/model_outputs --metadata_path results/CUB/CUB_probes_exp/metadata_aug.csv --lr 1e-03 --weight_decay 1e-4 --up_weight 0 --metadata_csv_name metadata.csv --model resnet50 --use_bert_params 0 --loss_type erm --include_probes
# # bash results/CUB/CUB_probes_10_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# final_epoch=79

# # python process_training.py --exp_name CUB_probes_10_exp --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch ${final_epoch} --deploy
# # for upweight in 20 50 100; do
# #     bash results/CUB/CUB_probes_10_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# # done

# python analysis.py --exp_name CUB_probes_10_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/ --dataset CUB




#### With corrupted inputs

# python generate_downstream.py --exp_name CUB_corrupted_probes --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --include_probes --use_mislabeled_examples --no_wandb

# bash results/CUB/CUB_corrupted_probes/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# final_epoch=29

# python process_training.py --exp_name CUB_corrupted_probes --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch ${final_epoch} --deploy
# for upweight in 20 50 100; do
#     bash results/CUB/CUB_corrupted_probes/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# done

# python analysis.py --exp_name CUB_corrupted_probes/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/ --dataset CUB





#### With corrupted inputs (new threshold of 85%)

# python generate_downstream.py --exp_name CUB_corrupted_probes_85p_acc --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --include_probes --use_mislabeled_examples --no_wandb

# bash results/CUB/CUB_corrupted_probes_85p_acc/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# python get_final_epoch.py ./results/CUB/CUB_corrupted_probes_85p_acc/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/
# final_epoch=$?
# print("Using final epoch to be:", final_epoch)

# python process_training.py --exp_name CUB_corrupted_probes_85p_acc --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch ${final_epoch} --deploy
# for upweight in 20 50 100; do
#     bash results/CUB/CUB_corrupted_probes_85p_acc/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# done

# python analysis.py --exp_name CUB_corrupted_probes_85p_acc/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/ --dataset CUB




#### With corrupted inputs 0.25 (new threshold of 60%)

python generate_downstream.py --exp_name CUB_corrupted_probes_0.25_60_thresh_acc --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --include_probes --use_mislabeled_examples --no_wandb

bash results/CUB/CUB_corrupted_probes_0.25_60_thresh_acc/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

# python get_final_epoch.py ./results/CUB/CUB_corrupted_probes_0.25_60_thresh_acc/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/
# final_epoch=$?
# echo "Using final epoch to be:"$final_epoch

# python process_training.py --exp_name CUB_corrupted_probes_0.25_60_thresh_acc --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch ${final_epoch} --deploy
# for upweight in 20 50 100; do
#     bash results/CUB/CUB_corrupted_probes_0.25_60_thresh_acc/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# done

# python analysis.py --exp_name CUB_corrupted_probes_0.25_60_thresh_acc/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch${final_epoch}/ --dataset CUB
