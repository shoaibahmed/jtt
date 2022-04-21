#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python generate_downstream.py --exp_name CUB_probes_10_exp --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM --include_probes --no_wandb

# python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 128 --root_dir ./cub --n_epochs 300 --aug_col None --log_dir results/CUB/CUB_probes_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/model_outputs --metadata_path results/CUB/CUB_probes_exp/metadata_aug.csv --lr 1e-03 --weight_decay 1e-4 --up_weight 0 --metadata_csv_name metadata.csv --model resnet50 --use_bert_params 0 --loss_type erm --include_probes
bash results/CUB/CUB_probes_10_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

# python process_training.py --exp_name CUB_probes_exp --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 8 --deploy

# for upweight in 20 50 100; do
#     bash results/CUB/CUB_probes_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch8/JTT_upweight_${upweight}_epochs_300_lr_1e-05_weight_decay_1.0/job.sh
# done

# python analysis.py --exp_name CUB_probes_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch8/ --dataset CUB
