import os
import glob
import natsort

import pandas as pd
import numpy as np 

import sklearn.neighbors
from sklearn.metrics import RocCurveDisplay

import matplotlib.pyplot as plt
from scipy.special import softmax
import argparse
import subprocess


def main(args):
    final_epoch = args.final_epoch
    dataset = args.dataset
    
    # CHANGE THESE FOLDERS
    exp_name = args.exp_name
    folder_name = args.folder_name
    data_dir = f"results/{args.dataset}/{exp_name}/{folder_name}/model_outputs/"
    if args.dataset == 'CelebA':
        metadata_path = "./celebA/data/metadata.csv"
    elif args.dataset == 'MultiNLI':
        metadata_path = "./multinli/data/metadata.csv"
    elif args.dataset == 'CUB':
        metadata_path = "./cub/data/waterbird_complete95_forest2water2/metadata.csv"
    elif args.dataset == "jigsaw":
        metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    else: 
        raise NotImplementedError 
    
    # Load the original metadata
    original_df = pd.read_csv(metadata_path)
    total_ex = len(original_df)
    
    # Load in train df and wrong points, this is the main part
    train_df = pd.read_csv(os.path.join(data_dir, f"output_train_epoch_{final_epoch}.csv"))
    train_df = train_df.sort_values(f"indices_None_epoch_{final_epoch}_val")
    # train_df["wrong_1_times"] = (1.0 * (train_df[f"y_pred_None_epoch_{final_epoch}_val"] != train_df[f"y_true_None_epoch_{final_epoch}_val"])).apply(np.int64)
    # print("Total wrong", np.sum(train_df['wrong_1_times']), "Total points", len(train_df))
    
    # Load the full loss trajectories
    train_files = natsort.natsorted(glob.glob(os.path.join(data_dir, "output_train_*.csv")))
    val_files = natsort.natsorted(glob.glob(os.path.join(data_dir, "output_val_*.csv")))
    
    loss_vals_sorted_list = []
    log_iter = 50
    for i, (train_file, val_file) in enumerate(zip(train_files, val_files)):
        if i % log_iter == 0:
            print("Loading file:", train_file)
        parts = os.path.splitext(os.path.split(train_file)[1])[0].split("_")
        assert parts[0] == "output"
        assert parts[1] == "train"
        assert parts[2] == "epoch"
        epoch = int(parts[3])
        
        # Initialize the loss list
        current_loss_list = [None for _ in range(total_ex)]
        
        # Read the train file
        df = pd.read_csv(train_file)
        indices = df[f"indices_None_epoch_{epoch}_val"].to_numpy()
        loss_vals = df[f"loss_None_epoch_{epoch}_val"].to_numpy()
        for i in range(len(indices)):
            assert current_loss_list[indices[i]] is None
            current_loss_list[indices[i]] = loss_vals[i]
        
        # Read the validation file
        df = pd.read_csv(val_file)
        indices = df[f"indices_None_epoch_{epoch}_val"].to_numpy()
        loss_vals = df[f"loss_None_epoch_{epoch}_val"].to_numpy()
        for i in range(len(indices)):
            assert current_loss_list[indices[i]] is None
            current_loss_list[indices[i]] = loss_vals[i]
        
        # Add the values in the final list
        loss_vals_sorted_list.append(current_loss_list)
    
    loss_vals_np = np.array(loss_vals_sorted_list)
    loss_vals_np = np.transpose(loss_vals_np, (1, 0))
    print("Loss values shape:", loss_vals_np.shape)
    
    # Maps from target and group attribute to whether the example is majority or minority group example
    # 0 refers to majority while 1 refers to minority
    # label_map = {(0, 0): 0, (1, 1): 0, (0, 1): 1, (1, 0): 1}
    
    if dataset == "CUB":
        # merged_csv["spurious"] = merged_csv['y'] != merged_csv["place"]
        metadata_target = original_df["y"].to_numpy()
        metadata_attrib = original_df["place"].to_numpy()
        metadata_spurious = (metadata_target != metadata_attrib).astype(np.int64)
        print("Number of minority group instances for CUB:", np.sum(metadata_spurious))
    elif dataset == "CelebA":
        # merged_csv = merged_csv.replace(-1, 0)
        # disagreement = np.sum(merged_csv[merged_csv["split"] == 0]["Blond_Hair"] != merged_csv[merged_csv["split"] == 0][f"y_true_None_epoch_{final_epoch}_val"])
        # assert 0 == disagreement, disagreement
        # merged_csv["spurious"] = (merged_csv["Blond_Hair"] == merged_csv["Male"]) 
        metadata_target = original_df["Blond_Hair"].to_numpy()
        metadata_attrib = original_df["Male"].to_numpy()
        metadata_spurious = (metadata_target == metadata_attrib).astype(np.int64)
        print("Number of minority group instances for CelebA:", np.sum(metadata_spurious))
    else: 
        raise NotImplementedError
    
    metadata_split = original_df["split"].to_numpy()
    print(len(metadata_target), len(metadata_attrib), len(metadata_split))

    val_data_np = np.stack([loss_vals_np[i] for i in range(len(loss_vals_np)) if metadata_split[i] == 1], axis=0).astype(np.float32)
    val_data_np_targets = np.array([metadata_spurious[i] for i in range(len(loss_vals_np)) if metadata_split[i] == 1]).astype(np.int64)
    print("Validation data:", val_data_np.shape, val_data_np_targets.shape)
    
    train_data_np = np.stack([loss_vals_np[i] for i in range(len(loss_vals_np)) if metadata_split[i] == 0], axis=0).astype(np.float32)
    train_data_np_targets = np.array([metadata_spurious[i] for i in range(len(loss_vals_np)) if metadata_split[i] == 0]).astype(np.int64)
    print("Training data:", train_data_np.shape, train_data_np_targets.shape)
    
    n_neighbors = 20
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(val_data_np, val_data_np_targets)
    
    assert 0. < args.conf_threshold_traj < 1., args.conf_threshold_traj
    prediction_probs = clf.predict_proba(train_data_np)  # 0: Majority group; 1: Minority group
    # prediction = np.argmax(prediction_probs, axis=1)
    prediction = prediction_probs[:, 1] > args.conf_threshold_traj
    print(f"Probs shape: {prediction_probs.shape} / Prediction shape: {prediction.shape} / Confidence thresh: {args.conf_threshold_traj}")
    
    assert len(train_df) == len(prediction), f"{len(train_df)} != {len(prediction)}"
    train_df["wrong_1_times"] = prediction.astype(np.int64)
    train_df["wrong_1_times_prob"] = prediction_probs[:, 1].astype(np.float32)  # Prob for class minority
    print("Total minority", np.sum(train_df['wrong_1_times']), "Total points", len(train_df))
    
    # Merge with original features (could be optional)
    original_train_df = original_df[original_df["split"] == 0]
    if dataset == "CelebA" or dataset == "jigsaw" or dataset == "MultiNLI":
        if "Unnamed: 0" in original_train_df in original_train_df.columns:
            original_train_df = original_train_df.drop(['Unnamed: 0'], axis=1)
        else:
            print("!! Warning: Unnamed:0 column not found which was expected by the original code...")

    merged_csv = original_train_df.join(train_df.set_index(f"indices_None_epoch_{final_epoch}_val"))
    if dataset == "CUB":
        merged_csv["spurious"] = merged_csv['y'] != merged_csv["place"]
    elif dataset == "CelebA":
        merged_csv = merged_csv.replace(-1, 0)
        disagreement = np.sum(merged_csv[merged_csv["split"] == 0]["Blond_Hair"] != merged_csv[merged_csv["split"] == 0][f"y_true_None_epoch_{final_epoch}_val"])
        assert 0 == disagreement, disagreement
        merged_csv["spurious"] = (merged_csv["Blond_Hair"] == merged_csv["Male"]) 
    elif dataset == "jigsaw":
        merged_csv["spurious"] = merged_csv["toxicity"] >= 0.5
    elif dataset == "MultiNLI":
        merged_csv["spurious"] = (
                (merged_csv["gold_label"] == 0)
                & (merged_csv["sentence2_has_negation"] == 0)
            ) | (
                (merged_csv["gold_label"] == 1)
                & (merged_csv["sentence2_has_negation"] == 1)
            )
    else: 
        raise NotImplementedError
    print("Number of spurious", np.sum(merged_csv['spurious']))
    
    # Make columns for our spurious and our nonspurious
    merged_csv["our_spurious"] = merged_csv["spurious"] & merged_csv["wrong_1_times"]
    merged_csv["our_nonspurious"] = (merged_csv["spurious"] == 0) & merged_csv["wrong_1_times"]
    print("Number of our spurious: ", np.sum(merged_csv["our_spurious"]))
    print("Number of our nonspurious:", np.sum(merged_csv["our_nonspurious"]))
    
    train_probs_df= merged_csv.fillna(0)
    
    # Output spurious recall and precision
    spur_precision = np.sum(
            (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
        ) / np.sum((merged_csv[f"wrong_1_times"] == 1))
    print("Spurious precision", spur_precision)
    spur_recall = np.sum(
        (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
    ) / np.sum((merged_csv["spurious"] == 1))
    print("Spurious recall", spur_recall)
    
    # Find confidence (just in case doing threshold)
    if dataset == "MultiNLI":
        raise NotImplementedError
        probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1", f"pred_prob_None_epoch_{final_epoch}_val_2"]]), axis = 1)
        train_probs_df["probs_0"] = probs[:,0]
        train_probs_df["probs_1"] = probs[:,1]
        train_probs_df["probs_2"] = probs[:,2]
        train_probs_df["confidence"] = (train_probs_df['gold_label']==0) * train_probs_df["probs_0"] + (train_probs_df['gold_label']==1) * train_probs_df["probs_1"] + (train_probs_df['gold_label']==2) * train_probs_df["probs_2"]
    else:
        probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1"]]), axis = 1)
        # train_probs_df["probs_0"] = probs[:,0]
        # train_probs_df["probs_1"] = probs[:,1]
        train_probs_df["probs_0"] = prediction_probs[:,0]
        train_probs_df["probs_1"] = prediction_probs[:,1]
        if dataset == 'CelebA':
            train_probs_df["confidence"] = train_probs_df["Blond_Hair"] * train_probs_df["probs_1"] + (1 - train_probs_df["Blond_Hair"]) * train_probs_df["probs_0"]
        elif dataset == 'CUB':
            train_probs_df["confidence"] = train_probs_df["y"] * train_probs_df["probs_1"] + (1 - train_probs_df["y"]) * train_probs_df["probs_0"]
        elif dataset == 'jigsaw':
            train_probs_df["confidence"] = (train_probs_df["toxicity"] >= 0.5) * train_probs_df["probs_1"] + (train_probs_df["toxicity"] < 0.5)  * train_probs_df["probs_0"]
        # train_probs_df["confidence"] = train_probs_df["wrong_1_times"].apply(np.float32)
    
    train_probs_df[f"confidence_thres{args.conf_threshold}"] = (train_probs_df["confidence"] < args.conf_threshold).apply(np.int64)
    # if dataset == 'CelebA':
    #     assert(np.sum(train_probs_df[f"confidence_thres{args.conf_threshold}"] != train_probs_df["wrong_1_times"]) == 0)
    
    # Save csv into new dir for the run, and generate downstream runs
    if not os.path.exists(f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"):
        os.makedirs(f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}")
    root = f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"
    
    # Plot the ROC curve
    RocCurveDisplay.from_predictions(merged_csv["spurious"], merged_csv["wrong_1_times_prob"])
    plt.savefig(os.path.join(root, "roc_curve.png"), dpi=300, box_inches="tight")

    train_probs_df.to_csv(f"{root}/metadata_aug.csv")
    root = f"{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"
    
    sbatch_command = (
            f"python generate_downstream.py --exp_name {root} --lr {args.lr} --weight_decay {args.weight_decay} --method JTT --dataset {args.dataset} --aug_col {args.aug_col}" + (f" --batch_size {args.batch_size}" if args.batch_size else "")
        )
    print(sbatch_command)
    if args.deploy:
        subprocess.run(sbatch_command, check=True, shell=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="CelebA", help="CUB, CelebA, or MultiNLI"
    )
    parser.add_argument(
        "--final_epoch",
        type=int,
        default=5,
        help="first epoch in training -- not significant i.e. only used to initialize the DF format",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--deploy", action="store_true", default=False)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold_traj", type=float, default=0.80)
    parser.add_argument("--aug_col", type=str, default='wrong_1_times')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()
    main(args)
    