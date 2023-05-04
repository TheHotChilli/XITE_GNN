"""!
@file
@ingroup Training
@brief Main function for running all training cases using the OpenFace_Features_Dataset.
"""

"""!
@addtogroup Training
@{
"""

import torch
import importlib
import training_functions as train
import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
# from torch_geometric.nn import summary

sys.path.append(str(Path(__file__).parents[2]))
from XITE_GNN.datasets.OpenFace import OpenFace_Features_Dataset

def main():       
    """!
    Runs a training pipeline on different graph neural networks (GNNs) using the OpenFace_Features_Dataset dataset.
    It sets up the training configuration and model parameters, and then trains the different models on multiple binary classification problems.
    
    """
    # Training settings
    cfg = {}
    cfg["DATA_PATH"] = "/home/jbauske/00_Code/Python/Thesis/XITE_GNN/preprocessing/slice_and_feature_extraction/results/2023-05-03_19-22/video/features.csv"
    cfg["ADJ_PATH"] = "/home/jbauske/00_Code/Python/Thesis/XITE_GNN/preprocessing/graph_generation/results/adjacency_matrix/adjacency_matrix_delta_normalized.csv"
    cfg["PAIN_LABELS"] = [-3]
    cfg["BASE_LABLES"] = [100]
    cfg["MODEL_TYPE"] = "GCN"
    cfg["DEVICE"] = "cuda"
    cfg["NUM_FOLDS"] = 4
    cfg["NUM_EPOCHS"] = 50
    cfg["BATCH_SIZE"] = 256
    cfg["LR"] = 0.01
    cfg["WEIGHT_DECAY"] = 0
    cfg["DROPOUT"] = 0.5
    cfg["DROPOUT_G"] = 0.2
    "/home/jbauske/00_Code/Python/Thesis/XITE_GNN/preprocessing/graph_generation/results/adjacency_matrix/adjacency_matrix_delta_normalized.csv"

    pain_labels = [3,-3,6,-6]
    base_labels = [100,100,400,400]
    models = ["GCN", "GAT"]
    hidden_layers = {"shrinking2L": [1,0.5], "const2L":[1,1], "growing2L":[1,2],
                     "shrinking3L": [1,1,0.5], "const3L":[1,1,1], "growing3L":[1,1,2]}

    dt_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    root_out = os.path.dirname(os.path.abspath(__file__))
    # root_out = "/home/jbauske/00_Code/Python/XITE_GNN/results/training"
    root_out = os.path.join(root_out, "{}_{}FOLDS_{}EPOCHS".format(dt_now, cfg["NUM_FOLDS"], cfg["NUM_EPOCHS"]))
    if not os.path.exists(root_out):
        os.makedirs(root_out)

    col_idx = [f"{pain_labels[i]}vs{base_labels[i]}" for i in range(len(base_labels))]
    col_idx = pd.MultiIndex.from_product([col_idx, ["val", "sd"]])
    df_summary_train_acc_last = pd.DataFrame(0, 
                                    index=[f"{model_id}_{arch_id}" for arch_id in list(hidden_layers.keys()) for model_id in models],
                                    columns=col_idx
                                    )
    df_summary_test_acc_last = df_summary_train_acc_last.copy()
    df_summary_train_loss_last = df_summary_train_acc_last.copy()
    df_summary_test_loss_last = df_summary_train_acc_last.copy()
    df_summary_train_acc_all = df_summary_train_acc_last.copy()
    df_summary_test_acc_all = df_summary_train_acc_last.copy()
    df_summary_train_loss_all = df_summary_train_acc_last.copy()
    df_summary_test_loss_all = df_summary_train_acc_last.copy()

    for model_id in models:
        cfg["MODEL_TYPE"] = model_id
        for i in range(len(pain_labels)):
            cfg["PAIN_LABELS"] = [pain_labels[i]]
            cfg["BASE_LABLES"] = [base_labels[i]]
            for architecture_id, hidden_channels in hidden_layers.items():
                cfg["HIDDEN_CHANNELS"] = hidden_channels
                path_out = os.path.join(root_out, model_id, f"{architecture_id}", f"{pain_labels[i]}vs{base_labels[i]}")
                print("===========================")
                print("Model: {} {}".format(model_id, architecture_id))
                df_summary = run_training(cfg, path_out)
                
                # create summary
                row_id = f"{model_id}_{architecture_id}"
                col_id = f"{pain_labels[i]}vs{base_labels[i]}"
                df_summary_test_acc_last.loc[row_id, (col_id, "val")] = df_summary.loc["test_acc_last","val"]
                df_summary_test_acc_last.loc[row_id, (col_id, "sd")] = df_summary.loc["test_acc_last","sd"]
                df_summary_test_acc_all.loc[row_id, (col_id, "val")] = df_summary.loc["test_acc_all","val"]
                df_summary_test_acc_all.loc[row_id, (col_id, "sd")] = df_summary.loc["test_acc_all","sd"]

                df_summary_train_acc_last.loc[row_id, (col_id, "val")] = df_summary.loc["train_acc_last","val"]
                df_summary_train_acc_last.loc[row_id, (col_id, "sd")] = df_summary.loc["train_acc_last","sd"]
                df_summary_train_acc_all.loc[row_id, (col_id, "val")] = df_summary.loc["train_acc_all","val"]
                df_summary_train_acc_all.loc[row_id, (col_id, "sd")] = df_summary.loc["train_acc_all","sd"]

                df_summary_test_loss_last.loc[row_id, (col_id, "val")] = df_summary.loc["test_loss_last","val"]
                df_summary_test_loss_last.loc[row_id, (col_id, "sd")] = df_summary.loc["test_loss_last","sd"]
                df_summary_test_loss_all.loc[row_id, (col_id, "val")] = df_summary.loc["test_loss_all","val"]
                df_summary_test_loss_all.loc[row_id, (col_id, "sd")] = df_summary.loc["test_loss_all","sd"]

                df_summary_train_loss_last.loc[row_id, (col_id, "val")] = df_summary.loc["train_loss_last","val"]
                df_summary_train_loss_last.loc[row_id, (col_id, "sd")] = df_summary.loc["train_loss_last","sd"]
                df_summary_train_loss_all.loc[row_id, (col_id, "val")] = df_summary.loc["train_loss_all","val"]
                df_summary_train_loss_all.loc[row_id, (col_id, "sd")] = df_summary.loc["train_loss_all","sd"]

    print("Overall Summary:")
    print(df_summary_test_acc_last)

    pd.to_csv(df_summary_train_acc_last, os.path.join(root_out, "train_acc_last.csv"))
    pd.to_csv(df_summary_test_acc_last, os.path.join(root_out, "test_acc_last.csv"))
    pd.to_csv(df_summary_train_loss_last, os.path.join(root_out, "train_loss_last.csv"))
    pd.to_csv(df_summary_test_loss_last, os.path.join(root_out, "test_loss_last.csv"))

    pd.to_csv(df_summary_train_acc_all, os.path.join(root_out, "train_acc_all.csv"))
    pd.to_csv(df_summary_test_acc_all, os.path.join(root_out, "test_acc_all.csv"))
    pd.to_csv(df_summary_train_loss_all, os.path.join(root_out, "train_loss_all.csv"))
    pd.to_csv(df_summary_test_loss_all, os.path.join(root_out, "test_loss_all.csv"))

def run_training(cfg, path_out):
    # print start
    print("Starting training")

    # Dataset Inizialization
    # adj_matrix = compute_adjacency_matrix_delta(pain_labels=[3,-3,6,-6], base_labels=[0])
    adj_matrix = pd.read_csv(cfg["ADJ_PATH"], index_col=0).drop("AU28_c",axis=0).drop("AU28_c", axis=1)
    dataset = OpenFace_Features_Dataset(root_path=cfg["DATA_PATH"], use_labels=cfg["PAIN_LABELS"]+cfg["BASE_LABLES"], adj_matrix=adj_matrix)
    print("===========================")
    print("Dataset:")
    print(dataset.df.index.get_level_values("label").value_counts())

    # Model Initialization
    assert (dataset.num_classes == 2), "Not a binary classification problem!"
    in_channels = int(dataset.num_node_features)
    hidden_channels = [int(round(multiplier * in_channels)) for multiplier in cfg["HIDDEN_CHANNELS"]]
    module = importlib.import_module("XITE_GNN.training.BinaryClassification.models.{}".format(cfg["MODEL_TYPE"]))
    model = getattr(module, cfg["MODEL_TYPE"])(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=1, dropout=cfg["DROPOUT"], dropout_G=cfg["DROPOUT_G"])   
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])

    # print model
    print("===========================")
    print("Model overview:")
    print(model)

    # train and eval
    print("===========================")
    print("Starting {}-Fold Cross Validation...".format(cfg["NUM_FOLDS"]))
    history, best_params = train.kfold_CV(dataset, model, cfg["DEVICE"], loss_func, optimizer, k=cfg["NUM_FOLDS"], 
        num_epochs=cfg["NUM_EPOCHS"], batch_size=cfg["BATCH_SIZE"], scheduler=None)

    # summary:
    row_idx = ["test_acc_last", "test_acc_all", "train_acc_last", "train_acc_all",
               "test_loss_last", "test_loss_all", "train_loss_last", "train_loss_all"]
    df_summary = pd.DataFrame(0, index=row_idx, columns= ["val", "sd"])
    for k in range(cfg["NUM_FOLDS"]):
        df_summary.loc["test_acc_last","val"] += history[k]["test_acc"][-1] / cfg["NUM_FOLDS"]
        df_summary.loc["train_acc_last","val"] += history[k]["train_acc"][-1] / cfg["NUM_FOLDS"]
        df_summary.loc["test_acc_all","val"] += np.sum(history[k]["test_acc"]) / (cfg["NUM_FOLDS"] * cfg["NUM_EPOCHS"])
        df_summary.loc["train_acc_all","val"] += np.sum(history[k]["train_acc"]) / (cfg["NUM_FOLDS"] * cfg["NUM_EPOCHS"])
        df_summary.loc["test_loss_last","val"] += history[k]["test_loss"][-1] / cfg["NUM_FOLDS"]
        df_summary.loc["train_loss_last","val"] += history[k]["train_loss"][-1] / cfg["NUM_FOLDS"]
        df_summary.loc["test_loss_all","val"] += np.sum(history[k]["test_loss"]) / (cfg["NUM_FOLDS"] * cfg["NUM_EPOCHS"])
        df_summary.loc["train_loss_all","val"] += np.sum(history[k]["train_loss"]) / (cfg["NUM_FOLDS"] * cfg["NUM_EPOCHS"])

    df_summary.loc["test_acc_last","sd"] += np.std([history[k]["test_acc"][-1] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["train_acc_last","sd"] += np.std([history[k]["train_acc"][-1] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["test_loss_last","sd"] += np.std([history[k]["test_loss"][-1] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["train_loss_last","sd"] += np.std([history[k]["train_loss"][-1] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["test_acc_all","sd"] += np.std([history[k]["test_acc"] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["train_acc_all","sd"] += np.std([history[k]["train_acc"] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["test_loss_all","sd"] += np.std([history[k]["test_loss"] for k in range(cfg["NUM_FOLDS"])])
    df_summary.loc["train_loss_all","sd"] += np.std([history[k]["train_loss"] for k in range(cfg["NUM_FOLDS"])])
    print(df_summary)

    # store results
    print("===========================")
    print("Saving results...")
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    torch.save(best_params, os.path.join(path_out, "model_params.pth"))
    model.state_dict = best_params
    torch.save(model, os.path.join(path_out, "model.pth"))
    with open(os.path.join(path_out, "model_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    print_settings = {
        "PAIN_LABELS": cfg["PAIN_LABELS"], "BASE_LABELS": cfg["BASE_LABLES"], 
        "DATA_PATH": cfg["DATA_PATH"], "ADJ_PATH":cfg["ADJ_PATH"], 
        "NUM_FOLDS": cfg["NUM_FOLDS"], "NUM_EPOCHS": cfg["NUM_EPOCHS"],
        "BATCH_SIZE": cfg["BATCH_SIZE"], "LR": cfg["LR"], "WEIGHT_DECAY":cfg["WEIGHT_DECAY"], "DROPOUT":cfg["DROPOUT"], "DROPOUT_G":cfg["DROPOUT_G"],
        "OPTIMIZER": optimizer.__class__.__name__,
        "LOSS_FUNC": loss_func.__class__.__name__
    }
    with open(os.path.join(path_out, "train_settings.json"), "w") as f:
        json.dump(print_settings, f, sort_keys=False, indent=4)
    with open(os.path.join(path_out, "model_overview.txt"), "w") as f:
        f.write(str(model))
    df_summary.to_csv(os.path.join(path_out, "summary.csv"))
    # print finish
    print("Finished training")
    
    return df_summary


if __name__ == "__main__":
    main()

"""!
@}
"""