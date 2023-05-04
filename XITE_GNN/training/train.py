"""!
@file
@ingroup Training
@brief Main function for running a single training case using the OpenFace_Features_Dataset.
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
import pandas as pd
from pathlib import Path
from datetime import datetime
# from torch_geometric.nn import summary

sys.path.append(str(Path(__file__).parents[3]))
from XITE_GNN.datasets.OpenFace import OpenFace_Features_Dataset

def main():
    """!
    Trains a graph neural network (GNN) model using the OpenFace_Features_Dataset dataset
    with the specified training settings, and saves the trained model and results to disk.
    """
    # Training settings
    DATA_PATH = "/home/jbauske/00_Code/Python/Thesis/XITE_GNN/preprocessing/slice_and_feature_extraction/results/2023-05-03_19-22/video/features.csv"
    ADJ_PATH = "/home/jbauske/00_Code/Python/Thesis/XITE_GNN/preprocessing/graph_generation/results/adjacency_matrix/adjacency_matrix_delta_normalized.csv"
    PAIN_LABELS = [-3]
    BASE_LABLES = [100]
    MODEL_TYPE = "GCN"
    DEVICE = "cuda"
    NUM_FOLDS = 4
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    LR = 0.001
    WEIGHT_DECAY = 0
    DROPOUT = 0.5
    # args = parse_args()

    # print start
    print("===========================")
    print("Starting training...")

    # Dataset Inizialization
    # adj_matrix = compute_adjacency_matrix_delta(pain_labels=[3,-3,6,-6], base_labels=[0])
    adj_matrix = pd.read_csv(ADJ_PATH, index_col=0).drop("AU28_c",axis=0).drop("AU28_c", axis=1)
    dataset = OpenFace_Features_Dataset(root_path=DATA_PATH, use_labels=PAIN_LABELS+BASE_LABLES, adj_matrix=adj_matrix)
    print("===========================")
    print("Dataset:")
    print(dataset.df.index.get_level_values("label").value_counts())

    # Model Initialization
    assert (dataset.num_classes == 2), "Not a binary classification problem!"
    in_channels = int(dataset.num_node_features)
    # hidden1_channels = int(in_channels)
    # hidden2_channels = int(round(1/2*hidden1_channels))
    HIDDEN_CHANNELS = [in_channels, int(round(0.5*in_channels))] #int(round(0.5*in_channels))
    module = importlib.import_module(f"XITE_GNN.training.BinaryClassification.models.{MODEL_TYPE}")
    model = getattr(module, MODEL_TYPE)(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS, out_channels=1, dropout=DROPOUT, dropout_G=0.3)   
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # print model
    print("===========================")
    print("Model overview:")
    print(model)

    # train and eval
    print("===========================")
    print(f"Starting {NUM_FOLDS}-Fold Cross Validation...")
    history, best_params = train.kfold_CV(dataset, model, DEVICE, loss_func, optimizer, k=NUM_FOLDS, 
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, scheduler=None)

    # store results
    print("===========================")
    print("Saving results...")
    dt_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    path_out = os.path.dirname(os.path.abspath(__file__))
    path_out = os.path.join(path_out, "results", MODEL_TYPE, dt_now)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    torch.save(best_params, os.path.join(path_out, "model_params.pth"))
    model.state_dict = best_params
    torch.save(model, os.path.join(path_out, "model.pth"))
    with open(os.path.join(path_out, "model_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    train_settings = {
        "PAIN_LABELS": PAIN_LABELS, "BASE_LABELS": BASE_LABLES, 
        "DATA_PATH": DATA_PATH, "ADJ_PATH":ADJ_PATH, 
        "NUM_FOLDS": NUM_FOLDS, "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE, "LR": LR, "WEIGHT_DECAY":WEIGHT_DECAY, "DROPOUT":DROPOUT,
        "OPTIMIZER": optimizer.__class__.__name__,
        "LOSS_FUNC": loss_func.__class__.__name__
    }
    with open(os.path.join(path_out, "train_settings.json"), "w") as f:
        json.dump(train_settings, f, sort_keys=False, indent=4)
    with open(os.path.join(path_out, "model_overview.txt"), "w") as f:
        f.write(str(model))

    # print finish
    print("Finished training")


if __name__ == "__main__":
    main()

"""!
@}
"""