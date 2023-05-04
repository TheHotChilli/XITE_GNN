"""!
@file
@ingroup Training
@brief Functions for training and testing of PyTorch Models.
"""

"""!
@addtogroup Training
@{
"""

import numpy as np
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).parents[3]))
from XITE_GNN.datasets.OpenFace import OpenFace_Features_Dataset


def reset_params(model):
    """!
    Resets parameters of all layers.

    @param model    PyTorch model whose parameters need to be reset.
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_epoch(model, dataloader, loss_func, optimizer, device, num_classes):
    """!
    Trains the input PyTorch model for a single epoch on the input data and returns the training loss and the confusion matrix.

    @param model        PyTorch model to be trained.
    @param dataloader   PyTorch DataLoader object providing the data for training.
    @param loss_func    PyTorch loss function to be used during training.
    @param optimizer    PyTorch optimizer to be used during training.
    @param device       Device (CPU/GPU) on which to perform the computations.
    @param num_classes  Number of classes in the output of the model.
    @retval Tuple       A tuple containing the training loss (float) and the confusion matrix (torch.Tensor).
    """
    # set model to train mode (impacts behaviour of some layers like Dropout, BatchNorm etc)
    model.train()
    # initialize epoch loss and counter for correct samples
    train_loss = 0
    train_cf = 0
    # enumerate train-samples/batches
    for data in dataloader:
        # copy tensor/graph/data to CUDA device
        data = data.to(device)
        # cleaer gradients
        optimizer.zero_grad()
        # perform forward path
        y_hat = model(data)     # batch_size x dim_out
        # compute loss
        loss = loss_func(y_hat, data.y.float().unsqueeze(1))         
        # perform backward path -> gradients
        loss.backward()
        # update model parameters
        optimizer.step()
        # aggregate evaluation metrics
        train_loss += loss.item()
        train_cf += confusion_matrix(data.y.detach().cpu(), \
             y_hat.round().detach().cpu(), labels=list(range(num_classes)))
    return (train_loss / len(dataloader)), train_cf

def evaluate_train_epoch(model, dataloader, loss_func, device, num_classes):
    """!
    Evaluates (Test Epoch) the input PyTorch model for a single epoch on the input data and returns the validation loss and the confusion matrix.
    This function does not use eval mode.
    
    @param model        PyTorch model to be evaluated.
    @param dataloader   PyTorch DataLoader object providing the data for evaluation.
    @param loss_func    PyTorch loss function to be used during evaluation.
    @param device       Device (CPU/GPU) on which to perform the computations.
    @param num_classes  Number of classes in the output of the model.
    @return Tuple       A tuple containing the validation loss (float) and the confusion matrix (torch.Tensor).
    """
    # initialize epoch loss and counter for correct samples
    test_loss = 0
    test_cf = 0
    # deactivate autograd gradient computation
    with torch.no_grad():
        # enumerate test samples
        for data in dataloader:
            # copy tensor/graph/data to CUDA device
            data = data.to(device)
            # perform forward path
            y_hat = model(data)
            # compute loss
            loss = loss_func(y_hat, data.y.float().unsqueeze(1))
            # aggregate loss and nof correct
            test_loss += loss.item()
            test_cf += confusion_matrix(data.y.detach().cpu(), \
                y_hat.round().detach().cpu(), labels=list(range(num_classes)))
    return (test_loss / len(dataloader)), test_cf

def evaluate_epoch(model, dataloader, loss_func, device, num_classes):
    """!
    Evaluates (Test Epoch) the input PyTorch model for a single epoch on the input data and returns the validation loss and the confusion matrix.
    This function uses eval mode. 
    
    @param model        PyTorch model to be evaluated.
    @param dataloader   PyTorch DataLoader object providing the data for evaluation.
    @param loss_func    PyTorch loss function to be used during evaluation.
    @param device       Device (CPU/GPU) on which to perform the computations.
    @param num_classes  Number of classes in the output of the model.
    @return Tuple       A tuple containing the validation loss (float) and the confusion matrix (torch.Tensor).
    """
    # set model to eval mode (impacts behaviour of some layers like Dropout, BatchNorm etc)
    model.eval()
    # initialize epoch loss and counter for correct samples
    test_loss = 0
    test_cf = 0
    # deactivate autograd gradient computation
    with torch.no_grad():
        # enumerate test samples
        for data in dataloader:
            # copy tensor/graph/data to CUDA device
            data = data.to(device)
            # perform forward path
            y_hat = model(data)
            # compute loss
            loss = loss_func(y_hat, data.y.float().unsqueeze(1))
            # aggregate loss and nof correct
            test_loss += loss.item()
            test_cf += confusion_matrix(data.y.detach().cpu(), \
                y_hat.round().detach().cpu(), labels=list(range(num_classes)))
    return (test_loss / len(dataloader)), test_cf


def kfold_CV(dataset, model, device, loss_func, optimizer, k, num_epochs, batch_size, scheduler=None):
    """!
    Perform k-fold cross-validation on a dataset using a specified PyTorch model.
    
    @param dataset      A PyGGraphDataset containing the data to use for cross-validation.
    @param model        A PyTorch model to use for cross-validation.
    @param device       The device to use for training and evaluation (e.g. "cuda:0" for GPU or "cpu" for CPU).
    @param loss_func    The loss function to use for training the model.
    @param optimizer    The optimizer to use for training the model.
    @param k            The number of folds to use in cross-validation.
    @param num_epochs   The number of epochs to train the model for in each fold.
    @param batch_size   The batch size to use for training and evaluation.
    @param scheduler    (Optional) A PyTorch learning rate scheduler to use during training.
    @retval dict        A dictionary containing the evaluation metrics for each fold of cross-validation.
    """
    subjects = dataset.subject_list
    graph_data = dataset.to_torch_geometric()
    # container for metrics per fold
    history = {fold:{"train_loss":[], "train_acc":[], "train_cf":[], "train_subjects":[], 
                     "train_loss_drop": [], "train_acc_drop":[], "train_cf_drop":[],
                     "train_loss_run": [], "train_acc_run":[], "train_cf_run":[],
                     "test_loss":[], "test_acc":[], "test_cf":[], "test_subjects":[]} 
               for fold in range(k)}
    # copy model to CUDA device
    model.to(device)
    # init variables for storing best model
    best_params = None
    best_acc = 0.0
    # init kfold cross validation
    kf = KFold(n_splits=k, shuffle=False)
    # iterate k folds
    for fold, (train_idxs, test_idxs) in enumerate(kf.split(subjects)):
        # print fold
        print(f"FOLD {fold+1}/{k}")
        # reset model weights
        model.apply(reset_params)      
        # prepare train and test data for current fold
        train_subjcets = subjects[train_idxs]
        test_subjects = subjects[test_idxs]
        train_idxs, test_idxs = dataset.get_train_test_idxs(test_subjects)
        train_loader = DataLoader([graph_data[idx] for idx in train_idxs], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader([graph_data[idx] for idx in test_idxs], batch_size=batch_size)
        # df_train, df_test = dataset.split_train_test(test_subjects)
        # graph_data_train = dataset.to_torch_geometric(df_train, adj_matrix)
        # graph_data_test = dataset.to_torch_geometric(df_test, adj_matrix)
        # train_loader = DataLoader(graph_data_train, batch_size=batch_size)
        # test_loader = DataLoader(graph_data_test, batch_size=batch_size)
        history[fold]["train_subjects"] = train_subjcets
        history[fold]["test_subjects"] = test_subjects

        # initial evalution
        train_loss_drop, train_cf_drop = evaluate_train_epoch(model, train_loader, loss_func, device, dataset.num_classes)
        train_acc_drop = np.diagonal(train_cf_drop).sum() / len(train_idxs) * 100
        history[fold]["train_loss_drop"].append(train_loss_drop)
        history[fold]["train_acc_drop"]. append(train_acc_drop)
        history[fold]["train_cf_drop"].append(train_cf_drop)
        train_loss, train_cf = evaluate_epoch(model, train_loader, loss_func, device, dataset.num_classes)
        train_acc = np.diagonal(train_cf).sum() / len(train_idxs) * 100
        history[fold]["train_loss"].append(train_loss)
        history[fold]["train_acc"]. append(train_acc)
        history[fold]["train_cf"].append(train_cf)       
        test_loss, test_cf = evaluate_epoch(model, test_loader, loss_func, device, dataset.num_classes)
        test_acc = np.diagonal(test_cf).sum() / len(test_idxs) * 100
        history[fold]["test_loss"].append(test_loss)
        history[fold]["test_acc"]. append(test_acc)
        history[fold]["test_cf"].append(test_cf)  
        print(f"Test Loss:{test_loss:.3f} | Test Acc:{test_acc:.2f}%")
        
        # iterate epochs
        for epoch in range(num_epochs):  
            # train model
            train_loss_run, train_cf_run = train_epoch(model, train_loader, loss_func, optimizer, device, dataset.num_classes)
            # evaluate model
            test_loss, test_cf = evaluate_epoch(model, test_loader, loss_func, device, dataset.num_classes)
            train_loss, train_cf = evaluate_epoch(model, train_loader, loss_func, device, dataset.num_classes)
            train_loss_drop, train_cf_drop = evaluate_train_epoch(model, train_loader, loss_func, device, dataset.num_classes)
            train_acc = np.diagonal(train_cf).sum() / len(train_idxs) * 100
            train_acc_run = np.diagonal(train_cf_run).sum() / len(train_idxs) * 100
            train_acc_drop = np.diagonal(train_cf_drop).sum() / len(train_idxs) * 100
            test_acc = np.diagonal(test_cf).sum() / len(test_idxs) * 100
            # adjust learning rate via scheduler
            if scheduler:
                scheduler.step()
            # fetch best model params
            if test_acc > best_acc:
                best_acc = test_acc
                best_params = model.state_dict()
            # print epoch
            print(f"EPOCH {epoch+1:2}/{num_epochs}: Train Loss:{train_loss:.3f} | " +
                f"Test Loss:{test_loss:.3f} | Train Acc:{train_acc:.2f}% | Test Acc:{test_acc:.2f}%")
            # keep track of evaluation metrics of last epoch in fold
            history[fold]["train_loss_run"].append(train_loss_run)
            history[fold]["train_acc_run"].append(train_acc_run)
            history[fold]["train_cf_run"].append(train_cf_run)
            history[fold]["train_loss"].append(train_loss)
            history[fold]["train_acc"].append(train_acc)
            history[fold]["train_cf"].append(train_cf)
            history[fold]["train_loss_drop"].append(train_loss_drop)
            history[fold]["train_acc_drop"]. append(train_acc_drop)
            history[fold]["train_cf_drop"].append(train_cf_drop)
            history[fold]["test_loss"].append(test_loss)
            history[fold]["test_acc"].append(test_acc)
            history[fold]["test_cf"].append(test_cf)

    # print summary 
    avg_train_loss = np.mean([history[fold]["train_loss"][-1] for fold in range(k)])
    avg_train_acc = np.mean([history[fold]["train_acc"][-1] for fold in range(k)])
    avg_test_loss = np.mean([history[fold]["test_loss"][-1] for fold in range(k)])
    avg_test_acc = np.mean([history[fold]["test_acc"][-1] for fold in range(k)])
    print(f"SUMMARY {k} FOLDS")
    print(f"Avg Train Loss: {avg_train_loss:.3f} | Avg Test Loss: {avg_test_loss:.3f} | " + 
        f"Avg Train Acc: {avg_train_acc:.2f}% | Avg Test Acc: {avg_test_acc:.2f}%")
    # # store history and best model
    # print("Saving results...")
    # model_name = model.__class__.__name__
    # dt_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    # path_out = os.path.join(config.dir_export, "training", model_name, dt_now)
    # torch.save(best_params, os.path.join(path_out, "model_weights.pth"))
    # with open(os.path.join(path_out, "model_history.pkl"), "wb") as f:
    #     pickle.dump(history, f)
    # # print finish
    # print("Finished training...")

    return history, best_params

"""!
@}
"""