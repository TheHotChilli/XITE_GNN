"""!
@file
@ingroup Graph_Generation
@brief Functions for generating an AU dependecy graph from AU frequencies.
@addtogroup Graph_Generation
@{
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def compute_rel_AU_freqs_uncond(df_counts_and, nof_frames):
    """!
    Computes for each AU pair the relative co-occurance frequency. 
        P(AU_i, AU_j) = n_{AU_i \land AU_j} / n_{total frames with label}

    @param df_AU_abs_freq       Pandas Multiindex Dataframe with absolute AU Co-Occurence frequencies; shape = (labels x nof_AU, nof_AU)
    @param nof_frames           Pandas Series with information about total number of frames per label; shape = (nof_labels, 1)
    @retval df_AU_rel_freq      Pandas Multiindex Dataframe with relative AU Co-Occurence frequencies; shape = (labels x nof_AU, nof_AU)
    """

    # allocate memory for results
    AU_names = list(df_counts_and.columns)
    df_rel_freqs = pd.DataFrame(0, index=AU_names, columns=AU_names)

    # compute (unconditional) probabibility of Co-Occurrance
    for AUi in AU_names:
        for AUj in AU_names:
            df_rel_freqs.loc[AUi, AUj] = df_counts_and.loc[AUi,AUj] /  nof_frames

    return df_rel_freqs


def compute_rel_AU_freqs_cond(df_counts_and):
    """!
    Takes as input the absolute AU Co-Occurence frequencies stored in a Dataframe df_AU_freq and computes
    for all AU combinations the conditional probability for Co-Occurence:
        P(AU_i | AU_j) = n_{AU_i \land AU_j} / n_{AU_j}

    @param df_counts_and    Pandas Multiindex Dataframe with absolute AU Co-Occurence frequencies; shape = (nof_labels x nof_AU, nof_AU)
    @retval df_rel_freqs    Pandas Multiindex Dataframe with conditional relative AU Co-Occurence frequencies; shape = (labels x nof_AU, nof_AU)
    """
    # allocate memory for results
    AU_names = list(df_counts_and)
    df_rel_freqs = pd.DataFrame(0, index=AU_names, columns=AU_names)

    # Compute conditional probability of Co-Occurance of AU Combinations: 
    for AUi in AU_names:
        for AUj in AU_names:
            df_rel_freqs.loc[AUi, AUj] = df_counts_and.loc[AUi, AUj] / df_counts_and.loc[AUj, AUj]    

    # replace nan's; can have nan results if n_AUjAUj = 0
    df_rel_freqs.fillna(0)

    return df_rel_freqs


def compute_rel_AU_freq_cond_symm(df_counts_and, df_counts_or):
    """!
    Takes as input the absolute AU Co-Occurence frequencies stored in a Dataframe df_AU_freq and computes
    for all AU combinations the symmetrized conditional probability for Co-Occurence:
        P(AU_i = 1 \land AU_j = 1 | AU_i = 1 \lor AU_j = 1) = n_{AU_i \land AU_j} / n_{AU_i \lor AU_j}
    The probability can be computed from the conditional co-occurrence frequencies by symmetrizing the matrix. 

    @param df_AU_freq_rel_cond          Pandas Multiindex Dataframe with conditional relative AU Co-Occurence frequencies; shape = (labels x nof_AU, nof_AU)
    @retval df_AU_freq_rel_cond_symm    Pandas Multiindex Dataframe with symmetrized conditional relative AU Co-Occurence frequencies; shape = (labels x nof_AU, nof_AU)
    """

    # allocate memory for results
    AU_names = list(df_counts_and)
    df_rel_freqs = pd.DataFrame(0, index=AU_names, columns=AU_names)

    # Compute symmetrized conditional probability of Co-Occurance of AU Combinations: 
    for AUi in AU_names:
        for AUj in AU_names:
            df_rel_freqs.loc[AUi, AUj] = df_counts_and.loc[AUi, AUj] / df_counts_or.loc[AUi, AUj]    
    
    return df_rel_freqs


def compute_adjacency_matrix(use_labels=[0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6], use_AUs=None, input_dir=None, AU_method="AUc", computation_method="symm", eps=None):
    """!
    Computes the weighted graph adjacency matrix based on co-occurrence counts obtained from a frequency analysis. 

    @param use_labels   (list, optional): A list of integers representing the labels to be used. Default is [0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6].
    @param use_AUs      (list, optional): A list of strings representing the Action Units (AUs) to use. Default is None.
    @param input_dir    (str, optional): Path to the directory containing the input files. Default is None.
    @param AU_method    (str, optional): Method used to obtain the Action Units. One of 'AUc' or 'AUr'. Default is 'AUc'.
    @param computation_method   (str, optional): Method used to compute the weights of the adjacency matrix. One of 'symm', 'cond' or 'uncond'. Default is 'symm'.
    @param eps          (float, optional): Threshold for filtering the adjacency matrix. Only top eps entries will be kept, rest will be set to 0. Default is None.

    @retval df_adj      (DataFrame): A DataFrame representing the delta adjacency matrix.
    """
    # use_AUs    select which AUs to use
    # eps           threshold: only take top eps entries, rest=0

    labels_valid = [0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]
    assert(np.isin(use_labels, labels_valid).all())
    assert(AU_method in ["AUc", "AUr"]), "invalid AU_method provided. Valid is ['AUc', 'AUr]."
    if not input_dir: 
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "frequency_analysis")

    # load pair co-occurr counts and aggregate counts of use_labels
    df_counts_and = pd.read_csv(os.path.join(input_dir, f"counts_{AU_method}_and.csv"), index_col=[0,1])
    df_counts_and = df_counts_and[df_counts_and.index.get_level_values("label").isin(use_labels)]
    df_counts_and = df_counts_and.groupby("AU").sum()

    # allocate memory for results
    AUs = df_counts_and.columns
    df_adj = pd.DataFrame(0, index=AUs, columns=AUs)

    # compute weights as probabilities approximated by relative frequencies
    if computation_method == "symm":
        df_counts_or = pd.read_csv(os.path.join(input_dir, f"counts_{AU_method}_or.csv"), index_col=[0,1])
        df_counts_or = df_counts_or[df_counts_or.index.get_level_values("label").isin(use_labels)]
        df_counts_or = df_counts_or.groupby("AU").sum()
        df_adj = compute_rel_AU_freq_cond_symm(df_counts_and, df_counts_or)
    elif computation_method == "cond":
        df_adj = compute_rel_AU_freqs_cond(df_counts_and)
    elif computation_method == "uncond":
        nof_frames = pd.read_csv(os.path.join(input_dir, "nof_frames_per_label.csv"))
        nof_frames = nof_frames[nof_frames.index.isin(use_labels)].sum()
        df_adj = compute_rel_AU_freqs_uncond(df_counts_and, nof_frames)
    else:
        raise(ValueError("Invalid computation_method. Supported is [symm, cond, uncond]."))

    # filter for specified AUs/nodes
    if use_AUs:
        df_adj = df_adj.loc[use_AUs, use_AUs]

    # only keep top eps % entries, set other to 0
    if eps:
        assert (0 < eps < 1), "eps has to be in (0,1)"
        percentile = np.percentile(df_adj.values.flatten(), eps*100)
        df_adj[df_adj < percentile] = 0

    return df_adj

def compute_adjacency_matrix_delta(pain_labels=[3,-3, 6, -6], base_labels=[0], use_AUs=None, input_dir=None, 
                                   AU_method="AUc", computation_method="symm", eps=None):
    """!
    Compute the adjecency matrix as delta: adj_pain - adj_base given pain and base labels. 

    @param pain_labels  (list): A list of integers representing the pain labels. (default=[3,-3, 6, -6])
    @param base_labels  (list): A list of integers representing the base labels. (default=[0])
    @param use_AUs      (list): A list of integers representing the action units to use. (default=None)
    @param input_dir    (str): The input directory to use. (default=None)
    @param AU_method    (str): The method to use for the action units. (default="AUc")
    @param computation_method   (str): The method to use for the computation. (default="symm")
    @param eps          (float): A float value between 0 and 1 to use as the percentile value. (default=None)
        
    @retval df_adj      (DataFrame): A DataFrame representing the delta adjacency matrix.
    """
    if not input_dir: 
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "frequency_analysis")

    df_adj_pain = compute_adjacency_matrix(use_labels=pain_labels, use_AUs=use_AUs, input_dir=input_dir, 
        AU_method=AU_method, computation_method=computation_method)
    df_adj_base = compute_adjacency_matrix(use_labels=base_labels, use_AUs=use_AUs, input_dir=input_dir, 
        AU_method=AU_method, computation_method=computation_method)
    df_adj = (df_adj_pain - df_adj_base)
    df_adj[df_adj<0] = 0
    
    if eps:
        assert (0 < eps < 1), "eps has to be in (0,1)"
        percentile = np.percentile(df_adj.values.flatten(), eps*100)
        df_adj[df_adj < percentile] = 0

    return df_adj


def normalize_adj(df_adj):
    """!
    Normalizes the values of a given adjecency matrix (pd.DataFrame) to a range of [0, 1].

    @param df_adj   (DataFrame) Adjacency matrix to be normalized.

    @retval df_adj  (DataFrame) Normalized adjacency matrix. 
    """
    df_adj = (df_adj - df_adj.min().min()) / (df_adj.max().max() - df_adj.min().min())
    return df_adj


def main():
    """!
    Main function for computation of adjacency matrices. Computes different adjacency matrices based on the 
    frequency_analysis results. Stores the resulting adjacency matrices as pandas DataFrames in the folder
    "./results/adjacency_matrix". 
    """
    # Compute adjacency matrices
    df_adj_pain = compute_adjacency_matrix(use_labels=[3,-3,6,-6])
    df_adj_base = compute_adjacency_matrix(use_labels=[0])
    df_adj_delta = compute_adjacency_matrix_delta()
    df_adj_delta_normalized = normalize_adj(df_adj_delta)
    # write to disk
    dir_export = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "adjacency_matrix")
    if not os.path.exists(dir_export):
        os.makedirs(dir_export)
    df_adj_pain.to_csv(os.path.join(dir_export, "adjacency_matrix_pain.csv"))
    df_adj_base.to_csv(os.path.join(dir_export, "adjacency_matrix_base.csv"))
    df_adj_delta.to_csv(os.path.join(dir_export, "adjacency_matrix_delta.csv"))
    df_adj_delta_normalized.to_csv(os.path.join(dir_export, "adjacency_matrix_delta_normalized.csv"))


if __name__ == "__main__":
    main()

"""!
@}
"""