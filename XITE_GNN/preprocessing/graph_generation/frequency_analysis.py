"""!
@file
@ingroup Graph_Generation
@brief Functions for counting AU occurrences based on OpenFace Classification results. 
@addtogroup Graph_Generation
@{
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed

from config import subjects_no_use
sys.path.append(str(Path(__file__).parents[3]))
from XITE_GNN.datasets.OpenFace import OpenFace_Raw_Dataset

def count_pair_occurrences(df_AU_data, labels, eps_activity=1.0, comparator=np.logical_and):
    """!
    Computes the absolute co-occurence frequencies for each AU pairs, given the AU-Data of one subject.
    The co-occurence frequencies is defined as:
        n_{AU_i \and AU_j} = \# frames where AU_i and AU_j co-occure 
    
    @param df_AU_features   Pandas Dataframe containing label information and AU classification or reggression value per frame; shape = (nof_frames, nof_AU + 1); 
    @param eps_activity     only values >= eps_activity will be counted; set to 1 if using classification results; set in [0,5] if using regression results
    @retval df_freq_abs     Pandas Multiindex Dataframe for storing the results; shape = (labels x nof_AU, nof_AU x 2)
    """

    assert(0 <= eps_activity <= 5)

    labels_valid = [0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]
    AU_names = list(df_AU_data.columns)

    rowidx = pd.MultiIndex.from_product([labels_valid, AU_names], names=["label", "AU"])
    colidx = pd.Index(AU_names, name="AU")
    df_counts = pd.DataFrame(0, index=rowidx, columns=colidx)

    # count occurences
    for label in labels_valid:
        for AUi in AU_names:
            for AUj in AU_names:
                df_counts.loc[(label, AUi), AUj] = df_AU_data[(labels == label) \
                    & comparator((df_AU_data[AUi] >= eps_activity), (df_AU_data[AUj] >= eps_activity)).values] \
                    .iloc[:,0].count()
    return df_counts 


def _subject_frequency_analysis(subject_id, eps_conf=None):
    """!
    Wrapper function. Performs the frequency analysis on a single subject with provided subject_id. 

    @param subject_id   (int) ID of the subject to be analyzed.
    @param eps_conf     (float, optional) Confidence threshold for filtering out inconfident data.
    @retval df_list     list of pandas DataFrames and a pandas Series. Four DataFrames with the count of AUs appearing 
                        with each other under different logical operations and a Series with the total number of frames 
                        per label.
    """
    # init
    labels_valid = [0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]
    dataset = OpenFace_Raw_Dataset()
    labels = dataset.load_labels(subject_id)
    AUc_data = dataset.load_data(subject_id, target_channels="AUc")
    AUr_data = dataset.load_data(subject_id, target_channels="AUr")
    nof_frames_per_label = pd.Series(0, index=labels_valid)
    # optional: filter out inconfident data
    if eps_conf:
        assert(0.0 < eps_conf <= 1.0)
        confidence = dataset.load_data(subject_id, target_channels="confidence")
        AUc_data = AUc_data[confidence>eps_conf]
        AUr_data = AUr_data[confidence>eps_conf]
        labels = labels[labels>eps_conf]
    # count occurences
    df_counts_AUc_and = count_pair_occurrences(AUc_data, labels, comparator=np.logical_and)
    df_counts_AUc_or = count_pair_occurrences(AUc_data, labels, comparator=np.logical_or)
    df_counts_AUr_and = count_pair_occurrences(AUr_data, labels, comparator=np.logical_and)
    df_counts_AUr_or = count_pair_occurrences(AUr_data, labels, comparator=np.logical_or)
    # count total nof frames per label
    for label in labels_valid:
        nof_frames_per_label[label] += len(labels[labels == label])
    print(f"Frequency analysis finished for S{subject_id}")

    return [df_counts_AUc_and, df_counts_AUc_or, df_counts_AUr_and, df_counts_AUr_or, nof_frames_per_label]

def frequency_analysis_parallel(eps_conf=None):
    """!
    Parallel frequency analysis of all valid subjects. 

    @param eps_conf     (float, optional) Confidence threshold for filtering out inconfident data.
    """
    print("Starting AU frequency analysis")

    # create dataset object for OpenFace data
    dataset = OpenFace_Raw_Dataset()

    # allocate memory for absolute frequencies
    labels_valid = [0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]
    row_idx = pd.MultiIndex.from_product([labels_valid, dataset.channels_AUc], names=["label", "AU"])
    col_idx = pd.Index(dataset.channels_AUc, name="AU")
    df_counts_AUc_and = pd.DataFrame(0, index=row_idx, columns=col_idx)
    df_counts_AUc_or = pd.DataFrame(0, index=row_idx, columns=col_idx)
    row_idx = pd.MultiIndex.from_product([labels_valid, dataset.channels_AUr], names=["label", "AU"])
    col_idx = pd.Index(dataset.channels_AUr, name="AU")
    df_counts_AUr_and = pd.DataFrame(0, index=row_idx, columns=col_idx)
    df_counts_AUr_or = pd.DataFrame(0, index=row_idx, columns=col_idx)
    nof_frames_per_label = pd.Series(0, index=labels_valid)

    # parallel processing of all subjects
    results = Parallel(n_jobs=30)(delayed(_subject_frequency_analysis)(subj_id) for subj_id in dataset.subject_list \
        if subj_id not in subjects_no_use)

    # join parallel results
    for result in results:
        df_counts_AUc_and += result[0]
        df_counts_AUc_or += result[1]
        df_counts_AUr_and += result[2]
        df_counts_AUr_or += result[3]
        nof_frames_per_label += result[4]

    # write the results
    print("Writing frequency analysis results")
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "frequency_analysis")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    df_counts_AUc_and.to_csv(os.path.join(export_dir, "counts_AUc_and.csv"))
    df_counts_AUc_or.to_csv(os.path.join(export_dir, "counts_AUc_or.csv"))
    df_counts_AUr_and.to_csv(os.path.join(export_dir, "counts_AUr_and.csv"))
    df_counts_AUr_or.to_csv(os.path.join(export_dir, "counts_AUr_or.csv"))
    nof_frames_per_label.to_csv(os.path.join(export_dir, "nof_frames_per_label.csv"))


def frequency_analysis(eps_conf=None):
    """!
    Sequentiell frequency analysis of all valid subjects. 

    @param eps_conf     (float, optional) Confidence threshold for filtering out inconfident data.
    """

    print("Starting AU frequency analysis")

    # create dataset object for OpenFace data
    dataset = OpenFace_Raw_Dataset()

    # allocate memory for absolute frequencies
    labels_valid = [0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]
    row_idx = pd.MultiIndex.from_product([labels_valid, dataset.channels_AUc], names=["label", "AU"])
    col_idx = pd.Index(dataset.channels_AUc, name="AU")
    df_counts_AUc_and = pd.DataFrame(0, index=row_idx, columns=col_idx)
    df_counts_AUc_or = pd.DataFrame(0, index=row_idx, columns=col_idx)
    row_idx = pd.MultiIndex.from_product([labels_valid, dataset.channels_AUr], names=["label", "AU"])
    col_idx = pd.Index(dataset.channels_AUr, name="AU")
    df_counts_AUr_and = pd.DataFrame(0, index=row_idx, columns=col_idx)
    df_counts_AUr_or = pd.DataFrame(0, index=row_idx, columns=col_idx)
    nof_frames_per_label = pd.Series(0, index=labels_valid)

    # loop all subjects 
    for subject_id in dataset.subject_list:
        if subject_id in subjects_no_use: continue
        # load data and labels of subject
        labels = dataset.load_labels(subject_id)
        AUc_data = dataset.load_data(subject_id, target_channels="AUc")
        AUr_data = dataset.load_data(subject_id, target_channels="AUr")
        # optional: filter out inconfident data
        if eps_conf:
            assert(0.0 < eps_conf <= 1.0)
            confidence = dataset.load_data(subject_id, target_channels="confidence")
            AUc_data = AUc_data[confidence>eps_conf]
            AUr_data = AUr_data[confidence>eps_conf]
            labels = labels[labels>eps_conf]
        # count occurences
        df_counts_AUc_and += count_pair_occurrences(AUc_data, labels, comparator=np.logical_and)
        df_counts_AUc_or += count_pair_occurrences(AUc_data, labels, comparator=np.logical_or)
        df_counts_AUr_and += count_pair_occurrences(AUr_data, labels, comparator=np.logical_and)
        df_counts_AUr_or += count_pair_occurrences(AUr_data, labels, comparator=np.logical_or)
        # count total nof frames per label
        for label in labels_valid:
            nof_frames_per_label[label] += len(labels[labels == label])
        print(f"Frequency analysis finished for S{subject_id}")
        

    # write the results
    print("Writing frequency analysis results")
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "frequency_analysis")
    if not os.path.exist(export_dir):
        os.makedirs(export_dir)
    df_counts_AUc_and.to_csv(os.path.join(export_dir, "counts_AUc_and.csv"))
    df_counts_AUc_or.to_csv(os.path.join(export_dir, "counts_AUc_or.csv"))
    df_counts_AUr_and.to_csv(os.path.join(export_dir, "counts_AUr_and.csv"))
    df_counts_AUr_or.to_csv(os.path.join(export_dir, "counts_AUr_or.csv"))
    nof_frames_per_label.to_csv(os.path.join(export_dir, "nof_frames_per_label.csv"))

    # print finish
    print("Finished with frequency analysis")


def main():
    """!
    Main function that performs (parallel) frequency analysis and stores the results to csv files.
    """
    # frequency_analysis()
    frequency_analysis_parallel() 


if __name__ == "__main__":
    main()



"""!
@}
"""