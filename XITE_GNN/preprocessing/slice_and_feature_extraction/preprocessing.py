#!/usr/bin/env python3
"""!
@file
@brief Classes and main function for slicing/segmentation and feature extraction of X-ITE pain database.
@ingroup Slicing_and_Feature_Extraction
@addtogroup Slicing_and_Feature_Extraction
@{
"""

import numpy as np
import pandas as pd
import os
import warnings
from scipy.io import loadmat
from itertools import product
from joblib import Parallel, delayed    
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))
from XITE_GNN.datasets.XITE import XITE
from XITE_GNN.datasets.OpenFace import OpenFace_Raw_Dataset
from XITE_GNN.datasets.Bio import Bio_raw_dataset

import feature_extraction_functions
from filtering_functions import butter_filter

# # Disable warnings in feature_extraction_functions. 
# # Warnings occure because there are samples where 0 division appears. 
# # Corrupt samples and their features are handled in a repair step. 
# warnings.filterwarnings("ignore", module="feature_extraction_functions")
# warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
# warnings.filterwarnings("error", category=RuntimeWarning)
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     import feature_extraction_functions





class XITE_Preprocessor(XITE):
    """!
    A class to segment the X-ITE data into slices/segments and extract features from each slice/segment.
    """

    # dir_export = os.path.join(XITE_config.dir_results, "slices_and_feature_extraction", 
    #     datetime.now().strftime('%Y-%m-%d_%H-%M'))
    
    _export_dt = datetime.now().strftime('%Y-%m-%d_%H-%M')

    def __init__(self, data_type="video"):
        """!
        Constructor of XITE_Preprocessor class.

        @param data_type    (str) The X-ITE data modality ["video", "bio", or "audio"] to be processed. Default is "video".
        """
        self._supported_data_types = ["video", "bio"]
        assert data_type in self._supported_data_types
        self.data_type = data_type
        self._read_config()

    def _read_config(self):
        """!
        Reads and processes the config file.
        """
        # import correct config to corresponding data_type
        config = __import__("config_"+self.data_type)

        if self.data_type == "video":
            self.dataset = OpenFace_Raw_Dataset(config.dir_data, config.dir_labels)
        elif self.data_type == "bio":
            self.dataset = Bio_raw_dataset(config.dir_data)
        else:
            raise ValueError(f"Data_type {self.data_type} not supported")
        self.n_processes = config.nof_processes
        self.dir_export = config.dir_export

        self.slice_extraction = config.slice_extraction
        self.feature_extraction = config.feature_extraction

        self.rescale_slices = config.rescale_slices
        self.rescale_features = config.rescale_features

        self.subjects_no_use = config.subjects_no_use
        self.subjects_valid = [subj_id for subj_id in self.dataset.subject_list \
                               if subj_id not in self.subjects_no_use]
        
        self.filter_settings = config.filter_settings
        self.channels_to_filter = list(self.filter_settings)

        self.feature_extraction_settings = config.feature_extraction_settings
        self.feature_extraction_settings_d1 = config.feature_extraction_settings_1st_derivative
        self.feature_extraction_settings_d2 = config.feature_extraction_settings_2nd_derivative
        self.channels_feature_extraction = list(config.feature_extraction_settings)
        self.channels_feature_extraction_d1 = list(config.feature_extraction_settings_1st_derivative)
        self.channels_feature_extraction_d2 = list(config.feature_extraction_settings_2nd_derivative)
        self.nof_features = sum([len(val) for val in self.feature_extraction_settings.values()] + \
                                [len(val) for val in self.feature_extraction_settings_d1.values()] + \
                                [len(val) for val in self.feature_extraction_settings_d2.values()])

        self.channels_to_process = list(set(config.slice_channels + self.channels_feature_extraction +\
                                            self.channels_feature_extraction_d1 + self.channels_feature_extraction_d2))
        
        self.slice_shifts = config.slice_shifts
        self.slice_lengths = config.slice_lengths
        self.interval_min_lengths = config.interval_min_lengths
        self.pre_interval_min_lengths = config.pre_interval_min_lengths
        self.post_interval_min_lengths = config.post_interval_min_lengths


    def _get_label_group(self,label):
        """!
        Returns the string pain group to which a int given label belongs. E.g. label 1 belongs to group "pH".
        
        @param label    (int) label.
        @return         (str) pain group to which the label belongs.
        """
        if label in [1,2,3]: return "pH"
        elif label in [-1,-2,-3]: return "pE"
        elif label in [4,5,6]: return "tH"
        elif label in [-4,-5,-6]: return "tE"
        elif label in [100,200,300]: return "BpH"
        elif label in [-100,-200,-300]: return "BpE"
        elif label in [400,500,600]: return "BtH"
        elif label in [-400,-500,-600]: return "BtE"
        else: raise ValueError(f"Invalid label '{label}'.")


    def preprocess_parallel(self):
        """!
        Perform parallel preprocessing of the XITE data.
        """
        print(f"Starting slicing and feature extraction of XITE {self.data_type} data")
        with Parallel(n_jobs=self.n_processes, backend="loky") as parallel: 
            if self.channels_to_process:
                print("Preprocessing video data of subjects:")
                print(f"{self.subjects_valid}")
                results = parallel(delayed(self._call_subject_preprocessing)(subject_id) for subject_id in self.subjects_valid)
                print("Writing results...")
                self._write_parallel_results(results)
                print("Finished")
            else:
                warnings.warn("No channels provided.")

    # Wrapper functions for parallel processing
    def _call_subject_preprocessing(self, subject_id):
        """!
        Wrapper function for the parallel processing. Calls the `preprocess` method of the Subject_Preprocessor class for a given subject.

        @param subject_id   (int) The ID of the subject.
        """
        print(f"Preprocessing S{subject_id}")
        subj = Subject_Preprocessor(subject_id)
        return subj.preprocess()

    # Variante: Ein df je subject
    def _write_parallel_results(self, results):
        """!
        Write the results of the parallel preprocessing to CSV files.

        Parameters:
            @param results      (list) The list of results returned by the `_call_subject_preprocessing` method.
            @param data_type    (str) The type of data ["video", "bio", or "audio"). Default is "video".
        """
        # dir_export = os.path.join(self.dir_export, datetime.now().strftime('%Y-%m-%d_%H-%M'))
        # dir_out = os.path.join(dir_export, self.data_type)
        dir_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", \
                                  datetime.now().strftime('%Y-%m-%d_%H-%M'), self.data_type)
        if not os.path.exists(dir_out): os.makedirs(dir_out)
        append_header_features = True
        append_header_slices = True
        for s in range(len(results)):
            features = results[s][0]
            slices = results[s][1]
            # write features
            if features is not None:
                if append_header_features:
                    features.to_csv(os.path.join(dir_out, "features.csv"))
                    append_header_features = False
                else: 
                    features.to_csv(os.path.join(dir_out, "features.csv"), mode="a", header=False)
            # write slices
            if slices is not None:
                if append_header_slices:
                    slices.to_csv(os.path.join(dir_out, "slices.csv"))
                    append_header_slices = False
                else:
                    slices.to_csv(os.path.join(dir_out, "slices.csv"), mode="a", header=False)


class Subject_Preprocessor(XITE_Preprocessor):
    """!
    A class for pre-processing an individual subject within the X-ITE pain database.
    """

    def __init__(self, subject_id):
        """!
        Constructor of Subject_Preprocessor class.

        @param subject_id   (int) ID of the subject to preprocess.
        """
        super().__init__()
        self.subject_id = subject_id


    def preprocess(self):
        """!
        Method that pre-processes the subjects X-ITE data and returns the computed features and slices data.

        @retval features        
        @retval slices_data     
        """
        # load data
        sample_rate = self.sample_rate[self.data_type]
        data = self.dataset.load_data(self.subject_id, self.channels_to_process)
        labels = self.dataset.load_labels(self.subject_id)

        # extract slices
        slices_data, slices_labels = self._extract_slices(data, labels, sample_rate)

        ## Compute features
        slices_data, features = self._process_slices(slices_data, slices_labels, sample_rate)

        return features, slices_data


    def _extract_slices(self, data, labels, f_sampling):
        """
        Extracts pain and baseline slices from data according to settings as specified via config. Returns
        a list of dataframes, where each dataframe represents a slice. 
        """
        # get start info of slices (as list of lists, where each inner list contains all start idxs of one label)
        slices_start_idxs, slices_labels = self._compute_slices_info(labels, f_sampling)
        # loop all slices
        all_slices = []
        for s,start_idx in enumerate(slices_start_idxs):
            label = slices_labels[s]
            pain_group = self._get_label_group(label)
            slice_len = self.slice_lengths[pain_group]*f_sampling
            end_idx = start_idx + slice_len
            slice_data = data.iloc[start_idx:end_idx+1,:]
            slice_data.index = pd.MultiIndex.from_product(
                [[self.subject_id], [s], [label], slice_data.index], 
                names=["subj_id", "slice_id", "label", "glob_idx"]
            )
            all_slices.append(slice_data)

        return all_slices, slices_labels


    def _compute_slices_info(self, labels, f_sampling):
        """
        Computes information about where each slice starts.

        @param[in]  labels              np.array; label info for each frame
        @param[in]  f_sampling          int; sampling frequency
        @param[out] slices_start_idxs   list; list of arrays, each array contains start idxs for one label
        @param[out] slices_labels       list; 
        """

        # interval information
        start_idxs, end_idxs, interval_labels, durations = self._compute_intervals_info(labels)   # get info about all intervals
        pre_interval_labels = np.concatenate(([np.nan], interval_labels[0:-1]))
        pre_durations = np.concatenate(([np.nan], durations[0:-1]))
        post_interval_labels = np.concatenate((interval_labels[1::], [np.nan]))
        post_durations = np.concatenate((durations[1::], [np.nan]))

        # containers for accumulation
        slices_start_idxs = []      #list of arrays, each array represents the global slice start idxs for one label
        slices_labels = []          #list of labels, each entry maps array in start_idxs to a label

        # check which intervals 
        for label in self.labels_pain:
            pain_group = self._get_label_group(label)
            # find pain intervals that match (check if label matches, pre and post interval are baseline, and if durations ok)
            is_pain_intervals = (
                (interval_labels == label)                                              # interval has pain label?                                
                & (np.isin(pre_interval_labels, self.labels_base + [0]))                # pre interval has baseline label?
                & (np.isin(post_interval_labels, self.labels_base + [0]))               # post interval has baseline label?
                & (pre_durations >= self.pre_interval_min_lengths[pain_group])          # pre (base)interval is long enough
                & (durations >= self.interval_min_lengths[pain_group]*f_sampling-2)     # interval is long enough
                # & (post_durations >= self.slice_shifts["B"+pain_group]*f_sampling       # post (base)interval is long enough for shifted baseline slice
                    # + self.slice_lengths["B"+pain_group]*f_sampling)                    # B_duration >= B_shift + B_len
            )
            pain_interval_ids = is_pain_intervals.nonzero()[0]
            # assure pain slice is not interlapping with next stimuli interval
            nof_interlapping_frames = self.slice_shifts[pain_group]*f_sampling +\
                + self.slice_lengths[pain_group]*f_sampling - durations[pain_interval_ids]  # interlapping = shift + slice_len - duration  
            pain_interval_ids = pain_interval_ids[nof_interlapping_frames <= post_durations[pain_interval_ids]]

            #base_interval_ids = pain_interval_ids + 1
            is_base_intervals = (interval_labels == label*100) & (pre_interval_labels == label) 
            base_interval_ids = is_base_intervals.nonzero()[0]
            # check that baseline interval is long enough for shifted baseline slice
            base_interval_ids = base_interval_ids[durations[base_interval_ids] >= \
                self.slice_shifts["B"+pain_group]*f_sampling + self.slice_lengths["B"+pain_group]*f_sampling]         # B_duration >= B_shift + B_len
            
            # Compute actual slice info
            pain_slice_start_idxs = start_idxs[pain_interval_ids] + self.slice_shifts[pain_group]*f_sampling
            base_slice_start_idxs = start_idxs[base_interval_ids] + self.slice_shifts["B"+pain_group]*f_sampling
            # Append info arrays of pain slices and base slices to list
            slices_start_idxs += list(pain_slice_start_idxs)
            slices_labels += [label for i in range(len(pain_slice_start_idxs))]
            slices_start_idxs += list(base_slice_start_idxs)
            slices_labels += [label*100 for i in range(len(base_slice_start_idxs))]

        return slices_start_idxs, slices_labels


    def _compute_intervals_info(self, labels):
        """!
        Computes the information about all intervals (pain, baseline and invalid labels). Returns for each interval
        information about start_idx, end_idx, interval_label and duration.
        """
        is_interval_start = np.concatenate(([True], labels[:-1] != labels[1:]), axis=0)
        start_idxs = is_interval_start.nonzero()[0]
        end_idxs = np.roll(is_interval_start, -1).nonzero()[0]
        durations = end_idxs - start_idxs + 1
        interval_labels = labels[start_idxs]
        # set label of baseline intervals that follow a pain interval to 100*label
        pre_interval_labels = np.concatenate(([np.nan], interval_labels[0:-1]))
        is_baseline_interval = (interval_labels == 0 &
                                np.isin(pre_interval_labels, self.labels_pain))
        interval_labels[is_baseline_interval] = pre_interval_labels[is_baseline_interval]*100
        return start_idxs, end_idxs, interval_labels, durations


    def _process_slices(self, slices_data, slices_labels, sample_rate):
        """!
        Processes the slices by applying a Butterworth filter, extracting features, and returning the features in a dataframe.

        @param slices_data      (list of pd.DataFrames) List of slices. Each entry is one slice/segment represented as a Dataframe.
        @param slices_labels    (list or numpy array) List of integer labels corresponding to the slices in slices_data.
        @param sample_rate      (int or float) the sampling rate of the data.

        @retval slices_data     (pd.DataFrame) Dataframe with raw data of all slices. Columns represent channels (e.g. col 1 = AU01_r). 
                                Each row represents one frame of a slice. Row idx is a multiindex with entries [subj_id, slice_id, label, glob_idx].
        @retval features        (pd.DataFrame) Dataframe with feature data of all slices. Columns represent channels and their features. 
                                Channel index is a multiindex where index-level 0 represents the channel and index-level 1 represents the corresponding features.
                                Each row represents one slice. Row index is a mutliindex with entries [subj_id slice_id label glob_idx_start glob_idx_end].
        """
        nof_slices = len(slices_data)

        # filter slices and extract features
        features = np.ndarray(shape=(nof_slices, self.nof_features))
        glob_start_idxs = np.ndarray(nof_slices)
        glob_end_idxs = np.ndarray(nof_slices)
        for s in range(nof_slices):
            # Filter slice (Butterworth)
            slices_data[s] = self._apply_filter(slices_data[s].copy(), sample_rate)
            # Extract features from slice
            features[s,:] = self._extract_features_from_slice(slices_data[s], sample_rate)
            # store global start and end idx of slice as slice info
            glob_start_idxs[s] = slices_data[s].index.get_level_values("glob_idx")[0]
            glob_end_idxs[s] = slices_data[s].index.get_level_values("glob_idx")[-1]
        
        # convert features from array to dataframe 
        row_idx = pd.MultiIndex.from_tuples(
            [(self.subject_id, s, slices_labels[s], glob_start_idxs[s], glob_end_idxs[s]) for s in range(nof_slices)],
            names = ["subj_id", "slice_id", "label", "glob_idx_start", "glob_idx_end"]
        )
        col_idx = []  
        for channel in slices_data[0].columns:
            if channel in self.channels_feature_extraction:
                col_idx += product([channel], self.feature_extraction_settings[channel])
            if channel in self.channels_feature_extraction_d1:
                col_idx += product([channel], [elem+"_d1" for elem in self.feature_extraction_settings_d1[channel]])
            if channel in self.channels_feature_extraction_d2:
                col_idx += product([channel], [elem+"_d2" for elem in self.feature_extraction_settings_d2[channel]])
        col_idx = pd.MultiIndex.from_tuples(col_idx, names=["channel", "feature"])
        features = pd.DataFrame(features, index=row_idx, columns=col_idx)

        # repair corrupt features
        features = self._repair_features(features, strategy="mean")

        # rescale features
        if self.rescale_features:
            # features = self.standardize_pre_subject(features)
            scaler = StandardScaler()
            features.iloc[:] = scaler.fit_transform(features)

        # rescale slices
        # TODO 

        # join list of slices into one large dataframe
        if len(slices_data) > 1:
            slices_data = pd.concat(slices_data)

        if self.slice_extraction:
            return slices_data, features
        else:
            return None, features


    def _extract_features_from_slice(self, data, sample_rate):
        """!
        This function takes the DataFrame of a single slice and the sample_rate at which the data was 
        recorded and extractes several features for each data channel (data column). Feature extraction
        methods are specified via config file. Returns a list of feature values. 

        @param data         (pd.DataFrame) DataFrame representing the data of a single slice. 
        @param sample_rate  (int or float) Sample rate at which data was recorded. 

        @retval feature     (list) List of features. Each entry is one feature. 
        """
        # init empty list for aggregation of all slice features
        features = []
        # loop each channel data channel --> extract features for each channel seperate
        for channel in data.columns:
            # apply feature extraction on signal
            if channel in self.channels_feature_extraction:
                for func_name in self.feature_extraction_settings[channel]:
                    func = getattr(feature_extraction_functions, func_name)
                    try: 
                        features.append(func(data[channel].to_numpy(), sample_rate=sample_rate))    #try calling func with opt param
                    except:
                        features.append(func(data[channel].to_numpy()))                             # call func without opt param
            # apply feature extraction on 1st derivative
            if channel in self.channels_feature_extraction_d1:
                derivative = feature_extraction_functions.compute_derivative(data[channel].to_numpy())
                for func_name in self.feature_extraction_settings_d1[channel]:
                    func = getattr(feature_extraction_functions, func_name)
                    try: 
                        features.append(func(derivative, sampling_rate=sample_rate))
                    except: 
                        features.append(func(derivative))
            # apply feature extraction on 2nd derivative
            if channel in self.channels_feature_extraction_d2:
                derivative = feature_extraction_functions.compute_derivative(\
                    feature_extraction_functions.compute_derivative(data[channel].to_numpy()))
                for func_name in self.feature_extraction_settings_d2[channel]:
                    func = getattr(feature_extraction_functions, func_name)
                    try:
                        features.append(func(derivative, sample_rate=sample_rate))
                    except:
                        features.append(func(derivative))
        return features


    def _repair_features(self,features, strategy="mean"):
        """!
        Replaces or removes NaN, inf, or -inf entries in the input features data. 

        @param features     (pd.DataFrame) Feature dataset of all samples/slices. 
        @param strategy     (str or float) The strategy to repair corrupt entries in features. Valid options are "mean", "delete",
                            or a float/integer that indicates a replacement value.
        """
    
        features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Delete all rows where an corrupt entrie appears 
        if strategy == "delete":
            return features.dropna(axis=0, how="any", inplace=True)

        # Try to replace corrupt entries with the mean value of the column/feature
        # of all non corrupt entries with matching label
        elif strategy == "mean":
            for label in np.unique(features.index.get_level_values("label")):
                data_label = features.loc[(slice(None),slice(None),label,slice(None),slice(None)),:]
                features.loc[(slice(None),slice(None),label,slice(None),slice(None)),:] = data_label.fillna(value=data_label.mean(), axis=0)
            features.dropna(axis=0, how="any", inplace=True)
            return features
        
        elif type(strategy) == int or type(strategy) == float:
            features.fillna(value=strategy, axis=0, inplace=True)
            return features

        else:
            raise(ValueError(f"Invalid strategy '{strategy}'! Supported strategies are mean, delete or\
                replacement with a providede value."))


    def _apply_filter(self, data, sample_rate):
        """!
        Applies a Butterworth filter to the input data using the filter settings specified via config file.

        @param data         (pd.DataFrame) The input data to be filtered.
        @param sample_rate  (int or float) The sampling rate at which the data was recorded. 

        @retval data        (pd.DataFrame) The filtered input data. 
        """
        for channel in set(data.columns).intersection(self.filter_settings):
            filter_settings = self.filter_settings[channel]
            data[channel] = butter_filter(data[channel].to_numpy(), cut=filter_settings["cut"],
                fs=sample_rate, order=filter_settings["order"],f_type=filter_settings["ftype"]) 
        return data



def main():
    """!
    Main function of the slicing and feature extraction. Performs slice and feature extraction preprocessing 
    on the data and saves the resulting preprocessed data. Use config file for specification of settings. 
    """
    print("Starting slice and feature extraction preprocessing of OpenFace data...")
    Preprocessor = XITE_Preprocessor(data_type="video")
    Preprocessor.preprocess_parallel()
    # subj_preprocessor = Subject_Preprocessor("002")
    # subj_preprocessor.preprocess()

if __name__ == "__main__":
    main()

"""!
@}
"""