"""!
@file
@brief Dataset classes related to OpenFace results of the frontal face videos of X-ITE pain database.
@ingroup Datasets
@addtogroup Datasets
@{
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import sys
import torch 
from torch_geometric.data import Data 
from torch_geometric.utils import dense_to_sparse

sys.path.append(str(Path(__file__).parents[2]))
from XITE_GNN import config
from XITE_GNN.datasets.XITE import XITE



class OpenFace_Raw_Dataset():
    """!
    A dataset class for loading raw OpenFace data for X-ITE pain database.
    """

    def __init__(self, root_data=config.dir_video_openface, root_labels=config.dir_video_labels):
        """!
        Constructor of OpenFace_Raw_Dataset class.

        @param root_data    (str) Root directory containing the OpenFace results data. Defaults to the path specified in the config. 
        @param root_labels  (str) Root directory containing the corresponding labels. Defaults to the path specified in the config. 
        """
        self.root_data = root_data 
        self.root_labels = root_labels  
        self.subject_list = self._get_subject_ids() 
        self.channels = self._get_channel_names() 
        self.channels_AUr = [elem for elem in self.channels if "AU" in elem and "_r" in elem]          
        self.channels_AUc = [elem for elem in self.channels if "AU" in elem and "_c" in elem]           
        self.channels_fl2d = [elem for elem in self.channels if "x_" in elem]                           
        self.channels_fl3d = [elem for elem in self.channels if "X_" in elem]
        self.channels_pose = [elem for elem in self.channels if "pose_" in elem]

    def _get_subject_ids(self):
        """!
        Gets the subject ids of all participants for which an OpenFace data file is available.

        @retval subject_ids    (numpy.ndarray) 1D-Array of subject ids.
        """
        fpaths = glob.glob(self.root_data + "/*.csv")
        subject_ids = sorted([os.path.basename(fpath).split("_fvf")[0].split("S")[1] for fpath in fpaths])
        return np.array(subject_ids)

    def _get_channel_names(self):
        """!
        Gets the channel names (column header names) in the OpenFace results files. 

        @retval channel_names   (numpy.ndarray) 1D-Array of channel names.
        """
        fpath = glob.glob(self.root_data + "/*.csv")[0]
        channel_names = pd.read_csv(fpath, skipinitialspace=True, nrows=0).columns.values
        return channel_names

    def load_data(self, subject_id, target_channels=None):
        """!
        Loads the OpenFace data for a given subject ID.

        @param subject_id           (str or int) Subject id of the participant. E.g. as string "001" or as int 1. 
        @param target_channels      (list or str) List of channel names to be loaded or a string representing one of the following options: 
                                        "AUr": Load all Action Unit Regression channels, 
                                        "AUc": Load all Action Unit Classification channels, 
                                        "pose": Load all Head Pose channels, 
                                        "fl2d": Load all Facial Landmark 2D-coordinates, 
                                        "fl3d": Load all Facial Landmark 3D-coordinates}.

        @retval data                (pandas.DataFrame) Dataframe with shape (n_frames x n_channels).
        """
        if not isinstance(subject_id, str):
            subject_id = "{:03d}".format(subject_id)
        
        # load all data or selected channels
        path = os.path.join(self.root_data, f"S{subject_id}_fvf.csv")
        if not target_channels:
            data = pd.read_csv(path, na_values=" NaN", skipinitialspace=True)
        elif target_channels == "AUr":
            data = pd.read_csv(path, na_values=" NaN", usecols=self.channels_AUr, skipinitialspace=True)
        elif target_channels == "AUc":
            data = pd.read_csv(path, na_values=" NaN", usecols=self.channels_AUc, skipinitialspace=True)
        elif target_channels == "pose":
            data = pd.read_csv(path, na_values=" NaN", usecols=self.channels_pose, skipinitialspace=True)
        elif target_channels == "fl2d":
            data = pd.read_csv(path, na_values=" NaN", usecols=self.channels_fl2d, skipinitialspace=True)
        elif target_channels == "fl3d":
            data = pd.read_csv(path, na_values=" NaN", usecols=self.channels_fl3d, skipinitialspace=True)
        else:
            data = pd.read_csv(path, na_values=" NaN", usecols=target_channels, skipinitialspace=True)
        #data.columns = [col.replace(" ", "") for col in data.columns]
        return data

    def load_labels(self, subject_id):
        """!
        Load the labels for a given subject ID.
        
        @param subject_id   (str) ID of the subject for which to load labels. Example: "002". 

        @retval labels      (numpy.ndarray) 1D-Array of the labels. 
        """
        path = os.path.join(self.root_labels, f"S{subject_id}.csv")
        labels = pd.read_csv(path).values
        return np.squeeze(labels)



class OpenFace_Features_Dataset(XITE):
    """!
    A dataset class for loading the features that describe pain related signal segments/slices.
    """

    def __init__(self, root_path, use_labels=None, adj_matrix=None):
        """!
        Constructor of OpenFace_Features_Dataset class.

        @param root_path    (str) Path to the dataset file.
        @param use_labels   (list) List of labels to use for filtering the dataset, default is None (use all).
        """
        self.root_path = root_path
        self.df = self._load_data(use_labels)
        self.subject_list = self.df.index.get_level_values("subj_id").unique().sort_values().to_numpy()
        self.labels_list = self.df.index.get_level_values("label").unique().sort_values().to_numpy()
        self._pain_labels_list = [elem for elem in self.labels_list if elem in self.labels_pain]
        self._base_labels_list = [elem for elem in self.labels_list if elem in self.labels_base]
        self.classes = {i:base_label for (i,base_label) in enumerate(self._base_labels_list)}
        self.classes.update({i+len(self.classes):pain_label for (i,pain_label) in enumerate(self._pain_labels_list)})
        self.num_classes = len(self.labels_list)
        self.labels = self.df.index.get_level_values("label").to_numpy()
        self._transform_labels()
        self.adj_matrix = adj_matrix
        self.num_nodes = len(self.df.columns.get_level_values("channel").unique())
        self.num_node_features = self.df.columns.get_level_values("channel").value_counts()[0]

    def _load_data(self, use_labels=None):
        """!
        Load the raw dataset. Optionally filter for specific labels.

        @param use_labels   (list) If not None, a list of labels to use for filtering the dataset.

        @retval df          (pandas dataframe) Multiindex pandas dataframe with the features. Columns represent features, rows represent samples.
                            Row index contains the following information ["subj_id", "slice_id", "label", "start_idx", "end_idx"]. 
        """
        df = pd.read_csv(self.root_path, index_col=[0,1,2,3,4], header=[0,1])
        df.index.names = ["subj_id", "slice_id", "label", "start_idx", "end_idx"]
        if use_labels:
            df = df[df.index.get_level_values("label").isin(use_labels)]
        df = df.sort_index(level="subj_id")
        return df
    
    def _transform_labels(self):
        """!
        Transform labels into ascending labels starting at zero [0,1,2,3,...]. First labels represent base labels, following labels
        pain labels. In binary case 0=base, 1=pain. 
        """
        for (i,base_label) in enumerate(self._base_labels_list):
            self.labels[self.labels==base_label] = i
        for (i,pain_label) in enumerate(self._pain_labels_list):
            self.labels[self.labels == pain_label] = i + len(self._base_labels_list)

    @property
    def subjects(self):
        """!
        Returns the array of subject IDs in the dataset.
        
        @retval subj_ids    (numpy array) 1D-Array of all subject ids.
        """
        return self.df.index.get_level_values("subj_id").to_numpy()

    # @property
    # def labels(self):
    #     return self.df.index.get_level_values("label").to_numpy()

    @property
    def data(self):
        """!
        Returns the data samples as dataframe.
        
        @retval df      (pandas dataframe) Multiindex pandas dataframe with the features. Columns represent features, rows represent samples.
        """
        return self.df.reset_index(drop=True)

    @property
    def AU_names(self):
        """!
        Returns a list of AU names as string. E.g. ["AU01", "AU02", ...].

        @retval AU_names    (list) List of AU names. 
        """
        return [elem.str.replace("_r","") for elem in self.df.index.get_level_values(0)]

    def split_train_test(self, test_subjects):
        """!
        Splits the dataset into training and testing subsets based on the list of test subjects provided.

        @param test_subjects    (list) A list of subject IDs to be used for testing.

        @retval train_df        (pandas.DataFrame): A dataframe containing the data for the training subset.
        @retval test_df         (pandas.DataFrame): A dataframe containing the data for the testing subset.
        """
        train_df = self.df[~self.df.index.get_level_values("subj_id").isin(test_subjects)]
        test_df = self.df[self.df.index.get_label_values("subj_id").isin(test_subjects)]
        return train_df, test_df

    def get_train_test_idxs(self, test_subjects):
        """!
        Returns the indices within the subject list of training and testing subjects.

        @param test_subjects    (list) A list of subject IDs to be used for testing.

        @retval train_idxs      (numpy.ndarray) An array containing the indices of training subjects.
        @retval test_idxs       (numpy.ndarray) An array containing the indices of testing subjects.
        """
        subj_mask = np.isin(self.subjects, test_subjects)
        test_idxs = subj_mask.nonzero()[0]
        train_idxs =(~subj_mask).nonzero()[0]
        return train_idxs, test_idxs

    def to_torch_geometric(self):
        """!
        Converts the dataset to a list of PyTorch Geometric graphs. 
        Each list entry is a PyTorchGeometric Data object and represents a graph sample.

        @retval graph_list      (list): A list of PyTorch Geometric graphs.
        """
        num_features_per_node = self.df.columns.get_level_values("channel").value_counts()
        # check that all nodes have same nr of features
        assert (num_features_per_node.to_numpy() == self.num_node_features).all(), \
            "Inconsistent number of node features for different nodes"
        # transform labels from arbitrary range to range [0,C-1]
        graph_list = []
        #for i,row in self.df.iterrows():
            # x = torch.tensor(row.values.reshape(self.num_nodes, self.num_node_features)).float()
            # y = self._transform_label(row.name[2], binary_base_labels, binary_pain_labels)
            # y = torch.tensor(y)
        for i in range(len(self.labels)):
            x = torch.tensor(self.df.iloc[i,:].values.reshape(self.num_nodes, self.num_node_features)).float()
            y = torch.tensor(self.labels[i])
            edge_index, edge_weights = dense_to_sparse(torch.tensor(self.adj_matrix.values))
            graph_list.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weights.float())) 
        return graph_list

    # def _transform_label(self, label, binary_base_labels, binary_pain_labels):
    #     """
    #     Transform labels from arbitrary range to range [0,C-1]. The first 0,...,k labels are baseline, the
    #     following k+1,...,C-1 labels represent pain classes. If binary_base_labels is set all values are baseline
    #     labels are set to 0. If binary_pain_labels is set all pain labels are set to same label. 
    #     """
    #     if label in self._base_labels_list:
    #         if binary_base_labels: 
    #             return 0
    #         else:
    #             return self._base_labels_list.index(label)
    #     elif label in self._pain_labels_list:
    #         if binary_pain_labels:
    #             if binary_base_labels:
    #                 return 1
    #             else:
    #                 return len(self._base_labels_list) + 1
    #         else:
    #             if binary_base_labels:
    #                 return self._pain_labels_list.index(label) + 1
    #             else:
    #                 return self._pain_labels_list.index(label) + len(self._base_labels_list)              


# class Openface_Features_Graph_Dataset():
#     def __init__(self) -> None:
#         pass


class OpenFace_Slices_Dataset():
    """!
    Dataset class for loading the slices of OpenFace AU intensities that result from segmentation preprocessing.
    """
    def __init__(self, root_path, use_labels=None):
        """!
        Constructor of OpenFace_Slices_Dataset class.
        """
        self.root_path = root_path
        self.df = self._load_data(use_labels)

    def _load_data(self, use_labels=None):
        df = pd.read_csv(self.root_path)
        if use_labels:
            df = df[df["label"].isin(use_labels)]
        # reset index
        end_idxs = np.diff(df["slice_id"]).nonzero()[0]
        start_idxs = end_idxs + 1
        end_idxs = np.append(end_idxs, [len(df)-1])
        start_idxs = np.concatenate([[0], start_idxs])
        new_idx = np.ndarray(len(df))
        for i in range(len(start_idxs)):
            new_idx[start_idxs[i]:end_idxs[i]] = i
        df.set_index(new_idx)
        return df


if __name__ == "__main__":
    data = OpenFace_Raw_Dataset()
    labels = data.load_labels("001")
    print("Done")


"""!
@}
"""