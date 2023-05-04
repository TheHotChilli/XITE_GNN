"""!
@file
@brief Dataset classes related to bio data of X-ITE pain database. 
@ingroup Datasets
@addtogroup Datasets
@{
"""

import pandas as pd
import glob 
import os
from scipy.io import loadmat
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))
from XITE_GNN import config

class Bio_raw_dataset():
    """!
    Class for loading bio data of the X-ITE pain database.
    """

    def __init__(self, dir_data=config.dir_bio_raw):
        """!
        Constructor of Bio_raw_dataset class.

        @param dir_data     Directory where the bio data is stored.
        """
        self.dir_data = dir_data
        self._channel_mapping = {'corrugator':0, 'zygomaticus':1, 'trapezius':2, 'scl':3, 'ecg':4}
        self.channels = list(self._channel_mapping)

    @property
    def subject_list(self):
        """!
        Property method to get a list of subject ids in the bio data directory.

        @retval subject_ids     List of subject ids.
        """
        fpaths = glob.glob(self.dir_data + "/*.mat")
        subject_ids = sorted([os.path.basename(fpath).split(".mat")[0].split("S")[1] for fpath in fpaths])
        return subject_ids

    def load_data(self, subject_id, target_channels=['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']):
        """!
        Method to load data for the specified subject, optionally specify which bio channels to use.

        @param subject_id       The id of the subject for which to load the data.
        @param target_channels  The list of bio channels to use. Default is all available channels.
        @retval df              A pandas DataFrame with the selected data columns.
        """
        data = loadmat(os.path.join(self.dir_data, f"S{subject_id}.mat"), variable_names=["data"])["data"]
        cols_to_use = [self._channel_mapping[channel] for channel in list(target_channels)]
        return pd.DataFrame(data[:,cols_to_use], columns=target_channels)

    def load_labels(self, subject_id):
        """!
        Method to load labels for the specified subject.

        @param subject_id   The id of the subject for which to load the labels.
        @retval labels      A numpy array of labels.
        """
        labels = loadmat(os.path.join(self.dir_data, f"S{subject_id}.mat"), variable_names=["stimuli"])["stimuli"]
        return labels


if __name__ == "__main__":
    bio_dataset = Bio_raw_dataset()
    labels = bio_dataset.load_labels("001")
    data = bio_dataset.load_data("001")
    print("Done")

"""!
@}
"""