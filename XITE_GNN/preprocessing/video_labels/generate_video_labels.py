"""!
@file
@brief Functions for computation of video labels from bio labels. 
@ingroup Video_Labels
@addtogroup Video_Labels
@{
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from scipy.io import loadmat
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parents[3]))
from XITE_GNN.config import dir_video_labels, dir_labels_raw, num_processes
from XITE_GNN.datasets.OpenFace import OpenFace_Raw_Dataset

def generate_video_labels_sequential():
    """!
    Generates video labels sequentially for each subject in the subject list from the OpenFace_Raw_Dataset object.
    """
    print("Starting video label generation...")
    openface_data = OpenFace_Raw_Dataset()
    for subj_id in openface_data.subject_list:
        generate_subject_video_labels(openface_data, subj_id)
    print("Finished video label generation.")

def generate_video_labels_parallel():
    """!
    Generates video labels in parallel for each subject in the subject list from the OpenFace_Raw_Dataset object
    """
    print("Starting video label generation...")
    openface_data = OpenFace_Raw_Dataset()
    Parallel(n_jobs=num_processes)(delayed(generate_subject_video_labels)(openface_data, subj_id) for subj_id in openface_data.subject_list)
    print("Finished video label generation.")


def generate_subject_video_labels(openface_dataset, subj_id, fs_video=25):
    """!
    Generates video labels for a subject. 
    
    @param openface_dataset     Instance of OpenFaceDataset class to load subject data
    @param subj_id              ID of the subject
    @param fs_video             Video frame rate (default 25 Hz)
    """
    try:
        label_data = loadmat(os.path.join(dir_labels_raw, f"S{subj_id}.mat"))
    except:
        print(f"Warning: No label file available for S{subj_id}")
        return
    fs_label = label_data["fs"][0][0]
    labels = np.squeeze(label_data["data"])
    nof_samples = len(labels)

    # Bestimme wo bio label Intervalle beginnen und aufhören 
    idxs_end = np.diff(labels).nonzero()[0]     # misses last endidx at nof_samples-1
    idxs_start = idxs_end + 1                   # misses first startidx at 0
    interval_labels = np.concatenate([[labels[0]],labels[idxs_start]])

    # Rechne von der schnelleren bio sampling frequenz (~1000Hz) auf die langsamere video
    # Frequenz (25 Hz) zurück. Nur jedes 40-e bio sample korrespondiert mit video sample.
    # Frequenzverhältnis f_video/f_bio = 1/40 stellt diesen Zusammenhang her. Multilpliziere
    # bio idxs mit 1/40 um video idxs zu erhalten. Achtung: Es ergeben sich krumme video idxs. 
    idxs_end = idxs_end * (fs_video/fs_label)
    idxs_start = idxs_start * (fs_video/fs_label)

    # Runden der krummen video end idxs auf nächsten ganzen idx Wert
    idxs_end_rounded = np.round(idxs_end)

    # Nehme diesen gerundeten end_idx als Referenz. Vergleiche nun ob die krummen start idxs oder
    # end idxs näher an diesem Referenz Wert liegen.
    is_endidx_mask = (np.abs(idxs_end_rounded - idxs_end) <= np.abs(idxs_end_rounded - idxs_start))

    idxs_end[is_endidx_mask] = idxs_end_rounded[is_endidx_mask]       # An stellen wo krumme endidx näher nehme Referenz als endidx
    idxs_end[~is_endidx_mask] = idxs_end_rounded[~is_endidx_mask] - 1 # An Stellen wo krumme startidx näher nehme Referenz-1 als endidx
    
    idxs_start[is_endidx_mask] = idxs_end_rounded[is_endidx_mask] + 1 # An Stellen wo krumme endidx näher nehme Referenz+1 als startidx
    idxs_start[~is_endidx_mask] = idxs_end_rounded[~is_endidx_mask]   # An Stellen wo krumme startidx näher nehme Referenz als startidx

    # add missing first start idx and last end idx
    idxs_start = np.concatenate([[0], idxs_start])
    idxs_end = np.append(idxs_end, np.round(nof_samples*(fs_video/fs_label)))

    # generate a video labels vector (label for each frame/sample)
    #labels_video = np.ndarray(int(idxs_end[-1])+1, dtype=int)
    nof_samples_video = len(openface_dataset.load_data(subj_id, target_channels=["AU01_r"]))
    labels_video = np.empty(nof_samples_video); labels_video[:] = np.nan
    for i in range(len(idxs_start)):
        start_idx = int(idxs_start[i])
        end_idx = int(idxs_end[i])
        label = interval_labels[i]
        labels_video[start_idx:end_idx+1] = label
        # Falls labels länger recorded wurden
        if end_idx >= (nof_samples_video-1):
            break

    # write to file
    if not os.path.exists(dir_video_labels): 
        os.makedirs(dir_video_labels)
    path = os.path.join(dir_video_labels, f"S{subj_id}.csv")
    pd.DataFrame(labels_video).to_csv(path, index=False)

    # Falls Videos länger als Label Recording
    if nof_samples_video > (idxs_end[-1]+1):
        diff = nof_samples_video - idxs_end[-1]+1
        print(f"Warning: For S{subj_id} video was recorded longer than labels (diff = {diff} samples = {diff*(1/fs_video):.2f}s; last available label = {interval_labels[-1]}).")
    print(f"Generated video labelfile for S{subj_id}")

if __name__ == "__main__":
    generate_video_labels_parallel()
    # generate_video_labels_sequential()

"""!
@}
"""