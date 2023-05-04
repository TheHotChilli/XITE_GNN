"""!
@file
@brief Config file to specify settings for the slicing and feature extraction preprocessing step of X-ITE video data.
@ingroup Slicing_and_Feature_Extraction
@addtogroup Slicing_and_Feature_Extraction
@{
"""

#### Root Paths
dir_data = "/home/jbauske/01_Data/00_X-ITE/OpenFace"                # root dir openface data
dir_labels = "/home/jbauske/01_Data/00_X-ITE/Label"                 # root dir video labels
dir_export = "/home/jbauske/01_Data/00_X-ITE/Feature_Extraction"    # export dir results

#### Parallel Processing
nof_processes = 34

#### Export Settings
slice_extraction = True
feature_extraction = True

#### Subjects to exclude
subjects_no_use = ["014", "024", "024_b", "028", "030", "030_2", "059"]
# subjects_no_use = ["014", "024", "024_b", "030", "030_2", "059", "042", "005", "004"]

#### Normalization/Standardization settings:
rescale_slices = True
rescale_features = True

#### Butterworth Filter settings
filter_settings = {
    ## OpenFace/Video
    "AU01_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU02_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU04_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU05_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU06_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU07_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU09_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU10_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU12_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU14_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU15_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU17_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU20_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU23_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU25_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU26_r": {"ftype":"lowpass", "order":1, "cut":1},
    "AU45_r": {"ftype":"lowpass", "order":1, "cut":1},
}

#### Feature Extraction Settings
default_features_AU = [# Features Werner et al
                      "signal_mean", "signal_median", "signal_min", "signal_max", "signal_range", "signal_std",
                      "signal_iqr", "signal_idr", "signal_mad", "signal_tmax", "signal_tgm", "signal_tga",
                      "signal_sgm", "signal_sga", "signal_area", "signal_area_min_max",
                      # Features added Ricken et al
                      "signal_mean_crossing",
                      "signal_split_equal_part_mean", "signal_split_equal_part_std",
                      "signal_var", "signal_rms",
                      "mean_absolute_values", "std_absolute_values",
                      "signal_split_equal_part_var",
                      # problematic features for video
                      "signal_area_min_max_ratio",
                      "signal_mean_local_max", "signal_mean_local_min",  "signal_p2pmv", 
                      #"max_to_min_peak_value_ratio"    # immer raus, da immer teilen durch min=0
]

default_features_deriv = ["signal_mean", "signal_median", "signal_min", "signal_max", "signal_range", "signal_std",
                          "signal_iqr", "signal_idr", "signal_mad", "signal_tmax", "signal_tgm", "signal_tga",
                          "signal_sgm", "signal_sga", "signal_area", "signal_area_min_max",
                          "signal_mean_crossing",
                          "signal_var", "signal_rms", 
                          "mean_absolute_values", "std_absolute_values",
                          # problematic features
                          #"max_to_min_peak_value_ratio", # immer raus, da immer teilen durch min=0
]

feature_extraction_settings = {
    ## OpenFace/Video
    "AU01_r": default_features_AU,
    "AU02_r": default_features_AU,
    "AU04_r": default_features_AU,
    "AU05_r": default_features_AU,
    "AU06_r": default_features_AU,
    "AU07_r": default_features_AU,
    "AU09_r": default_features_AU,
    "AU10_r": default_features_AU,
    "AU12_r": default_features_AU,
    "AU14_r": default_features_AU,
    "AU15_r": default_features_AU,
    "AU17_r": default_features_AU,
    "AU20_r": default_features_AU,
    "AU23_r": default_features_AU,
    "AU25_r": default_features_AU,
    "AU26_r": default_features_AU,
    "AU45_r": default_features_AU,
}

feature_extraction_settings_1st_derivative = {
    "AU01_r": default_features_deriv,
    "AU02_r": default_features_deriv,
    "AU04_r": default_features_deriv,
    "AU05_r": default_features_deriv,
    "AU06_r": default_features_deriv,
    "AU07_r": default_features_deriv,
    "AU09_r": default_features_deriv,
    "AU10_r": default_features_deriv,
    "AU12_r": default_features_deriv,
    "AU14_r": default_features_deriv,
    "AU15_r": default_features_deriv,
    "AU17_r": default_features_deriv,
    "AU20_r": default_features_deriv,
    "AU23_r": default_features_deriv,
    "AU25_r": default_features_deriv,
    "AU26_r": default_features_deriv,
    "AU45_r": default_features_deriv,
}
feature_extraction_settings_2nd_derivative = {
    "AU01_r": default_features_deriv,
    "AU02_r": default_features_deriv,
    "AU04_r": default_features_deriv,
    "AU05_r": default_features_deriv,
    "AU06_r": default_features_deriv,
    "AU07_r": default_features_deriv,
    "AU09_r": default_features_deriv,
    "AU10_r": default_features_deriv,
    "AU12_r": default_features_deriv,
    "AU14_r": default_features_deriv,
    "AU15_r": default_features_deriv,
    "AU17_r": default_features_deriv,
    "AU20_r": default_features_deriv,
    "AU23_r": default_features_deriv,
    "AU25_r": default_features_deriv,
    "AU26_r": default_features_deriv,
    "AU45_r": default_features_deriv,
}

#### Slicing settings
slice_channels = [
    ## Video
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

# Shift slice start relative to regular interval start
# positive values -> shift right
# negative values -> shift left
slice_shifts = {
    "pH" : 0,
    "pE" : 0,
    "tH" : 0,
    "tE" : 0,
    "BpH" : 4,
    "BpE" : 4,
    "BtH" : 120,
    "BtE" : 120
}

# Duration/length of slices (How long is a slice?)
slice_lengths = {
    "pH" : 6,
    "pE" : 6,
    "tH" : 60,
    "tE" : 60,
    "BpH" : 6,
    "BpE" : 6,
    "BtH" : 60,
    "BtE" : 60
}

# only consider intervals that had this length
interval_min_lengths = {
    "pH" : 4,
    "pE" : 5,
    "tH" : 60,
    "tE" : 60
}

# min duration of previous (base)interval
pre_interval_min_lengths = {
    "pH" : 8,
    "pE" : 8,
    "tH" : 60,
    "tE" : 60
}

# min duration of subsequent (base)interval
post_interval_min_lengths = {
    "pH" : 0,
    "pE" : 0,
    "tH" : 0,
    "tE" : 0
}

"""!
@}
"""