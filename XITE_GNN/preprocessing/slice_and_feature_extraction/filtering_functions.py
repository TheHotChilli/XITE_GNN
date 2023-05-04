# based on: https://github.com/Bennri/x-ite-feature-extraction

"""!
@file
@brief Implementation of Butterworth filter using scipy.signal.
@ingroup Slicing_and_Feature_Extraction
@addtogroup Slicing_and_Feature_Extraction
@{
"""

from scipy.signal import butter, filtfilt, sosfiltfilt, sosfilt
import numpy as np

def butter_params(cut, fs, order=5, f_type='band'):
    """! 
    Computes transfer function coefficients b,a of a butterworth filter.
    
    @parma cut      (double or list) Cutoff frequencies
    @parma fs       (double) Sampling frequencie of data 
    @param order    (int) order of butterworth filter
    @param f_type   (string) Filter type (can be {'lowpass', 'highpass', 'bandpass', 'bandstop'})

    @retval filter_coefficients     (tuple) filter coefficients b,a of butterworth filter. 
    """
    f_nyq = 0.5 * fs

    if (not isinstance(cut, list)) or (len(cut) == 1):
        if isinstance(cut, list): cut = cut[0]
        f_cut_normalized = cut / f_nyq
        b, a = butter(order, f_cut_normalized, btype=f_type, analog=False)
        if not np.all(np.abs(np.roots(a))<1):
            raise(RuntimeError("Instable Filter Coefficients"))
        return b, a
    elif len(cut) == 2:
        low = cut[0] / f_nyq
        high = cut[1] / f_nyq
        b, a = butter(order, [low, high], btype=f_type, analog=False)
        return b, a
    else:
        raise ValueError('Wrong input for cutoff frequency! Input was {} but needs to be [f_low], [f_high] or [f_low, f_high].'.format(cut))


def butter_params_sos(cut, fs, order=5, f_type="band"):
    """! 
    Computes transfer function coefficients of a butterworth filter as second order sections (sos).
    
    @parma cut      (double or list) Cutoff frequencies
    @parma fs       (double) Sampling frequencie of data 
    @param order    (int) order of butterworth filter
    @param f_type   (string) Filter type (can be {'lowpass', 'highpass', 'bandpass', 'bandstop'})

    @retval sos     (tuple) filter coefficients of butterworth filter as second order sections (sos).  
    """
    f_nyq = 0.5 * fs
    if (not isinstance(cut, list)) or (len(cut)==1):
       if isinstance(cut, list): cut = cut[0] 
       f_norm = cut / f_nyq
       sos = butter(order, f_norm, btype=f_type, output="sos", analog=False)
       return sos
    elif len(cut) == 2:
        f_low = cut[0] / f_nyq
        f_high = cut[1] / f_nyq
        sos = butter(order, [f_low, f_high], btype=f_type, output="sos", analog=False)
        return sos
    else:
        raise ValueError('Wrong input for cutoff frequency! Input was {} but needs to be [f_low], [f_high] or [f_low, f_high].'.format(cut))


def butter_filter(data, cut, fs, order=1, f_type='band', axis=0):
    """! 
    Applies a butterworth filter on input data. 

    @parma data     (Array like) Data to be filtered
    @parma cut      (double or list) Cutoff frequencies
    @parma fs       (double) Sampling frequencie of data 
    @param order    (int) order of butterworth filter
    @param f_type   (string) Optional; Filter type (can be {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}). Defaults to "band". 
    @param axis     (int) Optional, defaults to 0 (=row-wise); Specifies if filter is applied row-wise (axis=0) or col-wise (axis=1).

    @retval data    (Array like) Filterd input data
    """
    # b, a = butter_params(cut, fs, order=order, f_type=f_type)
    # y = filtfilt(b, a, data, axis=0)
    sos = butter_params_sos(cut, fs, order, f_type)
    #return sosfiltfilt(sos, data)
    return sosfilt(sos, data)


"""!
@}
"""