from scipy.stats import kurtosis, zscore
import numpy as np


def get_bad_channels(signal, ch_names, percentiles = [25,75]): #shape = (channels, samples)
    channels = signal.shape[1]
    Qstd = np.std(signal,axis = -1)
    Qkurt = kurtosis(signal,axis = -1)
    bads = np.empty(0)
    Q1,Q3 = np.percentile(Qstd,percentiles)
    IQR = (Q3-Q1)*1.5
    range_std = (Q1-IQR,Q3+IQR)
    Q1,Q3 = np.percentile(Qkurt,percentiles)
    IQR = (Q3-Q1)*1.5
    range_kurt = (Q1-IQR,Q3+IQR)
    right_std = np.where(Qstd > range_std[1])[0]
    left_std = np.where(Qstd < range_std[0])[0]
    right_kurt = np.where(Qkurt > range_kurt[1])[0]
    left_kurt = np.where(Qkurt < range_kurt[0])[0]
    bads = np.unique(np.concatenate((bads,left_kurt,right_kurt,left_std,right_std)))
    return bads.astype(int), np.array(ch_names)[bads.astype(int)]


def get_bad_epochs(signal,percentiles = [25,75]): #shape = ( epochs, channels, samples)
    channels = signal.shape[1]
    Qstd = np.std(signal,axis = -1)
    Qkurt = kurtosis(signal,axis = -1)
    bads = np.empty(0)
    for i in range(channels):
        Q1,Q3 = np.percentile(Qstd[:,i],percentiles)
        IQR = (Q3-Q1)*1.5
        range_std = (Q1-IQR,Q3+IQR)
        Q1,Q3 = np.percentile(Qkurt[:,i],percentiles)
        IQR = (Q3-Q1)*1.5
        range_kurt = (Q1-IQR,Q3+IQR)
        right_std = np.where(Qstd[:,i] > range_std[1])[0]
        left_std = np.where(Qstd[:,i] < range_std[0])[0]
        right_kurt = np.where(Qkurt[:,i] > range_kurt[1])[0]
        left_kurt = np.where(Qkurt[:,i] < range_kurt[0])[0]
        bads = np.unique(np.concatenate((bads,left_kurt,right_kurt,left_std,right_std)))
    return bads
