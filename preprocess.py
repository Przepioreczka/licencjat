import mne
from typing import Iterable, Union
import numpy as np
import scipy.signal as ss
from get_bads import get_bad_channels, get_bad_epochs
import pandas as pd

def preprocess(filename: str, fmin: Union[float, int], fmax: Union[float, int], 
               method: str, down_fs: Union[float, int], 
               tmin: Union[float, int], tmax: Union[float, int], exclude_ratio: Union[float, int], 
               method_bands: str, bands: Iterable[Iterable[Union[int, float]]], verbose: any, notch: False):
    if verbose == False:
        mne.set_log_level(verbose = 'CRITICAL')
    else:
        mne.set_log_level(verbose = None)
    
    '''Reading file'''
    file = mne.io.read_raw_eeglab(filename ,preload = True)
    
    '''Setting reference'''
    file.set_eeg_reference(ref_channels = ['A1','A2']) 
    
    '''Dropping unncessary channels'''
    file.drop_channels(['EOG','dioda','TSS','A1','A2'])
    
    '''Params'''
    Fs = int(file.info['sfreq'])
    channels = file.info['ch_names']
    
    '''First filtering'''
    file.filter(fmin, fmax, method = method, iir_params = dict(order=5, ftype='butter'))
    if notch:
        file.notch_filter(np.arange(50, 251, 50),method='fir')

    '''Finding and interpolating bad channels'''
    ch_ind, ch_names = get_bad_channels(file.get_data(), channels)
    file.info['bads'] = ch_names
    file.interpolate_bads(reset_bads = False, mode = 'accurate', origin = [0,0,0])
    
    '''Downsampling'''
    file.resample(down_fs)
    
    '''Marking events'''
    ratio = Fs//down_fs
    count = len(file.event['latency'])
    EVENTS = np.zeros((count,3))
    EVENTS[:,0] = file.event['latency']
    EVENTS[:,1] = 0
    EVENTS[:,2] = 1
    EVENTS = EVENTS/ratio
    
    '''Creating epochs'''
    epo = mne.Epochs(file, EVENTS.astype(int), metadata = pd.DataFrame(file.event['tag_type']), 
                     tmin = tmin, tmax = tmax, preload = True)
    
    '''Epoch params'''
    samples = len(epo._raw_times)
    time = epo._raw_times
    trials_count = len(epo.selection)
    nchan = epo.info['nchan']
    
    '''Finding mistaken trials'''
    mistakes = np.array([True if i['word_type'] != i['decision'] else False for i in file.event['tag_type']])
    
    '''Fidning bad epochs'''
    mistakes = np.array([True if i['word_type']!=i['decision'] else False for i in file.event['tag_type']])
    mistakes[get_bad_epochs(epo.get_data()).astype(int)] = True
    bad_epo = mistakes
    '''Excluding participant with more than half trials bad'''
    if np.count_nonzero(bad_epo) >= trials_count//exclude_ratio:
        return False
    epo.drop(bad_epo)
    
    '''Extracting epochs class'''
    word_type = np.array([0 if i == 'p' else 1 for i in epo._metadata['word_type']]).astype(int)
    
    '''Extracting frequency bands
       Hilbert envelope + tukey window'''
    no_bands = len(bands)
    trials_count = len(epo.selection)
    epochs = np.empty((no_bands, trials_count, nchan, samples))
    alpha = 0.1
    
    for ind, band in enumerate(bands):
        epochs[ind] = epo.copy().filter(band[0], band[1] ,method = method_bands, iir_params = dict(order=5, ftype='butter')).get_data()
        window = ss.tukey(samples, alpha)
        epochs[ind,:,:,:] = np.abs(ss.hilbert(epochs[ind,:,:,:] * window , axis = -1))
        alpha *= 0.9

    '''Baseline correction'''
    end_point = int(0.2*down_fs)
    epochs = (epochs - np.mean(epochs[:,:,:,:end_point], axis = (1,3)).reshape(no_bands,1,nchan,1) ) / np.std(epochs[:,:,:,:end_point], axis = (1,3)).reshape(no_bands,1,nchan,1)
    
    '''Resampling number 2'''
    epo_final = mne.filter.resample(epochs, up = 1, down = 4)

    return down_fs//4, epo_final, channels, word_type