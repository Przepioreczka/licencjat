import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from typing import Iterable, Union

def before_after_plot(before_data, after_data, fs, channel, ch_name, name, start = 150, stop = 250, Fs_new = None): 
    time = np.arange(start,stop,1/fs)
    plt.figure(figsize = (13,5))
    plt.subplot(1,2,1)
    plt.plot(time, before_data[channel,start*fs:stop*fs])
    plt.title(f'Channel {ch_name}, before ' + name)
    plt.xlabel('time [s]')
    
    ratio = 1
    if Fs_new:
        ratio = int(fs/Fs_new)
        time = np.arange(start, stop, 1/(fs/ratio))
        
    plt.subplot(1,2,2)
    plt.plot(time, after_data[channel,start*fs//ratio:stop*fs//ratio])
    plt.title(f'Channel {ch_name}, after ' + name)
    plt.xlabel('time [s]')
    plt.show()
    
def plot_mean_welch(epochs, fs, names, xlim = None, ylim = None):
    no_bands = len(names)
    plt.figure(figsize = (11,10))
    f, welch = ss.welch(epochs, fs=fs, axis = -1)
    mean_welch = np.mean(welch, axis = (1,2))
    for i in range(no_bands):
        plt.subplot(no_bands,1,i+1)
        plt.title(names[i])
        plt.plot(f,mean_welch[i,:])
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD")
    plt.tight_layout()
    plt.show()
    
    
def plot_features(features: Iterable, filt: int, ylim: Iterable):
    feature_maps = np.mean(features, axis=0)
    feature_std = 3*np.std(features, axis=0)/np.sqrt(features.shape[0])
    for i in range(5):
        plt.figure(figsize = (30,3))
        for x in range(filt):
            ax = plt.subplot(1, 8, x+1)
            curv1 = feature_maps[:, i, x] + feature_std[:, i, x]      
            curv2 = feature_maps[:, i, x] - feature_std[:, i, x]
            plt.fill_between(np.arange(0, feature_maps.shape[0]), curv1, curv2, alpha = 0.2, color = 'r')
            plt.plot(feature_maps[:, i, x])
            plt.ylim(ylim[0], ylim[1])
        plt.show()

def plot_compare_features(features1: Iterable, features2: Iterable, filt: int, ylim: Iterable[float]):
    feature_maps1 = np.mean(features1, axis=0)
    feature_std1 = 3*np.std(features1, axis=0)/np.sqrt(features1.shape[0])
    feature_maps2 = np.mean(features2, axis=0)
    feature_std2 = 3*np.std(features2, axis=0)/np.sqrt(features2.shape[0])
    for i in range(5):
        plt.figure(figsize = (30,3))
        for x in range(filt):
            ax = plt.subplot(1, 8, x+1)
            curv11 = feature_maps1[:, i, x] + feature_std1[:, i, x]      
            curv21 = feature_maps1[:, i, x] - feature_std1[:, i, x]
            curv12 = feature_maps2[:, i, x] + feature_std2[:, i, x]      
            curv22 = feature_maps2[:, i, x] - feature_std2[:, i, x]
            plt.fill_between(np.arange(0, feature_maps1.shape[0]), curv11, curv21, alpha = 0.2, color = 'r')
            plt.fill_between(np.arange(0, feature_maps2.shape[0]), curv12, curv22, alpha = 0.2, color = 'black')
            plt.plot(np.linspace(-8/64,58/64, 66), feature_maps1[:, i, x], color = 'r')
            plt.plot(np.linspace(-8/64,58/64, 66), feature_maps2[:, i, x], color = 'black')
            plt.ylim(ylim[0], ylim[1])
        plt.show()