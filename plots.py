import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss


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
    plt.tight_layout()
    plt.show()