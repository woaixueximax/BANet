
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import mne
import scipy.io as scio

class Load_BCIC_2b():

    def __init__(self, data_path, persion):
        self.stimcodes_train = ('769', '770')
        self.stimcodes_test = ('783')
        self.data_path = data_path
        self.persion = persion
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        self.train_name = ['1', '2', '3']
        self.test_name = ['4', '5']
        super(Load_BCIC_2b, self).__init__()

    def get_epochs_train(self, tmin=-0., tmax=4, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        x_data = []
        y_labels = []
        for session in self.train_name:
            file_to_load = 'B0{}0{}T.gdf'.format(self.persion+1, session)
            raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
            data_path_label = self.data_path + 'label/' + 'B0{}0{}T.mat'.format(self.persion+1, session)
            mat_label = scio.loadmat(data_path_label)
            mat_label = mat_label['classlabel'][:, 0] - 1
            if low_freq and high_freq:
                raw_data.filter(l_freq=low_freq, h_freq=high_freq)
            if downsampled is not None:
                raw_data.resample(sfreq=downsampled)
            self.fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims = [value for key, value in event_ids.items() if key in self.stimcodes_train]
            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(self.channels_to_remove)
            self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
            self.x_data = epochs.get_data()
            x_data.extend(self.x_data[:, :, :-1])
            y_labels.extend(self.y_labels)

        x_data = np.array(x_data)
        y_labels = np.array(y_labels)
        eeg_data = {'x_data': x_data,
                    'y_labels': y_labels,
                    'fs': self.fs}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=4, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        x_data = []
        y_labels = []
        for session in self.test_name:
            file_to_load = 'B0{}0{}E.gdf'.format(self.persion+1, session)
            raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
            data_path_label = self.data_path + 'label/' + 'B0{}0{}E.mat'.format(self.persion+1, session)
            mat_label = scio.loadmat(data_path_label)
            mat_label = mat_label['classlabel'][:, 0] - 1
            if (low_freq is not None) and (high_freq is not None):
                raw_data.filter(l_freq=low_freq, h_freq=high_freq)
            if downsampled is not None:
                raw_data.resample(sfreq=downsampled)
            self.fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims = [value for key, value in event_ids.items() if key in self.stimcodes_test]
            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(self.channels_to_remove)
            self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
            self.x_data = epochs.get_data()
            x_data.extend(self.x_data[:, :, :-1])
            y_labels.extend(self.y_labels)

        x_data = np.array(x_data)
        y_labels = np.array(y_labels)
        eeg_data = {'x_data': x_data,
                    'y_labels': y_labels,
                    'fs': self.fs}
        return eeg_data
#%%
def load_BCI2a_data(data_path, subject, training, all_trials = True):

    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48     
    window_Length = 7*250 
    
    # Define MI trial window 
    fs = 250          # sampling rate
    t1 = int(1*fs)  # start time_point
    t2 = int(5*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1        
    

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return



#%%
import json
from mne.io import read_raw_edf
from dateutil.parser import parse
import glob as glob
from datetime import datetime

#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


#%%
def get_data(path, subject, dataset = 'BCI2a',isStandard = True, isShuffle = True, onehot= True):
    
    # Load and split the dataset into training and testing 

    if (dataset == 'BCI2a'):
        path = path + 's{:}/'.format(subject + 1)
        X_train, y_train = load_BCI2a_data(path, subject + 1, True)
        X_test, y_test = load_BCI2a_data(path, subject + 1, False)

    elif dataset == 'BCI2b':
        load_raw_data = Load_BCIC_2b(path, subject)
        eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.)
        X_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
        eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
        X_test, y_test = eeg_data['x_data'], eeg_data['y_labels']

    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    if onehot == True:
        y_test_onehot = to_categorical(y_test)
    else: y_test_onehot=y_test
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

