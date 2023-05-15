import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


class RawData:
    #Loading data and formating all singals into numpy arrays
    def __init__(self, file_name: str) -> None:
        eeg_mat = sio.loadmat(file_name)['eeg']
        self.c3 = eeg_mat[:,4]
        self.c4 = eeg_mat[:,5]
        self.labels = eeg_mat[:,25]
        self.freq = 160
        self.t = np.arange(0, len(self.c3)/160, 1/160)

        self.c3_train, self.c4_train, self.c3_test, self.c4_test, self.labels_train, self.labels_test = None,None,None, None, None, None

    def view_data(self):
        fig, axs = plt.subplots(3, 1)

        axs[0].plot(self.t, self.c3)
        axs[0].set_title("EEG sa kanala 3")

        axs[1].plot(self.t, self.c4)
        axs[1].set_title("EEG sa kanala 4")

        axs[2].plot(self.t, self.labels)
        axs[2].set_title("Obelezja")

        plt.show()

    def filter_data(self):
        fs = 160
        fcutlow = 8
        fcuthigh = 30
        nyq = 0.5 * fs

        # Compute filter coefficients
        Wn = [fcutlow/nyq, fcuthigh/nyq]
        b, a = butter(2, Wn, btype='band')

        # Filter the signals using filtfilt to obtain zero-phase filtering
        self.c3 = filtfilt(b, a, self.c3)
        self.c4 = filtfilt(b, a, self.c4)




