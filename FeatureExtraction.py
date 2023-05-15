import numpy as np
from RawData import RawData
from EEGDataFrame import EEGDataFrame
from Classification import ClassificationEEG
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

class FeatureExtraction:

    @classmethod
    def featutre_extraction(self, c3, c4, labels) :
        fs = 160
        time = 8.3
        step = int(fs*time)
        final_matrix = []
        current_vector = []
        data_set: np.ndarray = np.empty((0, 25), float)
        for i in range(0, len(c3), step):
            for ch in [c3, c4]:
                freqs, psd = welch(ch[i:i+step], fs=fs, nperseg=fs*2)
                spectral_entropy = entropy(psd)
                signal_variance = np.var(ch[i:i+step])
                peak_frequency = freqs[np.argmax(psd)]
                max_value = np.max(ch[i:i+step])
                current_vector.append(spectral_entropy)
                current_vector.append(signal_variance)
                current_vector.append(peak_frequency)
                current_vector.append(max_value)

            current_vector.append(labels[i:i+step][-1])
            final_matrix.append(current_vector)
            current_vector = []

        return np.array(final_matrix)




