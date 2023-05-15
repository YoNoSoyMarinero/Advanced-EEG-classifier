import pandas as pd
from sklearn.model_selection import train_test_split

class EEGDataFrame:

    def column_label_generator(self) -> list:
        features = ["Spectral entropy", "Signal variance", "Peak frequency", "Max Value"]
        column_labels = []
        for i in range(2):
            for feature in features:
                column_labels.append(feature + str(i + 1))

        column_labels.append("type")
        return column_labels
    
    def train_test_split(self, matrix):
        df = pd.DataFrame(matrix, columns=self.column_label_generator())
        return train_test_split(df, test_size=0.2, random_state=10)


    def __init__(self, matrix) -> None:
        self.train_df, self.test_df = self.train_test_split(matrix)