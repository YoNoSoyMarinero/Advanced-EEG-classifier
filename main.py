from RawData import RawData
from FeatureExtraction import FeatureExtraction
from EEGDataFrame import EEGDataFrame
from Classification import ClassificationEEG
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

        

class App(tk.Tk):

    def load_data(self):
        try:
            self.raw_data = RawData('EEG_zad3.mat')
            self.raw_data.filter_data()
            matrix = FeatureExtraction.featutre_extraction(self.raw_data.c3, self.raw_data.c4, self.raw_data.labels)
            self.df = EEGDataFrame(matrix)
            messagebox.showinfo("Info!", "Data loaded successfully!")
        except:
            messagebox.showerror("Error", "Failed to load data!")

    def plot_signals(self):
        self.raw_data.view_data()

    def train_model(self):
        try:
            self.model = ClassificationEEG().classification_model_train(self.df, self.combo_box.get())
            messagebox.showinfo("Info!", "Model trained successfully!")
        except:
            messagebox.showerror("Error", "Failed to train the model!")

    def val_model(self):
        params = ClassificationEEG.train_model_with_grid_search(self.df, self.combo_box.get())[1]
        param_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        self.best_params.set(param_str)
    
    def test_model(self):
        self.cm, self.accuracy = ClassificationEEG.classification_model_test(self.df, self.model)
        self.accuracy_str_var.set(str(self.accuracy * 100) + "%")
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['1', '2'])
        disp.plot(cmap="YlOrBr_r")
        plt.show()

    def __init__(self) -> None:
        super().__init__()
        self.title("App")
        self.background_color = "gray17"
        self.font_color = "black"
        self.geometry("1200x800")
        self.config(bg=self.background_color)
        self.df = None
        self.model = None
        self.cm = None
        self.accuracy = None
        self.raw_data = None
        self.accuracy_str_var = tk.StringVar()
        self.best_params = tk.StringVar()

        self.load_button = tk.Button(self, text="LOAD", bg=self.background_color, fg="lime green", font=30, width=17, height=4, command=self.load_data)
        self.load_button.grid(row=0, column=0, pady=40, padx=50)

        self.train_button = tk.Button(self, text="TRAIN", bg=self.background_color, fg="IndianRed1", font=30, width=17, height=4, command=self.train_model)
        self.train_button.grid(row=1, column=0, pady=40, padx=50)

        self.test_button = tk.Button(self, text="TEST", bg=self.background_color, fg="cyan3", font=30, width=17, height=4, command=self.test_model)
        self.test_button.grid(row=2, column=0, pady=40, padx=50)

        self.val_button = tk.Button(self, text="FIND BEST PARAMS", bg=self.background_color, fg="cyan3", font=30, width=17, height=4, command=self.val_model)
        self.val_button.grid(row=3, column=0, pady=40, padx=50)

        self.plot_button = tk.Button(self, text="PLOT SIGNALS", bg=self.background_color, fg="cyan3", font=30, width=17, height=4, command=self.plot_signals)
        self.plot_button.grid(row=4, column=0, pady=40, padx=50)

        tk.Label(self, text="Pick model: ", bg=self.background_color, fg="gray83", font=20, width=17, height=4).grid(row= 0, column = 1, pady = 0, padx=0)
        self.combo_box = ttk.Combobox(self, value=['linear', 'quadratic', 'knn', 'random_forest'], background=self.background_color, foreground=self.font_color, font=30, width=17, height=4)
        self.combo_box.current(0)
        self.combo_box.grid(row = 0, column = 2, pady = 0, padx=0)

        tk.Label(self, text="Accuracy: ", bg=self.background_color, fg="gray83", font=20, width=17, height=4).grid(row= 1, column = 1, pady = 0, padx=0)
        self.accuracy_label = tk.Label(self, fg="gray83", bg=self.background_color, font=20, textvariable=self.accuracy_str_var)
        self.accuracy_label.grid(row = 1, column = 2, pady = 0, padx=0)

        tk.Label(self, text="Best params: ", bg=self.background_color, fg="gray83", font=20, width=17, height=4).grid(row= 1, column = 3, pady = 0, padx=0)
        self.accuracy_label = tk.Label(self, fg="gray83", bg=self.background_color, font=20, textvariable=self.best_params)
        self.accuracy_label.grid(row = 1, column = 4, pady = 0, padx=0)
        

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()