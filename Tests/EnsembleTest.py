import pandas as pd
import numpy as np
import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from EnsembleModels import AdaBoosting, BaggingClassicDecisionTree, BaggingRandomForest
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay 

def load_dataset():
    csv_path = "datasets/yeet.csv"
    csv_df = pd.read_csv(csv_path)
    df_values = csv_df.values
    no_countries = df_values[:, 1:]
    x_data = no_countries[:, :-1]
    y_data = np.unstack(no_countries[:, -1])
    return x_data, y_data

x, y = load_dataset()
i = BaggingRandomForest(splits=4)
for i in [AdaBoosting, BaggingClassicDecisionTree, BaggingRandomForest]:
    print(i.__name__)
    cf, accuracy, kappa = i(seed=5).benchmark_kfold(x, y)
    print("Accuracy: ", accuracy)
    print("Kappa: ", kappa)
    print("Confusion Matrix: ")
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=np.unique(y)) 
    disp.plot() 
    plt.show()
    print(cf)