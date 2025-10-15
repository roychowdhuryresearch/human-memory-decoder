import os
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from src.param.param_data import LABELS8


class Evaluator:
    def __init__(self, config, fold):
        self.config = config
        self.classes = LABELS8

        self.fold = fold
    
    def calculate_f1(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, zero_division=np.nan, average='macro')
        return f1
    
    def calculate_accuracy(self, y_true, y_pred):
        acc = []
        for i in range(len(self.classes)):
            acc.append(accuracy_score(y_true[:,i], y_pred[:,i]))
        accuracy = np.mean(acc)
        return accuracy
