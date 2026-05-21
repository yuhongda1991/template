import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Metrics:
    def __init__(self, metrics=None):
        if metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1']
        else:
            self.metrics = metrics

    def compute(self, y_true, y_pred):
        results = {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.metrics:
            results['precision'] = precision_score(y_true, y_pred, average='macro')
        
        if 'recall' in self.metrics:
            results['recall'] = recall_score(y_true, y_pred, average='macro')
        
        if 'f1' in self.metrics:
            results['f1'] = f1_score(y_true, y_pred, average='macro')
        
        return results

    def add_metric(self, metric_name):
        if metric_name not in self.metrics:
            self.metrics.append(metric_name)

    def remove_metric(self, metric_name):
        if metric_name in self.metrics:
            self.metrics.remove(metric_name)