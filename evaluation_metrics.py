# evaluation_metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class EvaluationMetrics:
    def __init__(self, y_actual, y_predicted, c, seed):
        self.seed = seed
        self.C = c
        self.accuracy = accuracy_score(y_actual, y_predicted)
        self.precision = precision_score(y_actual, y_predicted)
        self.recall = recall_score(y_actual, y_predicted)
        self.f1 = f1_score(y_actual, y_predicted)
        self.cm = confusion_matrix(y_actual, y_predicted)

    def get_properties(self):
        return self.accuracy, self.precision, self.recall, self.f1, self.cm, self.C, self.seed
