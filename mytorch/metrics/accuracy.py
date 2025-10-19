import numpy as np
def accuracy(pred, y):
    pred_labels = np.argmax(pred, axis=1)
    temp = pred_labels == y.data
    return np.mean(temp)
