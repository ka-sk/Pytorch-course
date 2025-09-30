import torch
from torch import nn # for neural networks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prediction_plot(X_train, 
                    X_test,
                    y_train, 
                    y_test, 
                    prediction=None):
    plt.figure(0)
    plt.plot(X_test, y_test, '.b', label='Test data')
    plt.plot(X_train, y_train, '.g', label='Train data')
    
    if prediction is not None:
        plt.plot(X_test, prediction, '.m', label="Prediction")

    plt.legend()
    plt.show()
    

# Prepering data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)


