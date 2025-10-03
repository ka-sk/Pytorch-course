import torch
from torch import nn # for neural networks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#Linear regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# Prepering data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)


def prediction_plot(X_train=X_train, 
                    X_test=X_test,
                    y_train=Y_train, 
                    y_test=y_test, 
                    prediction=None):
    plt.figure(0)
    plt.plot(X_test, y_test, '.b', label='Test data')
    plt.plot(X_train, y_train, '.g', label='Train data')

    if prediction is not None:
        plt.plot(X_test, prediction, '.m', label="Prediction")

    plt.legend()


torch.manual_seed(23)

model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr= 0.03)

# Epoch number
epochs = 300

# training and testing loss values
training_loss = np.empty(epochs)
testing_loss = np.empty(epochs)
epoch_count = np.arange(epochs)

for epoch in range(epochs):

    model_0.train()

    y_pred = model_0(X_train)

    loss = loss_fn(y_pred, Y_train)

    training_loss[epoch] = loss.detach().numpy()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_pred=model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
        testing_loss[epoch] = test_loss.numpy()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}/{epochs}, {model_0.state_dict()}')

prediction_plot(prediction=test_pred)
fig, axis = plt.subplots()
axis.plot(epoch_count, testing_loss, label='Testing')
axis.plot(epoch_count, training_loss, label='Trainig')
plt.show()