import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

dataset = load_iris()

# petal length and width
x = np.array(dataset.data[:, (2, 3)]).astype(np.int64)
# Iris Setosa?
y = (dataset.target == 0).astype(np.int64)

weights = np.array(np.random.rand(2, 150)).astype(np.int64)

class Perceptron:
  def __init__(self, weights, bias=1):
    self.weights = weights
    self.bias = bias
  
  def heaviside(self, z):
    for i in range(len(z)):
      if z[0][i] < 0:
        return -1
      if z[0][i] == 0:
        return 0
      if z[0][i] > 0:
        return 1

  def sgn(self, z):
    for i in range(len(z)):
      if z[0][i] < 0:
        return -1
      if z[0][i] == 0:
        return 0
      if z[0][i] > 0:
        return 1

  def activation(self, func, z):
    if func == 'heaviside':
      return self.heaviside(z)
    if func == 'sgn':
      return self.sgn(z)

  def rmse(self, y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))

  def update(self, x, y, y_hat, n, error):
    for i in range(len(x)):
      for j in range(len(y)):
        result = self.weights[i][j] + np.sum(np.dot((n * (y[j] - y_hat)), x[i]))
        self.weights[i][j] += result

  def fit(self, x, y, activation, n, epochs=1):
    for i in range(epochs):
      u = np.dot(self.weights, x)
      y_hat = self.activation(activation, u)
      
      error = self.rmse(y, y_hat)
      if error >= 0.5:
        self.update(x, y, y_hat, n, error)
        u = np.dot(self.weights, x)
        epochs += 1
      else:
        break

    return self.weights

  def predict(self, x, activation):
    u = np.dot(x, self.weights)
    y = self.activation(activation, u)
    if y == 0:
      return 'Iris-setosa'
    if y == 1:
      return 'Iris-versicolor'

perceptron = Perceptron(weights)
fitted = perceptron.fit(x, y, 'heaviside', 0.1)
pred = perceptron.predict(np.array([[2, 0.5]]).astype(np.int64), 'heaviside') # should return Iris-setosa
pred2 = perceptron.predict(np.array([[4, 0.5]]).astype(np.int64), 'heaviside') # should return Iris-versicolor

# print(pred)
print(pred2)
