import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

dataset = load_iris()

# petal length and width
x = dataset.data[:, (2, 3)]
# Iris Setosa?
y = (dataset.target == 0).astype(np.int64)

weights = np.random.rand(150, 2)

class Perceptron:
  def __init__(self, weights, bias=1):
    self.weights = weights
    self.bias = bias

  def rmse(self, y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))

  def update(self, x, y, y_hat, n, error):
    for i in range(len(x)):
      for j in range(len(y)):
        result = self.weights[i][0] + np.sum(np.dot((n * (y[j] - y_hat)), x[i]))
        self.weights[i][0] += result
        result = self.weights[i][1] + np.sum(np.dot((n * (y[j] - y_hat)), x[i]))
        self.weights[i][1] += result

  def fit(self, x, y, n, epochs=1):
    loop = True
    while loop == True:
      u = np.dot(np.transpose(self.weights), x)
      y_hat = np.heaviside(u, 0.5)
      print(y_hat)

      error = self.rmse(y, y_hat)

      if error >= 0.5:
        self.update(x, y, y_hat, n, error)
        u = np.dot(np.transpose(self.weights), x)
      else:
        loop = False

    return self.weights

  def predict(self, x):
    u = np.dot(x, np.transpose(self.weights))
    y = np.heaviside(u, 0.5)
    if y == 0:
      return 'Iris-setosa'
    if y == 1:
      return 'Iris-versicolor'

perceptron = Perceptron(weights)
fitted = perceptron.fit(x, y, 0.1)
pred = perceptron.predict([[2, 0.5]]) # should return Iris-setosa
pred2 = perceptron.predict([[4, 0.5]]) # should return Iris-versicolor

print(pred)
print(pred2)
