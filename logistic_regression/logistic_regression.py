import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

# 读取数据
path = 'ex2data1.txt'
data = pd.read_csv(path, names=('Exam1', 'Exam2', 'Admitted'), header=None)

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'],
           c='b', s=50, marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], c='r',
           s=50, marker='x', label='Not Admitted')
ax.legend(loc='best')
ax.set_title('Origin Data')
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(theta, X, y):
    theta, X, y = np.mat(theta), np.mat(X), np.mat(y)
    h = X * theta.T
    first = np.multiply(-y, np.log(sigmoid(h)))
    second = np.multiply(1-y, np.log(1-sigmoid(h)))
    return np.sum(first - second) / len(X)


def gradient(theta, X, y):
    theta, X, y = np.mat(theta), np.mat(X), np.mat(y)
    parameters = int(theta.ravel().reshape(1, -1).shape[1])
    grad = np.zeros(parameters)
    h = X * theta.T
    error = sigmoid(h) - y
    for j in range(parameters):
        term = np.multiply(error, X[:, j])
        grad[j] = np.sum(term) / len(X)
    return grad


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x > 0.5 else 0 for x in probability]


# 处理数据格式
data.insert(0, 'Ones', 1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.zeros(3)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

theta_min = np.mat(result[0])
predictions = predict(theta_min, X)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0)
           else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
