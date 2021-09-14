import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def costFunctionJ(X, y, theta):
    m = len(y)
    inner = np.power((X@theta.T - y), 2)
    cost = np.sum(inner) / (2*m)
    return cost

# 梯度下降法


def gradientDescent(X, y, theta, alpha, iters):
    m = len(y)
    temp = np.mat(np.zeros(theta.shape))
    parameters = int(theta.reshape(1, -1).shape[1])
    cost = np.zeros(iters)
    inner = X@theta.T - y
    for i in range(iters):
        for j in range(0, parameters):
            temp[0, j] = theta[0, j] - ((alpha / m) * np.sum(inner * X[:, j]))
        theta = temp
        cost[i] = costFunctionJ(X, y, theta)
    return theta, cost

# 正规方程法


def normalEquation(X, y):
    return np.linalg.inv(X.T@X)@X.T@y


# 读取数据
data = pd.read_csv('ex1data1.txt', names=('population', 'profit'))
data.insert(0, 'ones', 1)
X = data.iloc[:, :-1].values.reshape(-1, 2)
y = data.iloc[:, -1].values.reshape(-1, 1)
# theta = np.array([[-3.24140214, 1.1272942]], dtype=float)
theta = normalEquation(X, y)
theta = theta.T


def main():
    alpha = 0.01
    iters = 1000
    # final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    print(costFunctionJ(X, y, theta))
    x = np.linspace(data.population.min(), data.population.max(), 100)
    f = theta[0, 0] + theta[0, 1]*x
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(data.population, data.profit, c='r')
    ax.plot(x, f, c='b')
    plt.show()


if __name__ == "__main__":
    main()
