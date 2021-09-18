import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def costFunctionJ(x, y, theta):
    '''代价函数'''
    m = np.size(x, axis=0)
    predictions = x*theta
    sqrErrors = np.power((predictions - y), 2)
    j = 1/(2*m)*np.sum(sqrErrors)
    return j


def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    alpha为学习率
    num_iters为迭代次数
    '''
    m = len(y)
    n = len(theta)
    temp = np.mat(np.zeros([n, num_iters]))  # 用来暂存每次迭代更新的theta值，是一个矩阵形式
    j_history = np.mat(np.zeros([num_iters, 1]))  # 记录每次迭代计算的代价值
    h = x*theta
    for i in range(num_iters):   # 遍历迭代次数
        temp[:, i] = theta - (alpha/m)*np.dot(x[:, 1].T, (h-y)).sum()
        theta = temp[:, i]
        j_history[i] = costFunctionJ(x, y, theta)
    return theta, j_history, temp


data = pd.read_csv('ex1data1.txt', names=('population', 'profit'))
data.insert(0, 'ones', 1)
x = np.mat(data.iloc[:, :-1]).reshape(-1, 2)
y = np.mat(data.iloc[:, -1]).reshape(-1, 1)
theta = np.mat([0, 2]).reshape(-1, 1)
# x = np.mat([1, 3, 1, 4, 1, 6, 1, 5, 1, 1, 1, 4, 1, 3, 1,
#            4, 1, 3.5, 1, 4.5, 1, 2, 1, 5]).reshape(12, 2)
# theta = np.mat([0, 2]).reshape(2, 1)
# y = np.mat([1, 2, 3, 2.5, 1, 2, 2.2, 3, 1.5, 3, 1, 3]).reshape(12, 1)

# 求代价函数值
j = costFunctionJ(x, y, theta)
print('代价值：', j)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(np.array(x[:, 1])[:, 0], np.array(y[:, 0])[
            :, 0], c='r', label='real data')  # 画梯度下降前的图像
plt.plot(np.array(x[:, 1])[:, 0], x*theta, label='test data')
plt.legend(loc='best')
plt.title('before')

theta, j_history, temp = gradientDescent(x, y, theta, 0.01, 1)
print('最终theta值：\n', theta)
print('每次迭代的代价值：\n', j_history)
print('theta值更新历史：\n', temp)

plt.subplot(1, 2, 2)
plt.scatter(np.array(x[:, 1])[:, 0], np.array(y[:, 0])[
            :, 0], c='r', label='real data')  # 画梯度下降后的图像
plt.plot(np.array(x[:, 1])[:, 0], x*theta, label='predict data')
plt.legend(loc='best')
plt.title('after')
plt.show()
