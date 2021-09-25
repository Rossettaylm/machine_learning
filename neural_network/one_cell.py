# 只有一个S型神经元的神经网络，设定输入为1，要令输出为0，以此训练网络，并将训练过程中的损失函数储存

import matplotlib.pyplot as plt
import numpy as np


class Network(object):

    def __init__(self):
        # self.weight = np.random.randn()
        # self.bias = np.random.randn()
        self.weight, self.bias = 3.0, 3.0
        self.x = 1
        self.y = 0

    def feedforward(self, a):
        return sigmoid(self.weight*a+self.bias)

    def SGD(self, epochs, alpha):
        self.costs = np.zeros(epochs)
        for epoch in range(epochs):
            self.weight, self.bias = self.update_weight_bias(alpha)
            self.costs[epoch] = self.cost_crossEntropy()
            print('Epoch {}: output={}'.format(
                epoch, self.feedforward(self.x)))
        print('Epochs complete!')
        print('weight = {}, bias = {}'.format(self.weight, self.bias))

    def cost_function(self):
        return np.power((self.y - self.feedforward(self.x)), 2) / 2

    def cost_crossEntropy(self):
        a = self.feedforward(self.x)
        first = self.y * np.log(a)
        second = (1-self.y) * np.log(1-a)
        c = -(first + second)
        return c

    def update_weight_bias(self, alpha):
        a = self.feedforward(self.x)
        delta_w = self.x * (a - self.y)
        delta_b = a - self.y
        self.weight = self.weight - alpha * delta_w
        self.bias = self.bias - alpha*delta_b
        return self.weight, self.bias


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))


def main():
    nn = Network()
    epochs = 300
    nn.SGD(epochs, alpha=0.15)
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), nn.costs, c='b',
             linewidth=3, label='cost function')
    plt.legend()
    plt.title('costFunction with epochs')
    plt.show()


if __name__ == "__main__":
    main()
