import numpy as np


class Network(object):
    def __init__(self, size):
        self.num_layers = len(size)
        self.size = size
        # 初始化权重和偏置
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        # w = sn+1 @ sn  ->  y @ x
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(size[:-1], size[1:])]

    def feedforward(self, a):
        """Return the output of the network if 'a' is input.
            a' = w@a + b
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.

        Args:
            training_data ([list][tuple]): [A list of tuples "(x, y)" representing the training inputs and the desired outputs.]
            epochs ([int]): [The size of iteration period]
            mini_batch_size ([int]): [The size of each batch]
            eta ([float]): [Learning rate]
            test_data ([list][tuple], optional): [If 'test_data' is provided the the network will be evaluated against the test data after each epoch, and partial progress printed out.]. Defaults to None.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 打乱数据集的顺序，进行随机采样
            np.random.shuffle(training_data)
            # 得到每一批次的minibatch数据，形成一个列表
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            # 对一个batch内的数据进行梯度下降
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,
                      self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Updating the network's weights and biases by applying gradient descent using Backpropagation to a single mini batch.

        Args:
            mini_batch ([list][tuple]): [A list of tuples of "(x, y)"]
            eta ([float]): [Learning rate]
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) *
                        nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) *
                       nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """反向传播算法
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activation[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """[summary]

        """
        # 利用训练好的w和b来验证数据
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """[summary]

        """
        return (output_activations - y)

# Miscellaneous functions


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function.

    """
    return sigmoid(z) * (1-sigmoid(z))


# nn = Network(size=[28*28, 16, 10])
# print(nn.size)
# print(nn.biases[0].shape, nn.biases[1].shape)
# print(nn.weights[0].shape, nn.weights[1].shape)
