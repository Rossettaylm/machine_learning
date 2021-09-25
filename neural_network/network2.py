import numpy as np


class Network(object):
    def __init__(self, size) -> None:
        self.size = size
        self.num_layers = len(size)
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(size[1:], size[:-1])]

    def feedforward(self, a):
        """输入激活值得到输出

        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, alpha, test_data=None):
        """对训练集数据进行随机梯度下降训练，得到合适的权重和偏执

        Args:
            training_data ([type]): [description]
            epochs ([type]): [description]
            mini_batch_size ([type]): [description]
            alpha ([type]): [description]
            test_data ([type], optional): [description]. Defaults to None.
        """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j,
                      self.evaluate(test_data), n_test))
            else:
                print('Epoch: {0} complete!'.format(j))

    def update_mini_batch(self, mini_batch, alpha):
        """对选取的小批量数据进行随机梯度下降

        Args:
            mini_batch ([[list][tuple]]): [小批量输入数据]
            alpha ([float]): [学习率]
        """
        m = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (alpha/m)*nb for b,
                       nb in zip(self.biases, nabla_b)]
        self.weights = [w - (alpha/m)*nw for w,
                        nw, in zip(self.weights, nabla_w)]

    def backpropagation(self, x, y):
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].T)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(-2, -self.num_layers, -1):
            z = zs[l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[l+1].T, delta) * sp
            delta_nabla_b[l] = delta
            delta_nabla_w[l] = np.dot(delta, activations[l-1].T)
        return delta_nabla_b, delta_nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_a, y):
        return (output_a - y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
