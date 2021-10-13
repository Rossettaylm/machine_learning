import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
nn = network.Network([784, 30, 10])
nn.sgd(training_data, 30, 10, 3.0, test_data)
