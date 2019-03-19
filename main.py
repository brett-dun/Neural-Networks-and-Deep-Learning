
import network
import mnist_loader_python3

training_data, validation_data, test_data = mnist_loader_python3.load_data_wrapper()

net = network.Network([784, 30, 10])
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
net.save()