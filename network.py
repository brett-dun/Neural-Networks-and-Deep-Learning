
import numpy as np
import random

import json # not originally in the book

'''
NOTICE:
The majority of the code in this document came from http://neuralnetworksanddeeplearning.com/chap1.html
I have noted where I made my own personal additions.
'''

'''
TODO:
- understand backprop code
'''

class Network(object):

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		# biases for all of the neurons except the input layer
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		# weights between neurons where self.weights[j][k] is the weight between the jth neuron in the n layer and the kth neuron in the n-1 layer
		self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

	# a must be an (n, 1) array
	def feedforward(self, a):

		for b, w in zip(self.biases, self.weights):

			#calculate the output of the current layer
			a = sigmoid(np.dot(w, a)+b)

		return a

	'''
	training_data: list of tuples of training data
	epochs: integer number of epochs
	mini_batch_size: integer size of mini batches
	eta: learning rate
	test_data: data to evaluate the model on after each epoch [optional]
	'''
	def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):

		if test_data:
			n_test = len(test_data)

		n = len(training_data)

		for j in range(epochs):

			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

			if test_data:
				print('Epoch {0}: {1}/{2}'.format(j, self.evaluate(test_data), n_test))
			else:
				print('Epoch {0} complete.'.format(j))

			break

	def update_mini_batch(self, mini_batch, eta):

		# https://en.wikipedia.org/wiki/Del
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [(nb+dnb) for nb,dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [(nw+dnw) for nw,dnw in zip(nabla_w, delta_nabla_w)]

		self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]

		zs = []

		for b,w in zip(self.biases, self.weights):

			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		
		for l in range(2, self.num_layers):

			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-1+l].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)

	def evaluate(self, test_data):

		test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data]

		return sum(int(x == y) for x,y in test_results)

	def cost_derivative(self, output_activations, y):

		return (output_activations-y)


	# personal addition for saving the file to a json file
	def save(self):
		
		d = {}

		d['num_layers'] = self.num_layers
		d['sizes'] = self.sizes
		d['biases'] = [b.tolist() for b in self.biases]
		d['weights'] = [w.tolist() for w in self.weights]

		with open('network.json', 'w') as file:
			json.dump(d, file, indent=4)


# z is a np array that has the sigmoid function applied to it element-wise
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# can this function be rewritten to call sigmoid z only once? (how does numpy handle that?)
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))


# personal addition for loading the network from a json file
def load(path):

	with open('network.json') as file:
		data = json.load(file)

	sizes = data['sizes']
	biases = [np.asarray(b) for b in data['biases']]
	weights = [np.asarray(w) for w in data['weights']]

	net = Network(sizes)
	net.biases = biases
	net.weights = weights

	return net
