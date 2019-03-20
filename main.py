
import numpy as np

import network
import mnist_loader_python3
import load_image_data
import load_word_data


# code for training network on MINST data

'''
training_data, validation_data, test_data = mnist_loader_python3.load_data_wrapper()

net = network.Network([784, 30, 10])
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
net.save()
'''


# code for training network on shape data
# the data used for this is from https://www.kaggle.com/smeschke/four-shapes/home#process_data.py
'''
training_data, test_data = load_image_data.load_data_wrapper('shape_data.npy')

net = network.Network([625, 30, 4])
net.sgd(training_data, 15, 10, 1.0, test_data)
net.save('shape_detection_network.json')
'''


# code for training network on language data
training_data, test_data = load_word_data.load_data_wrapper('word_data.npy')

net = network.Network([10*26, 30, 11])
net.sgd(training_data, 8, 10, 3.0, test_data)
net.save('language_detection_network.json')