
import sys
from PIL import Image
import numpy as np

import network


net = network.load('shape_detection_network.json')

with Image.open(sys.argv[1]) as image:
	data = np.array(image.getdata(), dtype=np.float32).reshape((625,1)) / 255.

lookup = ['circle', 'square', 'star', 'triangle']

z = net.feedforward(data)
print(z)
print(lookup[np.argmax(z)])