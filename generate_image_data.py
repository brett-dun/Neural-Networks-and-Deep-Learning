
import os
from PIL import Image
import numpy as np

'''
note: it may be better to randomize the images in the future so that this function will always return the same data
'''

folders = ['circle_small', 'square_small', 'star_small', 'triangle_small']
directory = 'shapes_small'

data = np.zeros((3765*4, 25*25+4), dtype=np.uint8)

count = 0
for i,folder in enumerate(folders):

	y = np.zeros(4, dtype=np.uint8)
	y[i] = 1

	for filename in os.listdir(directory+'/'+folder):

		with Image.open(directory+'/'+folder+'/'+filename) as img:

			x = np.array(img.getdata(), dtype=np.uint8)

		data[count] = np.append(x, y)
		count += 1

np.random.shuffle(data)

np.save('shape_data', data)

