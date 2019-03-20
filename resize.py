
import os
from PIL import Image

folders = ['circle', 'square', 'star', 'triangle']
directory = 'shapes'

try:
	os.mkdir(directory+'_small')
except OSError:
	print('Error')

for folder in folders:

	path = directory+'_small/'+folder+'_small'

	try:
		os.mkdir(path)
	except OSError:
		print('Error')

	for filename in os.listdir(directory+'/'+folder):

		with Image.open(directory+'/'+folder+'/'+filename) as image:

			img = image.resize((25, 25))
			img.save(path+'/'+lfilename[:-4]+'_small.png')