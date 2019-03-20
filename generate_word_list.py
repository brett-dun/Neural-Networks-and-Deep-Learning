
import os
import json
from random import shuffle

lookup = []
words = []

for i,filename in enumerate(os.listdir('words')):

	lookup.append((filename[:-4], i))
	
	with open('words/'+filename) as f:

		for line in f:

			word = line.split(',')[0]

			words.append((word, i))

shuffle(words)

data = {}
data['lookup'] = lookup
data['words'] = words

with open('words.json', 'w') as f:
	json.dump(data, f, indent=4)