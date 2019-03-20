
import json
import numpy as np

import network

MAX_WORD_LENGTH = 10


# this is not exactly the same because it returns an array of shape (n,1)
def vectorize_word2(word):

	v = np.zeros((MAX_WORD_LENGTH*26, 1), dtype=np.uint8)
	
	for i,c in enumerate(word.lower()):

		# print(i, c, i*26+(ord(c)-ord('a')))
		v[i*26+(ord(c)-ord('a'))] = 1

	return v


net = network.load('language_detection_network.json')

words = input('enter a string of words without special characters: ').split()

with open('words.json') as f:
	lookup = json.load(f)['lookup']


count = np.zeros(11)
for word in words:

	if len(word) > MAX_WORD_LENGTH:
		print(word, 'is too long.')
		continue

	z = net.feedforward(vectorize_word2(word))

	count[np.argmax(z)] += 1

	print(word)
	print(z)
	print(lookup[np.argmax(z)])

print()
print('The text you entered is probably in', lookup[np.argmax(count)][0])
