
import json
import numpy as np

MAX_WORD_LENGTH = 10


def vectorize(i):

	v = np.zeros((11,), dtype=np.uint8)
	v[i] = 1

	return v

def vectorize_word(word):

	v = np.zeros((MAX_WORD_LENGTH*26,), dtype=np.uint8)
	
	for i,c in enumerate(word.lower()):

		# print(i, c, i*26+(ord(c)-ord('a')))
		v[i*26+(ord(c)-ord('a'))] = 1

	return v


with open('words.json') as f:

	imported_data = json.load(f)

word_list = imported_data['words']

# x = np.zeros((len(word_list), MAX_WORD_LENGTH*26), dtype=np.float32)
# y = np.zeros((len(word_list), 11), dtype=np.float32)

data = np.zeros((len(word_list), MAX_WORD_LENGTH*26+11), dtype=np.uint8)

count = 0
for i, (word,lang) in enumerate(word_list):

	if len(word) > MAX_WORD_LENGTH:
		continue

	count += 1

	x = vectorize_word(word)
	y = vectorize(lang)

	data[i] = np.append(x, y)

np.save('word_data', data[:count])

