
import numpy as np

MAX_WORD_LENGTH = 10

# this needs to be rewritten to be compatible with the network
def load_data(path):

	arr = np.load(path)

	# returns (training_data, test_data)
	return (arr[:-10000], arr[-10000:])

def load_data_wrapper(path):

	a, b = load_data(path)

	training_data = [(aa[:MAX_WORD_LENGTH*26].reshape((MAX_WORD_LENGTH*26,1)).astype(np.float32), aa[MAX_WORD_LENGTH*26:].reshape((11,1)).astype(float)) for aa in a]
	test_data = [(bb[:MAX_WORD_LENGTH*26].reshape((MAX_WORD_LENGTH*26,1)).astype(np.float32), np.argmax(bb[MAX_WORD_LENGTH*26:])) for bb in b]

	#print(training_data[0])
	#print(test_data[0])

	return (training_data, test_data)