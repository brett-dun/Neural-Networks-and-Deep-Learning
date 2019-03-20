
import numpy as np

# this needs to be rewritten to be compatible with the network
def load_data(path):

	arr = np.load(path)

	# returns (training_data, test_data)
	return (arr[:-1000], arr[-1000:])

def load_data_wrapper(path):

	a, b = load_data(path)

	training_data = [(aa[:625].reshape((625,1)).astype(np.float32)/255, aa[625:].reshape((4,1)).astype(float)) for aa in a]
	test_data = [(bb[:625].reshape((625,1)).astype(np.float32)/255, np.argmax(bb[625:])) for bb in b]

	#print(training_data[0])
	#print(test_data[0])

	return (training_data, test_data)