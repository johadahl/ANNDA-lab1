import numpy as np


class hebbian:
	def __init__(self, size):
		self.weights = np.zeros((size,size))

	def updateWeights(self, x):
		print("An update")
		print(np.dot(x.T, x))
		self.weights += np.dot(x.T, x)

	def recall(self, x):
		guess = np.dot(self.weights, x.T)
		j = 0
		for i in guess:
			if i<0:
				guess[j] = -1
			else:
				guess[j] = 1
			j = j + 1
		return guess


if __name__ == '__main__':
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, - 1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

    x1d = np.array([(1,-1,1,-1,1,-1,-1,1)])
    x2d = np.array([(1,1,-1,-1,-1,1,-1,-1)])
    x3d = np.array([(1,1,1,-1,1,1,-1,1)])

	# initiate the hebbian class
	hebbian = hebbian(x1.size)

	# Update the weight matrix with all three inputs
	hebbian.updateWeights(x1)
	hebbian.updateWeights(x2)
	hebbian.updateWeights(x3)

	# Check if the sign function can reproduce the right inputs
	print("\nGuess")
	guess = hebbian.recall(x1d)
	print(guess.T)
	print("\nGuess")
	guess = hebbian.recall(x2d)
	print(guess.T)
	print("\nGuess")
	guess = hebbian.recall(x3d)
	print(guess.T)