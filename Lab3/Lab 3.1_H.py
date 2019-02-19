import numpy as np


class hebbian:
	def __init__(self, size):
		self.weights = np.zeros((size,size))

	def updateWeights(self, x):
		print("An update")
		print(np.dot(x.T, x))
		self.weights += np.dot(x.T, x)

	def sign(self, x):
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
	x1d = np.array([(1,-1,1,-1,1,-1,-1,1)])
	x2d = np.array([(1,1,-1,-1,-1,1,-1,-1)])
	x3d = np.array([(1,1,1,-1,1,1,-1,1)])

	# initiate the hebbian class 
	hebbian = hebbian(x1d.size)

	# Update the weight matrix with all three inputs
	hebbian.updateWeights(x1d)
	hebbian.updateWeights(x2d)
	hebbian.updateWeights(x3d)

	# Check if the sign function can reproduce the right inputs 
	print("\nGuess")
	guess = hebbian.sign(x1d)
	print(guess.T)
	print("\nGuess")
	guess = hebbian.sign(x2d)
	print(guess.T)
	print("\nGuess")
	guess = hebbian.sign(x3d)
	print(guess.T)