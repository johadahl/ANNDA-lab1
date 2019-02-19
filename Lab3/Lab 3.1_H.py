import numpy as np


class hebbian:
	def __init__(self, size):
		self.weights = np.zeros((size,size))


	def updateWeights(self, x):
		self.weights += np.dot(x.T, x)

	def sign(self, x, iterations=1):
		guess = np.dot(self.weights, x.T)
		j = 0
		for i in guess:
			if i<0:
				guess[j] = -1
			else:
				guess[j] = 1
			j = j + 1
		if np.array_equal(guess.T, x):
			print("Right guess on iteration: " + str(iterations))
		else:
			iterations = iterations + 1
			self.sign(guess.T, iterations)
		return guess

if __name__ == '__main__':
	x1 = np.array([(-1,-1,1,-1,1,-1,-1,1)])
	x2 = np.array([(-1,-1,-1,-1,-1,1,-1,-1)])
	x3 = np.array([(-1,1,1,-1,-1,1,-1,-1)])

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
	guess = hebbian.sign(x1d)
	guess = hebbian.sign(x2d)
	guess = hebbian.sign(x3d)