import numpy as np
import random, math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
np.random.seed(42)

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

patterns = np.array([x1, x2, x3])

def weightMatrix(patterns):
	dim = patterns.shape[1] #dimension of pattern (8 in this case)
	W = np.zeros((dim,dim))
	for i in range(dim):
		for j in range(dim):
			s = 0
			for p in patterns: #one pattern at the time, out of three patterns
				s += p[i]*p[j]
			W[i][j] = s
	return W

def checkConvergence():
	while True:

	updateRule()

def updateRule(W, pattern):
	dim = pattern.size
	res = np.zeros(dim)
	for i in range(dim):
		s = 0
		for j in range(dim):
			s += W[i][j]*pattern[j]
		res[i] = sign(s)
	return res


def checkIfTrue(pattern):
	for i in range(patterns.shape[0]):
		if np.array_equal(pattern, patterns[i]):
			print("Match!")
			print("Matched with x", i+1, "!")
			return True
	return False


def sign(x):
	if x>=0:
		return 1
	else:
		return -1

if __name__ == '__main__':
	W = weightMatrix(patterns)

	returnedPattern = updateRule(W, x1)
	returnedPattern2 = updateRule(W, x2)
	returnedPattern3 = updateRule(W, x3)
	checkIfTrue(returnedPattern3)
	checkIfTrue(returnedPattern2)
	checkIfTrue(returnedPattern)
	#checkIfTrue(patterns)


	"""

	plt.plot(range(len(testError)), testError, label = "test error")
	plt.plot(range(len(testError)), trainError, label = "training error")
	plt.legend()
	plt.show()"""

