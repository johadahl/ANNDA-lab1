import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

x1 = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype = "float")
x2 = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype = "float")
x3 = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype = "float")


x1d = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype = "float")
x2d = np.array([1, 1, 0, 0, 0, 1, 0, 0], dtype = "float")
x3d = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype = "float")


x1dist = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype = "float")
x2dist = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype = "float")
x3dist = np.array([1, 1, 0, 0, 0, 1, 1, 1], dtype = "float")

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

def energy(W, patterns):
	dim = patterns.size
	#print(dim)
	energy = 0
	for i in range(dim):
		for j in range(dim):
				energy += -1*patterns[i]*patterns[j]*W[i][j]
	return energy


def checkConvergence(W, pattern):
	numIterations = 0
	previousPattern = np.zeros(pattern.size)
	while True:
		res = updateRule(W, pattern)
		pattern = res
		en = energy(W, pattern)
		print("The energy is: ", en)
		if checkIfTrue(res):
			print("It took: ", numIterations, "number of iterations.")
			print("Pattern: ", pattern)
			break
		elif np.array_equal(pattern, previousPattern):
			print("------------------")
			print("Local minima found!")
			print("It took: ", numIterations, "number of iterations.")
			print("Pattern: ", pattern)
			break
		previousPattern = pattern
		numIterations += 1

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
			print("---------------")
			print("Matched with x", i+1, "!")
			return True
	return False


def sign(x):
	if x>=0:
		return 1
	else:
		return -1

def checkOldList(res, attractors):

	if(attractors.size)>0:
		contains = False
		for i in range(attractors.shape[0]):
			#go through list of all and see if it contains
			if (np.array_equal(res, attractors[i])):
				contains = True
		if not contains:
			#if it doesnt contain the attractor, add it to the list
			attractors = np.vstack((attractors, res))
	else:
		attractors = res
	return attractors

def checkAttractors(W):
	iterations = 0
	attractors = np.array([])
	oldRes = np.zeros(patterns.size)
	while iterations < 1000:
		pattern = np.array([1,1,1,1,-1,-1,-1,-1])
		np.random.shuffle(pattern)
		dim = pattern.size
		#print(iterations)
		it = 0
		while True:
			#print(it)
			res = updateRule(W, pattern)
			if np.array_equal(pattern, x1) or np.array_equal(pattern, x2) or np.array_equal(patterns, x3):
			#if the returned array equals the original pattern, then we have an attractor
				attractors = checkOldList(res, attractors)
				break
			elif np.array_equal(res, oldRes):
			#if the returned array equals the last array, then we have local minima
				attractors = checkOldList(res, attractors)
				break
			oldRes = res
			it +=1
		iterations += 1
	print("Number of attractors is: ")
	print(np.size(attractors,0))
	print(attractors)

if __name__ == '__main__':
	W = weightMatrix(patterns)
	#print("x2: ", x2)
	#returnedPattern = updateRule(W, x1)
	#returnedPattern2 = updateRule(W, x2)
	#returnedPattern3 = updateRule(W, x3)

	'''
	checkConvergence(W, x1)
	checkConvergence(W, x2)
	checkConvergence(W, x3)
	print("\n\n\n")

	"""checkConvergence(W, x1d)
	checkConvergence(W, x2d)
	checkConvergence(W, x3d)"""

	checkConvergence(W, x1dist)
	checkConvergence(W, x2dist)
	checkConvergence(W, x3dist)


	print("\n\n\n--------distorted patterns--------")
	print("Energy at x1d: ", energy(W, x1d))
	print("Energy at x2d: ", energy(W, x2d))
	print("Energy at x3d: ", energy(W, x3d))

	#checkAttractors(W)
	"""

	plt.plot(range(len(testError)), testError, label = "test error")
	plt.plot(range(len(testError)), trainError, label = "training error")
	plt.legend()
	plt.show()"""


	'''


