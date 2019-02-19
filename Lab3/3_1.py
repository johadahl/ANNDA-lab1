import numpy as np


class hebbian:
	def __init__(self, size):
		self.weights = np.zeros((size,size))
		self.patterns = []	# Array with stored patterns

	def update_weights(self, x):
		print("An update")
		self.patterns.append(x)
		self.weights = np.add(self.weights, np.outer(x.T, x))
#		np.fill_diagonal(self.weights, 0)		# Reset diagonals to 0

	def check_True(self, prediction):
		solved = False
		for i in range(len(self.patterns)):
			if np.array_equal(self.patterns[i], prediction):
				print("Pattern matches stored pattern nr:", i + 1)
				solved = True
		return solved

	def update_rule(self, x):
		dim = x.size
		count = 0

		previousPattern = np.zeros(dim)
		old = []
		cont = True
		while cont:
			out = np.zeros(dim)
			for i in range(dim):
				s = 0
				for j in range(dim):
					w = self.weights[i][j]
					x_j = x[j]
					s += w*x_j
				if s >= 0:
					sign = 1
				else:
					sign = -1
				out[i] = sign
			x = out		# New updated pattern
			old.insert(0, previousPattern)

			if self.check_True(x):
				print("It took:", count, "nr of iterations")
				cont = False

			elif np.array_equal(x, previousPattern):
				print("Local minimum found in iteration:", count)
				print(x)
				cont = False
			# If stuck in an endless oscillating loop
			elif count > 100:
				for i in range(3):
					if np.array_equal(x, old[i]):
						print("Updated pattern %s similar to old %s after %s" % (x, old[i], count))
						cont = False
			previousPattern = x
			count += 1

def main():
	x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
	x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
	x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

	x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
	x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
	x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])

	# initiate the hebbian class
	h = hebbian(x1.size)

	# Update the weight matrix with all three inputs
	h.update_weights(x1)
	h.update_weights(x2)
	h.update_weights(x3)
	print(h.weights)

	# Check if the sign function can reproduce the right inputs
	print("\nGuess")
	h.update_rule(x1d)
	print("\nGuess")
	h.update_rule(x2d)
	print("\nGuess")
	h.update_rule(x3d)

	

if __name__ == '__main__':
	main()
