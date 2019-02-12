import numpy as np
from math import exp
from sklearn.datasets import make_moons
from matplotlib import pyplot, cm
import pandas as pd
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
'''
Forward pass 	CHECK
Backpropagation	CHECK
Change Weights 	CHECK
Generate data 	CHECK
Algorithm to test results

See comment bellow where to continue
'''

class neuralNet:

	# Parameters for the Neural Network
	def __init__(self, inputSize, hiddenSize, outputSize):
		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		self.outputSize = outputSize

		# Weights between input and hidden layer
		self.W1 = np.random.randn(hiddenSize, inputSize+1)

		# Weights between hidden layer and output layer
		self.W2 = np.random.randn(outputSize, hiddenSize+1)

		# An array to store all the mean square errors
		self.squareError = []
		self.squareErrorArray = 0
	
	# X is the input vector in each iteration.
	# Input in the input layer and the calculated activation values in the hidden layer
	def fwdProp(self, row):
		h_in = np.dot(self.W1, row)
		h_out = np.concatenate((self.transFunc(h_in), np.array([[1]])), axis=0)
		o_in = np.dot(self.W2, h_out)
		o_out = self.transFunc(o_in)

		return h_out, o_out

	def bkwProp(self, target, h_out, o_out):
		# (2,1) matrix with delta errors in output
		error = o_out - target
		delta_o = np.multiply(error, self.transDeriv(o_out))
		
		# (3,1) matrix with delta errors in hidden layer
		delta_h = np.multiply(np.dot(self.W2.T,delta_o), self.transDeriv(h_out))[:self.hiddenSize,:] # Snap of the last index of vector since we wont need it in future (it's the bias term)
	
		return delta_h, delta_o

	def updateWeights(self, delta_h, delta_o, row, h_out):
		# Introduce momentu (alpha) taking previous changes in to account as well to suppress fast variations and allow larger learning rate (eta)
		alpha = 0.9
		eta = 0.001

		# Store the delta_w as a parameter so it is easy accessible for the next round
		self.delta_W1 = np.multiply(self.delta_W1, alpha) - np.multiply((1-alpha), np.dot(delta_h, row.T))
		self.delta_W2 = np.multiply(self.delta_W2, alpha) - np.multiply((1-alpha), np.dot(delta_o, h_out.T))

		# Uppdate the weights
		self.W1 += self.delta_W1*eta
		self.W2 += self.delta_W2*eta

	def transFunc(self, x):
		return 2/(1+np.exp(-x))-1

	def transDeriv(self, x):
		return np.multiply((1+x),(1-x))/2

	def calcError(self, errors):
		sqrError = errors.item(0)**2 + errors.item(1)**2 
		self.squareError.append(sqrError)
		
def generate_data(N):
	# Generate data in X and Y axis
	X = np.linspace(-5, 5, N)
	Y = np.linspace(-5, 5, N)

	XX, YY = np.meshgrid(X, Y)
	Z = np.exp(-XX*XX*0.1) * np.exp(-YY*YY*0.1) - 0.5

	return XX, YY, Z

def trainNetwork(epochs, X, T, Z):
	for epoch in range(epochs):
		# Re-initialized the delta_weights for each epoch
		neuralnet.delta_W1 = 0
		neuralnet.delta_W2 = 0
		
		# for each element in the grid
		for i in range(np.size(X,1)): # For rows
			for j in range(np.size(X,0)): # For columns
				row = np.matrix([[X[i][j]], [Y[i][j]], [1]]) # add 1 for bias
				target = np.matrix([[Z[i][j]]]) # Fetch the corresponding target value from Z-grid
				h_out, o_out = neuralnet.fwdProp(row)
				delta_h, delta_o = neuralnet.bkwProp(target, h_out, o_out)
				neuralnet.updateWeights(delta_h, delta_o, row, h_out)
	# Makes an numpy array of all the square errors over the training 

def splitSet(X, T, Z, testratio):
	# Nr of test samples given the ratio
	nrOfTestSamples = int(np.size(X,1)*testratio)

	#Split the input values
	j = 0
	test = []
	train = []
	for i in X.T:
		if j < nrOfTestSamples:
			test.append(i)
		else:
			train.append(i)
		j = j + 1
	X_test = np.asarray(test)
	X_train = np.asarray(train)

	# Splitt the target values
	j = 0
	test = []
	train = []
	for i in T.T:
		if j < nrOfTestSamples:
			test.append(i)
		else:
			train.append(i)
		j = j + 1
	T_test = np.asarray(test)
	T_train = np.asarray(train)
	return X_train.T, X_test.T, T_train.T, T_test.T

def testNetwork(X, Y):
	# Have the network predicting the values for Z based on inputs from X, Y
	# for each element in the grid
	for i in range(np.size(X,1)): # For rows
		for j in range(np.size(X,0)): # For columns
			row = np.matrix([[X[i][j]], [Y[i][j]], [1]]) # add 1 for bias
			h_out, o_out = neuralnet.fwdProp(row)
			Z[i][j] = o_out

	return Z
			# o_out is the Z value which the network predicts it to be


if __name__ == '__main__':
	# Variables 
	outputLayerSize = 1
	hiddenLayerSize = 15
	N = 20 # Bredth and width of grid
	epochs = 100
	testratio = 0.25 # Amount of testing data

	# Generate random data points (linearly inseparable)
	X, Y, Z = generate_data(N)

	neuralnet = neuralNet(2, hiddenLayerSize, outputLayerSize)

	# Train the network with input matrix and corresponding target values
	# Includes forwardprop, backprop and weight chainging
	trainNetwork(epochs, X, Y, Z)

	Z = testNetwork(X, Y)
	
	fig = pyplot.figure()
	ax = fig.gca(projection='3d')

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	# Customize the z axis.
	ax.set_zlim(-0.5, 0.5)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


		# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	pyplot.show()
'''	
	
	# Return the amount of training- and test data with respect to the test ratio
	X_train, X_test, T_train, T_test = splitSet(X,T, testratio)

	# Run different test runs over the same input and target data
	nrOfHiddenNeurons = [1,2,4,8,16,32,64,128]
	for hiddenLayerSize in nrOfHiddenNeurons:	
		# Initiate the neural network object
		neuralnet = neuralNet(2,hiddenLayerSize,outputLayerSize)

		# Train the network with input matrix and corresponding target values
		# Includes forwardprop, backprop and weight chainging
		trainNetwork(epochs, X_train, T_train)

		# Test the network with the testing data
		testNetwork(X_test, T_test)
	'''