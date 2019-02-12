import numpy as np
from math import exp
from sklearn.datasets import make_moons
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame

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
	def fwdProp(self, row, target):
		h_in = np.dot(self.W1, row)
		h_out = np.concatenate((self.transFunc(h_in), np.array([[1]])), axis=0) # Add another row to be able to calculate bias in next layer of neurons
		o_in = np.dot(self.W2, h_out)
		o_out = self.transFunc(o_in)

		# Calculate the square error in this particular guess and appends it to a list of length N
		error = o_out - target
		self.calcError(error)

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
		eta = 0.5	

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
		

def plotData(X,t):
	# scatter plot, dots colored by class value
	df = DataFrame(dict(x=X[:,0], y=X[:,1], label=t))
	colors = {0:'red', 1:'blue'}
	fig, ax = pyplot.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
	    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	pyplot.show()

def generate_data(N):
	# generate 2D classification dataset
	X, t = make_moons(n_samples=N, noise=0.2)

	# Add the bias term to the input values (A row of 1's at the bottom)
	bias = np.ones((1,N))
	X = np.concatenate((X.T,bias), axis=0)

	# Make a (2,N) target matrix (since we have 2 differnt classes)
	T = targetMatrix(t)

	#plotData(X, t)
	return X, T

def targetMatrix(inT):
	# Initialize an empty (1,N) array
	T_lower = np.zeros(inT.size).astype(int)
	for i in range(inT.size):
		if inT[i] == 0:
			T_lower[i] = 1
	return np.column_stack((inT,T_lower)).T

def predict(row, target):
	h_out, o_out = neuralnet.fwdProp(row, target)
	if np.argmax(o_out.T) == 0:
		return 0
	elif np.argmax(o_out.T) == 1:
		return 1
	elif np.argmax(o_out.T) == 2:
		return 2
	elif np.argmax(o_out.T) == 3:
		return 3
	elif np.argmax(o_out.T) == 4:
		return 4
	elif np.argmax(o_out.T) == 5:
		return 5
	elif np.argmax(o_out.T) == 6:
		return 6
	else:
		return 7

def trainNetwork(epochs, X, T):
	for epoch in range(epochs):
		# Re-initialized the delta_weights for each epoch
		neuralnet.delta_W1 = 0
		neuralnet.delta_W2 = 0
		j=0
		for i in X.T:
			# Store the input row in a (3,1) matrix
			row = np.matrix([[i[0]],[i[1]], [i[2]], [i[3]], [i[4]], [i[5]], [i[6]], [i[7]], [i[8]]])
			# Store the target value in a (1,1) matrix
			target = np.matrix([[T.T[j][0]], [T.T[j][1]], [T.T[j][2]], [T.T[j][3]], [T.T[j][4]], [T.T[j][5]], [T.T[j][6]], [T.T[j][7]]])
			h_out, o_out = neuralnet.fwdProp(row, target)
			delta_h, delta_o = neuralnet.bkwProp(target, h_out, o_out)
			neuralnet.updateWeights(delta_h, delta_o, row, h_out)
			j = j+1
	# Makes an numpy array of all the square errors over the training 

def splitSet(X, T, testratio):
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

def testNetwork(X_test, T_test):
	j=0
	right = 0
	for i in X_test.T:
		row = np.matrix([[i[0]],[i[1]], [i[2]], [i[3]], [i[4]], [i[5]], [i[6]], [i[7]], [i[8]]])
		prediction = predict(row, T_test)
		print("Prediction: " + str(prediction) + " Real Value: " + str(np.argmax(T_test.T[j])))
		'''
		print("\nPREDICTION")
		print(prediction)
		print("TARGETS")
		print(np.argmax(T_test.T[j]))
		'''
		if prediction == np.argmax(T_test.T[j]):
			right = right + 1
		j = j+1
	#Store all the square errors for each iteration. For exampl 20*300 + 100 with variables:
	# N = 400
	# epocs = 20
	# testratio = 0.25 (therefore 100 testing points)
	neuralnet.squareErrorArray = np.asarray(neuralnet.squareError)
	
	'''
	PAULERI KIKA PRECIS NEDAN. FATTAR FAN INGENTING

	HJÄLP 

	
	x_value = np.linspace(1, np.size(neuralnet.squareErrorArray), np.size(neuralnet.squareErrorArray))
	y_value = neuralnet.squareErrorArray
	pyplot.scatter(x_value, y_value)
	pyplot.show()
	'''
	print("\nSize of hidden layer:\t" + str(neuralnet.hiddenSize))
	rate = right/8
	print("Predicted right:\t" + str(rate*100) + "%")


if __name__ == '__main__':
	# Variables 
	hiddenLayerSize = 2
	outputLayerSize = 8
	epochs = 40
	testratio = 0.25 # Amount of testing data

	# Generate random data points (linearly inseparable)
	#X, T = generate_data(N)

	# Make another generate_data(N) here for the Encoder problem

	# Return the amount of training- and test data with respect to the test ratio
	#X_train, X_test, T_train, T_test = splitSet(X,T, testratio)

	X = np.array([[1,-1,-1,-1,-1,-1,-1,-1], [-1,1,-1,-1,-1,-1,-1,-1], [-1,-1,1,-1,-1,-1,-1,-1], [-1,-1,-1,1,-1,-1,-1,-1], [-1,-1,-1,-1,1,-1,-1,-1], [-1,-1,-1,-1,-1,1,-1,-1], [-1,-1,-1,-1,-1,-1,1,-1], [-1,-1,-1,-1,-1,-1,-1,1]])
	T = X
	bias = np.ones((1,8))
	X = np.concatenate((X.T,bias), axis=0)

	# Run different test runs over the same input and target data
	
	# Initiate the neural network object
	neuralnet = neuralNet(8, hiddenLayerSize, outputLayerSize)

	# Train the network with input matrix and corresponding target values
	# Includes forwardprop, backprop and weight chainging
	trainNetwork(epochs, X, T)

	# Test the network with the testing data
	testNetwork(X, T)