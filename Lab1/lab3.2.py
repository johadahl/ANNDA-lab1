import numpy as np
from math import exp
'''
Forward pass 	CHECK
Backpropagation	CHECK
Change Weights 	Work in progress
Generate data
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
		self.W1 = np.random.randn(inputSize+1, hiddenSize)

		# Weights between hidden layer and output layer
		self.W2 = np.random.randn(hiddenSize+1, outputSize)
	
	# X is the input vector in each iteration.
	# Input in the input layer and the calculated activation values in the hidden layer
	def fwdProp(self, row):
		bias = np.array([[1]])
		h_in = np.dot(self.W1.transpose(), row)
		h_out = np.concatenate((self.transFunc(h_in), bias), axis=0)
		self.h_out = h_out
		o_in = np.dot(self.W2.transpose(), h_out)
		o_out = self.transFunc(o_in)
		return o_out

	def bkwProp(self, target, output):
		# (2,1) matrix with delta errors in output
		delta_o = np.multiply(output - target, self.transDeriv(output))

		# (3,1) matrix with delta errors in hidden layer
		delta_h = np.multiply(np.dot(self.W2,delta_o), self.transDeriv(self.h_out))
		delta_h = delta_h[:self.hiddenSize,:] # Take away the last bias part, cause we don't need it
		return delta_h, delta_o

	def updateWeights(self, delta_h, delta_o, row, delta_w1, delta_w2):
		alpha = 0.9
		eta = 0.001

		delta_w1 = np.multiply(delta_w1, alpha) - np.multiply((1-alpha), np.dot(delta_h, row.T))

		delta_w2 = np.multiply(delta_w2, alpha) - np.multiply((1-alpha), np.dot(delta_o, self.h_out.T))

		self.W1 += eta*delta_w1
		self.W2 += eta*delta_w2
		'''

		CONTINUE HERE

		'''
		return

	def transFunc(self, x):
		return 2/(1+np.exp(-x))-1

	def transDeriv(self, x):
		return np.multiply((1+x),(1-x))/2


if __name__ == '__main__':
	neuralnet = neuralNet(2,3,1)
	row = np.matrix([[-1],[-1],[1]])
	target = np.matrix([[1]])
	output = neuralnet.fwdProp(row)
	delta_h, delta_o = neuralnet.bkwProp(target, output)
	neuralnet.updateWeights(delta_h, delta_o, row)