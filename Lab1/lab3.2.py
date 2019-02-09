import numpy as np
from math import exp
'''
Forward pass 	CHECK
Backpropagation	CHECK
Change Weights 	CHECK
Generate data 	WIP
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
		self.delta_W1 = np.zeros((inputSize+1, hiddenSize))

		# Weights between hidden layer and output layer
		self.W2 = np.random.randn(hiddenSize+1, outputSize)
		self.delta_W2 = np.zeros((hiddenSize+1, outputSize))
	
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
		delta_h = np.multiply(np.dot(self.W2,delta_o), self.transDeriv(self.h_out))[:self.hiddenSize,:] # Snap of the last index of vector since we wont need it in future (it's the bias term)
		return delta_h, delta_o

	def updateWeights(self, delta_h, delta_o, row):
		# Introduce momentu (alpha) taking previous changes in to account as well to suppress fast variations and allow larger learning rate (eta)
		alpha = 0.9
		eta = 0.01	

		# Store the delta_w as a parameter so it is easy accessible for the next round
		self.delta_W1 = np.multiply(self.delta_W1, alpha) - np.multiply((1-alpha), np.dot(delta_h, row.T))
		self.delta_W2 = np.multiply(self.delta_W2, alpha) - np.multiply((1-alpha), np.dot(delta_o, self.h_out.T)).T

		# Uppdate the weights
		self.W1 += self.delta_W1*eta
		self.W2 += self.delta_W2*eta

	def transFunc(self, x):
		return 2/(1+np.exp(-x))-1

	def transDeriv(self, x):
		return np.multiply((1+x),(1-x))/2


def generate_data():
	N = 200 # nr of inputs per class
	D = 2 # Dimensionality
	K = 2 # nr of classes
	X = np.zeros((D,N*K)) # Empty input matrix with each column being an input
	for j in range(2):
		ix = range(N*j,N*(j+1))
		r = np.linspace(0.0,1,N)
		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	print(X)

'''

CONTINUE HERE 


'''



if __name__ == '__main__':
	generate_data()

	neuralnet = neuralNet(2,3,1)
	row = np.matrix([[-1],[-1],[1]])
	target = np.matrix([[1]])
	output = neuralnet.fwdProp(row)
	delta_h, delta_o = neuralnet.bkwProp(target, output)
	neuralnet.updateWeights(delta_h, delta_o, row)