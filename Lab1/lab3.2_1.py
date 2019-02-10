import numpy as np
from math import exp
from sklearn.datasets import make_moons
from matplotlib import pyplot
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
		self.delta_W1 = np.zeros((hiddenSize, inputSize+1))

		# Weights between hidden layer and output layer
		self.W2 = np.random.randn(outputSize, hiddenSize+1)
		self.delta_W2 = np.zeros((outputSize, hiddenSize+1))
	
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
		delta_o = np.multiply(o_out - target, self.transDeriv(o_out))

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


def generate_data(N):
	# generate 2d classification dataset
	X, t = make_moons(n_samples=N, noise=0.1)
	'''
	# scatter plot, dots colored by class value
	df = DataFrame(dict(x=X[:,0], y=X[:,1], label=t))
	colors = {0:'red', 1:'blue'}
	fig, ax = pyplot.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
	    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	pyplot.show()
	'''
	return X.T, t

def targetMatrix(inT):
	# Initialize an empty (1,N) array
	T_lower = np.zeros(T.size).astype(int)
	for i in range(T.size):
		if T[i] == 0:
			T_lower[i] = 1
	return np.column_stack((inT,T_lower)).T

def predict(row):
	h_out, o_out = neuralnet.fwdProp(row)
	print(o_out)
	if np.argmax(o_out.T) == 0:
		return 1
	else:
		return 0
	

if __name__ == '__main__':
	# Variables 
	hiddelLayerSize = 3
	outputLayerSize = 2
	N = 400 # nr of samples (200 each)
	epochs = 20

	# Generate random data points (linearly inseparable)
	X, T = generate_data(N)

	# Add the bias term to the input values (A row of 1's at the bottom)
	bias = np.ones((1,N))
	X = np.concatenate((X,bias), axis=0)

	# Make a (2,N) target matrix (since we have 2 differnt classes)
	T = targetMatrix(T)

	# Initiate the neural network object
	neuralnet = neuralNet(2,hiddelLayerSize,outputLayerSize)

	# Iterate over the input vectors in input matrix X
	for epoch in range(epochs):
		j=0
		for i in X.T:
			# Store the input row in a (3,1) matrix
			row = np.matrix([[i[0]], [i[1]], [i[2]]])
			# Store the target value in a (1,1) matrix
			target = np.matrix([[T.T[j][0]], [T.T[j][1]]])
			h_out, o_out = neuralnet.fwdProp(row)
			delta_h, delta_o = neuralnet.bkwProp(target, h_out, o_out)
			neuralnet.updateWeights(delta_h, delta_o, row, h_out)
			j = j+1
		#neuralnet.delta_W1 = np.zeros((hiddelLayerSize, 2+1))
		#neuralnet.delta_W2 = np.zeros((outputLayerSize, hiddelLayerSize+1))
		print("Done with epoch nr: " + str(epoch))
		
	j=0
	right = 0
	for i in X.T:
		row = np.matrix([[i[0]], [i[1]], [i[2]]])
		prediction = predict(row)
		print(T.T[j])
		#print("Prediction: " + str(prediction) + " Real Value: " + str(T.T[j][1]))
		if j>300:
			if prediction == T.T[j][0]:
				right = right + 1
		j = j+1
	rate = right/(100)
	print("Guessed right: " + str(rate*100) + "%")
		
	#row = np.matrix([[-1],[-1],[1]])
	'''
	target = np.matrix([[1]])
	output = neuralnet.fwdProp(row)
	delta_h, delta_o = neuralnet.bkwProp(target, output)
	neuralnet.updateWeights(delta_h, delta_o, row)
	'''