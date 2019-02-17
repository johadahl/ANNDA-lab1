import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy import signal
np.random.seed(42)


trainX = np.arange(0, 2*np.pi, 0.1)
testX = np.arange(0.05, 2*np.pi, 0.1)
trainTSin = np.sin(2*trainX)
trainTSin += np.random.normal(0, math.sqrt(0.1), trainTSin.shape[0])

testTSin = np.sin(2*testX)
testTSin += np.random.normal(0, math.sqrt(0.1), testTSin.shape[0])

#plotSin = np.stack((trainTSin, testTSin), axis = 1).flatten()
#plotX = np.stack((trainX, testX), axis = 1).flatten()
#plt.plot(plotX, plotSin)
#plt.show()

class LeastSquares:
	def __init__(self, train, trainTarget, nodesEV):
		self.train = train
		self.trainTarget = trainTarget
		self.nodesEV = nodesEV
		self.var = np.ones(self.nodesEV.size)
		self.phi = self.createPhi()
		self.weight = self.trainWeights()


	def createPhi(self):
		phiMat = np.zeros((self.train.size, self.nodesEV.size))
		c = 0
		for x in self.train:
			phiMat[:][c] = self.activation(x)
			c+=1
		return phiMat

	def activation(self, x):
		return np.exp(np.multiply((-(x-self.nodesEV)**2),((2*self.var)**(-1))))

	def trainWeights(self):
		invPhiPhi = np.linalg.inv(np.dot(np.transpose(self.phi), self.phi))
		phiF = np.dot(np.transpose(self.phi), self.trainTarget)
		return np.dot(invPhiPhi, phiF)

	def testMethod(self, test):
		y = np.dot(self.phi, self.weight)
		return y


	def run(self, input):
		result = np.zeros(input.size)
		c = 0
		for x in input:
			result[c] = sum(np.multiply(self.weight, self.activation(x)))
			c +=1
		return result


	def calcError(self, input, target):
		output = self.run(input)
		diff = np.absolute(output-target)
		return np.average(diff)

	def calcTransformError(self, input, target):
		output = self.run(input)
		output[output >= 0] = 1
		output[output <= 0 ] = -1
		diff = np.absolute(output - target)
		return np.average(diff)

class DeltaRule:

	def __init__(self, train, trainTarget, nodesEV, step = 0.01, maxEpoch = 200):
		self.train = train
		self.trainTarget = trainTarget
		self.nodesEV = nodesEV
		self.nodes = nodesEV.size
		self.weight = np.ones(self.nodesEV.size)
		self.step = step
		self.maxEpoch = maxEpoch
		self.var = np.ones(self.nodesEV.size) #variance
		self.epoch = 0
		self.errTreshold = 0.1
		self.mainLoop()

	def mainLoop(self):
		while self.epoch < self.maxEpoch:
			for i in range(self.train.size):
				self.weight += self.deltaWeights(i)
				if self.errorCalc(self.train, self.trainTarget) < self.errTreshold:
					return
			self.epoch += 1


	def deltaWeights(self, index):
		e = self.trainTarget[index] - np.dot(self.activation(self.train[index]), self.weight)
		deltaWeight = e*self.step*self.activation(self.train[index])
		return deltaWeight

	def activation(self, x):
		return np.exp(np.multiply((-(x-self.nodesEV)**2),((2*self.var)**(-1))))

	def run(self, input):
		result = np.zeros(input.size)
		c = 0
		for x in input:
			result[c] = sum(np.multiply(self.weight, self.activation(x)))
			c +=1
		return result

	def errorCalc(self, input, target):
		result = self.run(input)
		diff = np.absolute(result-target)
		return np.average(diff)



if __name__ == '__main__':
	index = np.linspace(0, trainX.size-1, num = 20, dtype = int)
	b = DeltaRule(trainX, trainTSin, trainX[index], 0.01, 100)

	plt.plot(trainX, b.run(trainX), 'green')
	plt.plot(trainX, trainTSin, 'red')
	plt.plot(trainX[index], trainTSin[index], 'b+')
	plt.title("Training data")
	plt.show()

	print("Residual error is: ", b.errorCalc(trainX, trainTSin))
	print("Nodes \t Training error \t Testing error") 
	for i in range(1, trainX.size +1):
		index = np.linspace(0, trainX.size-1, num = i, dtype = int)
		dr = DeltaRule(trainX, trainTSin, trainX[index], 0.01, 100)
		print(i, " \t ", dr.errorCalc(trainX, trainTSin), " \t ", dr.errorCalc(testX, testTSin))



