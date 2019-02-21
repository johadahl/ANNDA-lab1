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

	def __init__(self, train, trainTarget, nodesEV, step = 0.01, maxEpoch = 100):
		self.train = train
		self.trainTarget = trainTarget
		self.nodesEV = nodesEV
		self.nodes = nodesEV.size
		self.weight = np.ones((np.size(nodesEV,0),2))
		self.step = step
		self.maxEpoch = maxEpoch
		self.var = np.ones((np.size(nodesEV,0),2)) #variance
		self.epoch = 0
		self.errTreshold = 0.1
		self.mainLoop()

	def mainLoop(self):
		while self.epoch < self.maxEpoch:
			for i in range(np.size(self.train,0)):
				self.weight += self.deltaWeights(i)
				if self.errorCalc(self.train, self.trainTarget) < self.errTreshold:
					return
			self.epoch += 1


	def deltaWeights(self, index):
		activations = self.activation(self.train[index])
		a = self.trainTarget[index] - np.dot(activations.T, self.weight)
		a = a[[0],:]
		b = self.step * a
		delta_w = b * self.activation(self.train[index])
		return delta_w

	def activation(self, x):
		activations = np.exp(np.multiply((-(x-self.nodesEV)**2),((2*self.var)**(-1))))

		#3 vikter ska uppdateras        
		#1a
		bestActivationIndex1 = np.argmax(activations.sum(axis=1))
		activations[bestActivationIndex1]= 0


		#2a
		bestActivationIndex2 = np.argmax(activations.sum(axis=1))
		activations[bestActivationIndex2]=0

		#3a
		bestActivationIndex3 = np.argmax(activations.sum(axis=1))
		activations[bestActivationIndex3]=0

		#4e
		bestActivationIndex4 = np.argmax(activations.sum(axis=1))
		activations[bestActivationIndex4]=0

		#5e
		bestActivationIndex5 = np.argmax(activations.sum(axis=1))
		activations[bestActivationIndex5]=0

		activations = np.zeros_like(activations)
		activations[bestActivationIndex1]=1
		activations[bestActivationIndex2]=0.8
		activations[bestActivationIndex3]=0.6
		activations[bestActivationIndex4]=0.4
		activations[bestActivationIndex5]=0.2
		return activations

	def run(self, input):
		result = np.zeros((np.size(input,0),2))
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
	ballist = np.loadtxt('ballist.dat')
	balltest = np.loadtxt('balltest.dat')

	# Training data
	angleVelocity = ballist[:,[0,1]]
	distHeight = ballist[:,[2,3]]

	# Testing data
	angleVelocityTest = balltest[:,[0,1]]
	distHeightTest = balltest[:,[2,3]]


	index = np.linspace(0, (angleVelocity.size-1)/2, 10, dtype = int)
	b = DeltaRule(angleVelocity, distHeight, angleVelocity[index], 0.01, 100)
	#print(distHeightTest[:,0])
	#print(distHeightTest[:, 1])
	plt.scatter(distHeightTest[:,0], distHeightTest[:,1], c = 'blue', label = "Output data")
	plt.scatter(b.run(angleVelocityTest)[:,0], b.run(angleVelocityTest)[:,1], c = 'orange', label = "Prediction")
	#plt.scatter(angleVelocityTest[:,1], b.run(angleVelocityTest)[:,0], c = 'blue')
	#plt.plot(angleVelocity[index], distHeight[index], 'b+')
	plt.title("Balltest")
	plt.xlabel("Distance")
	plt.ylabel("Height")
	plt.legend()
	plt.show()


	'''

	testError = []
	trainError = []

	
	print("Residual error is: ", b.errorCalc(angleVelocityTest, distHeightTest))
	
	print("Nodes \t Training error \t Testing error") 
	for i in range(1, 25):
		index = np.linspace(0, (angleVelocity.size-1)/2, num = i, dtype = int)
		dr = DeltaRule(angleVelocity, distHeight, angleVelocity[index], 0.01, 100)
		print(i, " \t ", dr.errorCalc(angleVelocity, distHeight), " \t ", dr.errorCalc(angleVelocityTest, distHeightTest))
		testError.append(dr.errorCalc(angleVelocityTest, distHeightTest))
		trainError.append(dr.errorCalc(angleVelocity, distHeight))


	plt.plot(range(len(testError)), testError, label = "test error")
	plt.plot(range(len(testError)), trainError, label = "training error")
	plt.legend()
	plt.show()
	'''