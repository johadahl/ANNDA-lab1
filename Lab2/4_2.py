import numpy as np
import matplotlib.pyplot as plt

def init_weights():
#    np.random.seed(42)
    weights = np.random.rand(10,2)
    return weights

# Calculates similarity between a pattern (animal) and weights, choses the weightnode with smallest distance
def similarity(indata, weights):
    winner = 1000
    winnerNode = 0
    for i in range(weights.shape[0]):
        sub = np.subtract(indata, weights[i])
        sim = np.dot(sub.T, sub)
        if sim < winner:
            winner = sim
            winnerNode = i
    return winnerNode


# Takes the index of the winner node, uses the window to call update weight function for
# appropriate neighbours
def getNeighbours(weights, size, winner, ind):
    indexes = []

    if (size > 0):
        for i in range(winner - size, winner + size):
            tmp = i % 10
            indexes.append(tmp)
    else:
        indexes.append(winner)

    indexes = np.array(indexes)
    updateWeights(weights, indexes, ind)


# Updates weight W[i]
def updateWeights(weights, weightIndex, ind, eta=0.2):
    for i in np.nditer(weightIndex):
        weights[i] = weights[i] + eta * (np.subtract(ind, weights[i]))


# Trains a SOM
def trainSOM(indata, weights, epochs=40):
    size = 2  # Size of neighbourhood

    for epoch in range(epochs):  # 20 is standard

        # For each pattern in indata
        for i in range(indata.shape[0]):
            winnerNode = similarity(indata[i], weights)  # Find best node
            getNeighbours(weights, size, winnerNode, indata[i])  # Get list of neighbours with winnerNode in center
            plt.title("Epoch %s, datapoint %s" % (epoch, i))
            plt.plot(weights[:, 0], weights[:, 1], linestyle='-', marker='x', color='r')
            for j in indata:
                plt.plot(j[0], j[1], linestyle='-', marker='o', color='b')
#            plt.show()
            f_name = "./img_42/%s_%s.png" % (epoch, i)
            print(f_name)
            plt.savefig(f_name)
            plt.clf()

        # Update size of neighbourhood
        if epochs < 15:
            size = 1
        elif epochs < 10:
            size = 0


# Creates a SOM based on training and plots tour
def predictSOM(indata, weights):
    pos = []
    # Loop through animals
    for i in range(indata.shape[0]):
        winnerNode = similarity(indata[i], weights)  # Find best node
        print(winnerNode)
        pos.append([winnerNode, indata[i][0], indata[i][1]])

    pos = np.array(pos, dtype=object)

    pos = pos[pos[:, 0].argsort()]
    print(weights)
    # for i in range(pos.shape[0]):
    plt.plot(weights[:, 0], weights[:, 1], linestyle='-', marker='x', color='r')
    plt.plot(pos[:, 1], pos[:, 2], linestyle='-', marker='o', color="b")
    # path = Path(pos[:,1:])
    # plt.plot(path) #spara denna
    plt.show()


def main():
    indata = np.loadtxt('./data_lab2/cities.dat', delimiter=",", skiprows=4, dtype=str)
    weights = init_weights()
    for i in range(indata.shape[0]):
        # row = row.strip(";")
        indata[i][1] = indata[i][1].strip(";")

    indata = indata.astype(float)

    trainSOM(indata, weights)
    predictSOM(indata, weights)

if __name__ == '__main__':
    main()
