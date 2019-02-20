import numpy as np

def init_weights():
    np.random.seed(42)
    weights = np.random.rand(100, 84)
    return weights

# Calculates similarity between a pattern (animal) and weights, choses the weightnode with smallest distance
def similarity(indata, weights):
    winner = 1000   # Arbitrary large number to initialise
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
def getNeighbours(weights, size, winner_index, ind):
    left = []
    right = []

    add = range(1, size+1)
    for val in add:
        rgt = winner_index + val
        if rgt < 100:
            right.append(rgt)

    subtract = range(-size, 0)
    for val in subtract:
        lft = winner_index + val
        if lft >= 0:
            left.append(lft)

    left = np.array(left)
    right = np.array(right)

    if left.size != 0:
        updateWeights(weights, left, ind)
    if right.size != 0:
        updateWeights(weights, right, ind)

# Updates weight W[i]
def updateWeights(weights, weight_index, ind, eta=0.2):
    for i in np.nditer(weight_index):
        weights[i] = weights[i] + eta * (np.subtract(ind, weights[i]))

# Trains a SOM
def trainSOM(indata, weights, epochs=20):

    size = 25  # Size of neighbourhood (in each direction) at start
    # For each pattern in indata
    for epoch in range(epochs):  # 20 is standard
        for i in range(indata.shape[0]):
            winnerNode = similarity(indata[i], weights)  # Find best node
            getNeighbours(weights, size, winnerNode, indata[i])  # Get list of neighbours with winnerNode in center

        # print("Size:",size)
        # Update size of neighbourhood
        if size > 5:
            size -= 2
        elif size == 2:
            pass
        else:
            size -= 1

# Creates a SOM based on training
def predictSOM(indata, weights, animal_names):
    pos = []
    # Loop through animals
    for i in range(indata.shape[0]):
        winnerNode = similarity(indata[i], weights)  # Find best node
        pos.append([winnerNode, animal_names[i]])

    pos = np.array(pos, dtype=object)
    a = pos[pos[:, 0].argsort()]
    for i in a:
        print(str(i[0]) + ", " + i[1] + ", ")


def main():

    myarray = np.loadtxt('./data_lab2/animals.dat',delimiter=",",dtype=int)
    animal_names = np.loadtxt('./data_lab2/animalnames.txt',dtype=str)

    #indata = 32x84
    indata = np.reshape(myarray, (-1, 84))
    #weights = 100x84
    weights = init_weights()

    print(indata.shape)
    print(animal_names.shape)

    trainSOM(indata, weights)
    predictSOM(indata, weights, animal_names)

    # Visualize this later

def eval():
    same = 0
    bat = [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0]
    el = [0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0]
    for i in range(len(bat)):
        if bat[i] == el[i]:
            same += 1
    print(same)

if __name__ == '__main__':
#    main()
    eval()
