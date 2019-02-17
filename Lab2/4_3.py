import numpy as np
import matplotlib.pyplot as plt

# Calculates similarity between a pattern (animal) and weights, choses the weightnode with smallest distance
def similarity(indata, weights):
    num_change = 0
    winner = 1000
    winner_row = 0
    winner_col = 0
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            sub = np.subtract(indata, weights[i][j])
            sim = np.dot(sub.T, sub)
            if sim < winner:
                winner = sim
                winner_row = i
                winner_col = j
                num_change += 1
    return (winner_row, winner_col)


# Takes the index of the winner node, uses the window to call update weight function for
# appropriate neighbours
def updateNeighbours(weights, radius, winner, ind, eta=0.2):
    if (radius > 0):
        for i in range(winner[0] - radius, winner[0] + radius + 1):
            for j in range(winner[1] - radius, winner[1] + radius + 1):
                if i < 10 and i >= 0 and j < 10 and j >= 0:
                    weights[i][j] = weights[i][j] + eta * (np.subtract(ind, weights[i][j]))
    else:
        row = winner[0]
        col = winner[1]
        weights[row][col] = weights[row][col] + eta * (np.subtract(ind, weights[row][col]))

# Trains a SOM
def trainSOM(indata, weights, epochs=50):
    size = 3  # Size of neighbourhood

    for epoch in range(epochs):  # 20 is standard
        # For each pattern in indata - 349 times

        for i in range(indata.shape[0]):
            winnerNode = similarity(indata[i], weights)  # Find best node
            updateNeighbours(weights, size, winnerNode, indata[i])  # Get list of neighbours with winnerNode in center

        # Update size of neighbourhood
        if epochs < 15:
            size = 1
        elif epochs < 10:
            size = 0


def map_gender(indata, weights, gender):
    pos = []
    # Loop through data
    for i in range(indata.shape[0]):
        winnerNode = similarity(indata[i], weights)  # Find best node
        pos.append([winnerNode, gender[i]])

    male = np.zeros((10, 10))
    female = np.zeros((10, 10))
    ratio = np.full((10, 10), 0.5)
    total = np.zeros((10, 10))

    for d in pos:
        row = d[0][0]
        col = d[0][1]
        val = int(d[1])
        if val == 0:
            male[row][col] += 1
        if val == 1:
            female[row][col] += 1
        total[i][j] += 1

    # Calculating the ratio
    for i in range(10):
        for j in range(10):
            if male[i][j] != 0 or female[i][j] != 0:
                m = male[i][j]
                f = female[i][j]
                if m != 0 and f != 0:
                    r = m/f
                elif m == 0:
                    r = 1
                elif f == 0:
                    r = 0
                ratio[i][j] = r
#    print(ratio)
    print()

def main():
    indata = np.reshape(np.loadtxt('./data_lab2/votes.dat', delimiter=",", dtype=float), (-1, 31))
    names = [line.rstrip('\n') for line in open('./data_lab2/mpnames.txt')]
    district = np.loadtxt('./data_lab2/mpdistrict.dat', dtype=int)

    # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    party = np.loadtxt('./data_lab2/mpparty.dat', skiprows=2, dtype=str)
    # Coding: Male 0, Female 1
    sex = np.loadtxt('./data_lab2/mpsex.dat', skiprows=2, dtype=str)

    weights = np.random.rand(10, 10, 31)

    trainSOM(indata, weights)
    map_gender(indata, weights, sex)


if __name__ == '__main__':
    main()
