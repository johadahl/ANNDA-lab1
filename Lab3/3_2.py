import numpy as np
from matplotlib import pyplot as plt

def init_weights(patterns):
    dim = patterns.shape[1]
    w = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            s = 0
            for pattern in patterns:
                s += pattern[i] * pattern[j]
            w[i][j] = (1 / (patterns.size)) * s
    np.fill_diagonal(w, 0)  # Reset diagonals to 0
    return w

def recall(pattern, w):
    dim = pattern.size
    count = 0
    energyLevels = []

    previousPattern = np.zeros(dim)
    while True:
        s = np.dot(w, pattern)
        s[s >= 0] = 1
        s[s < 0] = -1

        pattern = s
        count += 1

        plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
        plt.show()

        if check_True(pattern):
            print("It took:", count, "nr of iterations")
            plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
            plt.show()
            break

        elif np.array_equal(pattern, previousPattern):
            print("Local minimum found %s iterations" % (count))
            plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
            plt.show()
            break

        previousPattern = pattern
        if count == 50:
            print("Stopped after 50 iterations")
            break

def check_True(predicted_pattern):
    solved = False
    for i in range(patterns.shape[0]):
        if np.array_equal(patterns[i], predicted_pattern):
            print("True! It matched pattern nr:", i)
            solved = True
    return solved

def generate_random_data(show=False):
    # Generate random input data
    r = np.random.randn(1024)
    for i in range(r.size):
        if r[i] >= 0:
            r[i] = 1
        else:
            r[i] = -1
    if show:
        plt.imshow(data[10].reshape(32, 32), interpolation="nearest")
        plt.show()
    return r

def random_recall(pattern, w):
    np.fill_diagonal(w, 0)  # Reset diagonals to 0
    dim = pattern.size
    count = 0

    previousPattern = np.zeros(dim)
    while True:
        i = np.random.randint(0, dim)   # Select random unit

        s = np.dot(w[i], pattern.T)
        count += 1

        if s >= 0:
            sign = 1
        else:
            sign = -1
        pattern[i] = sign

        if (count % 5) == 0:
            print(count)
            plt.title("Output after %s iterations" % count)
            plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
            f_name = "./gif_32/%s.png" % count
            plt.savefig(f_name)
            plt.clf()

        if check_True(pattern):
            print("It took:", count, "nr of calculations")
            plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
            plt.show()
            break
        previousPattern = pattern

def genRandWeights(patterns):
    dim1 = patterns.shape[1]  # Just to get the size of the weight vector
    w = np.random.randn(dim1, dim1)
    return w


def genStartingState(w):
    dim = 1024
    out = np.ones(dim)
    for i in range(dim):
        s = 0
        for j in range(dim):
            s += w[i][j] * out[j]
        if s >= 0:
            sign = 1
        else:
            sign = -1
        out[i] = sign

    plt.imshow(out.reshape(32, 32), interpolation="nearest")
    plt.show()

def bullet_1():
    recall(data[0], w)
    recall(data[1], w)
    recall(data[2], w)

def bullet_2():
    # Should match p1
    plt.imshow(data[9].reshape(32, 32), interpolation="nearest")
    plt.show()
    print("Start p1")
    recall(data[9], w)
    # Should match p2
    print("Start p2")
    plt.imshow(data[10].reshape(32, 32), interpolation="nearest")
    plt.show()
    recall(data[10], w)

def bullet_3():
    random_recall(data[10], w)


data = np.loadtxt('./pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
for d in data:
    plt.imshow(d.reshape(32, 32), interpolation="nearest")
    plt.show()

# Init & trains model on p1, p2, p3
patterns = data[0:3,:]
w = init_weights(patterns)
print("Weights trained")

# Bulletpoint 1 - Test if stable
#bullet_1()

# Bulletpoint 2 - Can the network complete a degraded pattern?
#bullet_2()

# Bulletpoint 3 - Testing random units
bullet_3()

quit()


w = genRandWeights(patterns[0:3, :])
genStartingState(w)
# recall(data[0],w)


randW = genRandWeights(patterns[0:3,:])
symW = np.multiply(0.5,np.add(randW,randW.T))
#np.fill_diagonal(symW,0)

recall(data[0],symW)
recall(data[0],randW)
#random_recall(data[0], symW)