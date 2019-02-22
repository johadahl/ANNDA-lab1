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

        #plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
        #plt.show()

        if check_True(pattern):
            #print("It took:", count, "nr of iterations")
            return True
            #plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
            #plt.show()

        elif np.array_equal(pattern, previousPattern):
            #print("Local minimum found %s iterations" % (count))
            #plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
            #plt.show()
            return False

        previousPattern = pattern
        if count == 10:
            #print("Stopped after 10 iterations")
            return False

def check_True(predicted_pattern):
    solved = False
    for i in range(patterns.shape[0]):
        if np.array_equal(patterns[i], predicted_pattern):
            #print("True! It matched pattern nr:", i)
            solved = True
    return solved

def generate_random_data(show=False):
    # Generate random input data
    r = np.random.randn(100)
    #r += 0.5
    for i in range(r.size):
        if r[i] >= 0:
            r[i] = 1
        else:
            r[i] = -1
    if show:
        plt.imshow(r.reshape(10, 10), interpolation="nearest")
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


def flip(pattern, percentage):
    index = np.arange(pattern.size)
    np.random.shuffle(index)

    abso = int(percentage*pattern.size)
    flipIndex = index[:abso]

    flipped = np.copy(pattern)
    flipped[flipIndex] = -1*flipped[flipIndex]

    #plt.imshow(flipped.reshape(32,32),interpolation="nearest")
    #plt.show()

    return flipped

def bullet_1():
    recall(data[0], w)
    recall(data[1], w)
    recall(data[2], w)
    recall(data[3], w)
    recall(data[4], w)
    recall(data[5], w)
    recall(data[6], w)

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

def bullet_341():

    array1 = []
    array2 = []
    array3 = []
    for j in range(0,3):
        for i in range(1,100):
            print("Bild: ", 1, " Procent: ",str(int(i*1)))
            d = flip(data[j],i*0.01)
            found = recall(d,w)
            if found == True:
                if j==0:
                    array1.append(1)
                elif j==1:
                    array2.append(1)
                else:
                    array3.append(1)
            else:
                if j==0:
                    array1.append(0)
                elif j==1:
                    array2.append(0)
                else:
                    array3.append(0)
            print("\nKörning färdig")

    plt.plot(range(len(array1)), array1, label="Picture 1")
    plt.plot(range(len(array2)), array2, label="Picture 2")
    plt.plot(range(len(array3)), array3, label="Picture 3")
    plt.legend()
    plt.title("Finding attractors over different %")
    plt.show()

def bullet_342(numOfPatterns):
    for i in range(0,numOfPatterns):
        r = generate_random_data()
        if i == 0:
            patterns = np.array([r])
        else:
            patterns = np.concatenate((patterns, [r]), axis=0)
    return patterns

#data = np.loadtxt('./pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)


#for d in data:
#    plt.imshow(d.reshape(32, 32), interpolation="nearest")
#    plt.show()

# Init & trains model on p1, p2, p3
#patterns = data[0:7,:]
#w = init_weights(patterns)

# Bulletpoint 1 - Test if stable
#bullet_1()

# Bulletpoint 2 - Can the network complete a degraded pattern?
#bullet_2()

# Bulletpoint 3 - Testing random units
#bullet_3()

"""
Uppgift nr 3.4 börjar här
"""
# Bulletpoint 1
#bullet_341()

# Bulletpoint 2
foundPerNumber0 = []
foundPerNumber1 = []
foundPerNumber2 = []
foundPerNumber3 = []
foundPerNumber4 = []
foundPerNumber5 = []
noiseLevel = [0, 20, 30, 40, 50, 60]
for n in noiseLevel:
    print("\nLevel of noise: " + str(n) + "%")
    for k in range(1,40):
        patterns = bullet_342(k)
        w = init_weights(patterns)
        nrsFound = 0
        for i in range(0, np.size(patterns, axis=0)):
            d = flip(patterns[i], n*0.01)
            found = recall(d, w)
            if found == True:
                nrsFound = nrsFound +1
        if n == 0:
            print("Done with nr: " + str(len(foundPerNumber0)))
            foundPerNumber0.append(nrsFound)
        elif n == 20:
            print("Done with nr: " + str(len(foundPerNumber1)))
            foundPerNumber1.append(nrsFound)
        elif n == 30:
            print("Done with nr: " + str(len(foundPerNumber2)))
            foundPerNumber2.append(nrsFound)
        elif n == 40:
            print("Done with nr: " + str(len(foundPerNumber3)))
            foundPerNumber3.append(nrsFound)
        elif n == 50:
            print("Done with nr: " + str(len(foundPerNumber4)))
            foundPerNumber4.append(nrsFound)
        else:
            print("Done with nr: " + str(len(foundPerNumber5)))
            foundPerNumber5.append(nrsFound)
x = []
for i in range(len(foundPerNumber0)):
    x.append(i+1)
print(x)

plt.xlabel("Memory size")
plt.ylabel("Patterns safely stored")
plt.plot(x, foundPerNumber0, label="0% Noise")
plt.plot(x, foundPerNumber1, label="20% Noise")
plt.plot(x, foundPerNumber2, label="30% Noise")
plt.plot(x, foundPerNumber3, label="40% Noise")
plt.plot(x, foundPerNumber4, label="50% Noise")
plt.plot(x, foundPerNumber5, label="60% Noise")
plt.legend()
plt.title("Numbers of safely stored patterns over memory and different amount of noise")
plt.show()

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