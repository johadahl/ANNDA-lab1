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

def recall(pattern, w, show_ori=False, show_out=False):
    dim = pattern.size
    count = 0
    e = []
    e.append(calc_energy(pattern, w))
    c = []
    c.append(count)

    previousPattern = np.zeros(dim)

    if show_ori:
        plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
        plt.show()

    while True:
        s = np.dot(w, pattern)
        s[s >= 0] = 1
        s[s < 0] = -1

        pattern = s

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
        if count == 25:
            print("Stopped after 50 iterations")
            break
        count += 1
        print("Iteration: %s" % count)
        c.append(count)
        e.append(calc_energy(pattern, w))

    if show_out:
        plt.imshow(pattern.reshape(32, 32), interpolation="nearest")
        plt.show()
    return pattern, e, c

def check_True(predicted_pattern):
    solved = False
    for i in range(patterns.shape[0]):
        if np.array_equal(patterns[i], predicted_pattern):
            print("True! It matched pattern nr:", i)
            solved = True
    return solved

def generate_random_data(dim, show=False):
    # Generate random input data
    r = np.random.randn(dim)
    for i in range(r.size):
        if r[i] >= 0:
            r[i] = 1
        else:
            r[i] = -1
    if show and dim == 1024:
        plt.imshow(r.reshape(32, 32), interpolation="nearest")
        plt.show()
    return r

def sequential_random_recall(pattern, w, draw=False, step_size=1000):
    print("Starting sequential")
    np.fill_diagonal(w, 0)  # Reset diagonals to 0
    dim = pattern.size
    count = 0

    e = []
    c = []
    while True:
        i = np.random.randint(0, dim)   # Select random unit
        s = np.dot(w[i], pattern.T)
        if s >= 0:
            sign = 1
        else:
            sign = -1
        pattern[i] = sign
        if (count % step_size) == 0:
            e.append(calc_energy(pattern, w))
            c.append(count)
            print(count)
        if draw:
            if (count % step_size) == 0:
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
        if count > 5000:
            break
        previousPattern = pattern
        count += 1
    print(count)
    return pattern, e, c

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

def calc_energy(pattern, w):
    dim = pattern.size
    energy = 0
#    enrg = np.multiply(w, np.multiply.outer(pattern, pattern.T)).sum()
    for i in range(dim):
        for j in range(dim):
            energy += w[i][j]*pattern[i]*pattern[j]
    return -energy

def bullet_1():
    print("Energy for image 1: %s " % (calc_energy(recall(data[0], w), w)))
    print("Energy for image 2: %s " % (calc_energy(recall(data[1], w), w)))
    print("Energy for image 3: %s " % (calc_energy(recall(data[2], w), w)))
    print("Energy for local_minima: %s " % calc_energy(recall(data[10], w), w))
#    r = generate_random_data()
#    print("Energy for local_minima: %s " % calc_energy(recall(r, w), w))

def bullet_2():
    print("Energy for image 10: %s " % (calc_energy(data[9], w)))
    print("Energy for image 11: %s " % (calc_energy(data[10], w)))

def bullet_3():
    p, e, c = sequential_random_recall(data[9], w)
    plt.title("Energy levels over sequential updates")
    plt.plot(c, e, linestyle='-', color='b')
    plt.show()

def bullet_4():
    dim = 64
    w = np.random.randn(dim, dim)
    np.fill_diagonal(w, 0)  # Reset diagonals to 0
    p = generate_random_data(dim, show=True)
    print(calc_energy(p, w))
    p, e, c = recall(p, w)
    plt.title("Energy levels over sequential updates on random weights and data")
    plt.plot(c, e, linestyle='-', color='b')
    plt.show()

def bullet_5():
    dim = 64
    w = np.random.randn(dim, dim)
    w = 0.5*np.add(w, w.T)
    np.fill_diagonal(w, 0)  # Reset diagonals to 0
    p = generate_random_data(dim, show=False)
    p, e, c = recall(p, w)
#    p, e, c = sequential_random_recall(p, w, step_size=100)
    plt.title("Energy levels over sequential updates on random weights and data")
    plt.plot(c, e, linestyle='-', color='b')
    plt.show()


data = np.loadtxt('./pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

# Init & trains model on p1, p2, p3
patterns = data[0:3,:]
#w = init_weights(patterns)
#print("Weights trained")

# Bullet 1 - What is the energy at the different attractors?
#bullet_1()

# Bullet 2 - What is the energy at the points of the distorted patterns?
#bullet_2()

# Bullet 3 - Follow how the energy changes from iteration to iteration when you use the sequential update rule to approach an attractor.
#bullet_3()

# Bullet 4 - Generate a weight matrix by setting the weights to normally distributed random numbers,
# and try iterating an arbitrary starting state. What happens?
#bullet_4()

# Bullet 5 - Make the weight matrix symmetric (e.g. by setting w=0.5*(w+w')). What happens now? Why?
bullet_5()



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