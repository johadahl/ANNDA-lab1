import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_data(n = 100):
    mA = [10, 5]
    mB = [-10, 0]
    sigmaA = 0.5
    sigmaB = 1
    classA = np.random.randn(n, 2)*sigmaA + mA
    classB = np.random.randn(n, 2)*sigmaB + mB

    # Create X and T matrices
    target = np.array([1]*n + [-1]*n)
    data = np.concatenate((classA, classB), axis=0)
    # Adds ones for bias
    data = np.insert(data, 2, values=np.ones(n*2), axis=1)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return data[s], target[s]

def delta_rule(x, t, learning_rate=0.001, epochs=20):
    w = np.random.randn(2)  # Initializing weights

    # Pseudo-code
    # delta_w = -learning_rate(w*x - t)*x.transposed

    pass
def something_else(nodes=1, iterations=1, step_length=0.1):
    pass


if __name__ == "__main__":
    x, t = generate_data(5)
