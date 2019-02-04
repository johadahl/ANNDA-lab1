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
    data = np.insert(data, 2, values=np.ones(n*2), axis=1)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return data[s], target[s]

def delta_rule(nodes=1, iterations=1, step_length=0.1):
    pass

if __name__ == "__main__":
    w = np.random.randn(2)  # Initializing weights
    x, t = generate_data(5)
