import numpy as np
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
    pattern = np.concatenate((classA, classB), axis=0)
    # Adds ones for bias
    pattern = np.insert(pattern, 2, values=np.ones(n*2), axis=1)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return pattern[s].T, target[s]

def delta_rule(x, t, epochs=20, learning_rate=0.001):
    w = np.random.randn(3)  # Initializing weights
    print(w)
    for i in range(epochs):
        # delta_w = -learning_rate(w*x - t)*x.transposed
        delta_w = -learning_rate*(np.dot((np.dot(w, x) - t), x.T))
        w = w + delta_w
        print(delta_w)
    return w


def test(w, x, t):
    predictions = []
    corr = 0
    x = x.T
    for i in range(len(t)):
        pred = np.dot(w, x[i])
        if pred > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    for i in range(len(predictions)):
        if predictions[i] == t[i]:
            corr += 1
    acc = corr / len(t)
    return acc

if __name__ == "__main__":
    p, t = generate_data(100)
    w = delta_rule(p, t)
    print(w)
    print(test(w, p, t))
