import numpy as np
import matplotlib.pyplot as plt

## Generates two classes of data
def generate_data(n = 100, bias=True, specific=True):
    if specific:
        mA = [2, 2]
        mB = [-2, -2]
        sigmaA = [2, 2]
        sigmaB = [2, 2]
    else:
        mA = [1, 0.3]
        mB = [0, -0.1]
        sigmaA = [0.2, 0.2]
        sigmaB = [0.3, 0.3]

    classA = np.random.randn(int(0.5*n), 2)*sigmaA + mA
    classB = np.random.randn(int(0.5*n), 2)*sigmaB + mB

    # Create X and T matrices
    target = np.array([1]*int(n*0.5) + [-1]*int(n*0.5))
    pattern = np.concatenate((classA, classB), axis=0)
    # Adds ones for bias
    if bias:
        pattern = np.insert(pattern, 2, values=np.ones(n*2), axis=1)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return pattern[s].T, target[s]


def delta_rule_batch(w, x, t, epochs=20, learning_rate=0.0001):
    acc = []
    for i in range(epochs):
        # delta_w = -learning_rate(w*x - t)*x.T
        delta_w = -learning_rate*(np.dot((np.dot(w, x) - t), x.T))
        w = w + delta_w
        acc.append(accuracy(w, x, t))
    return w, acc

## Returns ratio of missclassified data points
def accuracy(w, x, t):
    predictions = []
    corr = 0
    x = x.T
    for i in range(len(t)):
#        pred = np.dot(w[:2], x[i][:2])  # Don't take bias into consideration for threshold function
        pred = np.dot(w,x[i])             # Takes bias into account
        if pred > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    for i in range(len(t)):
        if predictions[i] == t[i]:
            corr += 1
    acc = corr / len(t)
    return acc

def plot_learning_rate(list, type):
    plt.plot(range(len(list)),list, '-',label="Accuracy")
    title='Accuracy for ' + type + ' learning'
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

## Plots separation boundary
def plot_sep_bound(w, x, t, title="Graph title"):
    x = x.T
    for i in range(len(t)):
        if t[i] == -1:
            plt.plot(x[i][0], x[i][1], 'o', color='blue')
        else:
            plt.plot(x[i][0], x[i][1], 'x', color='red')

    lin_x1 = np.linspace(-10, 10, 2)
    lin_x2 = (-w[0]/w[1])*lin_x1 + w[2]/np.linalg.norm(w)

    plt.plot(lin_x1, lin_x2, '-r', label='Separation Boundary')
    plt.title(title)
    plt.xlabel('x1', color='#1C2833')
    plt.ylabel('x2', color='#1C2833')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

def main():
    l_rate = 0.0001
    e = 20

    p, t = generate_data(100)
    w_rand = np.random.randn(3)  # Initializing weights
    d_batch_w, db_acc = delta_rule_batch(w_rand, p, t, epochs=e, learning_rate=l_rate)

    print("Final accuracy for Delta rule, batch: " + str(accuracy(d_batch_w, p, t)))
    print(d_batch_w)
    plot_sep_bound(d_batch_w, p, t, title="Separation boundary for Single Perceptron usning Delta rule")

    plt.plot(range(len(db_acc)), db_acc, '-', label="Batch Delta rule learning curve")
    title = 'Learning curve for non-linear separable data'
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
