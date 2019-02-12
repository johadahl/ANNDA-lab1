import numpy as np
import matplotlib.pyplot as plt

## Generates two classes of data
def generate_data(n = 100, bias=True):
    mA = [-20, -20]
    mB = [20, 20]
    sigmaA = [1, 1]
    sigmaB = [2, 2]
    classA = np.random.randn(n, 2)*sigmaA + mA
    classB = np.random.randn(n, 2)*sigmaB + mB

    # Create X and T matrices
    target = np.array([1]*n + [-1]*n)
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
        delta_w = -learning_rate*(np.dot((np.dot(w, x) - t), x.T))
        w = w + delta_w/len(t)
        acc.append(accuracy(w, x, t))
    return w, acc

def delta_rule_seq(w, x, t, epochs=20, learning_rate=0.0001):
    acc = []
    x = x.T
    for i in range(epochs):
        for j in range(len(t)):
            delta_w = -learning_rate*(np.dot((np.dot(w, x[j]) -t[j]), x[j].T))
            w = w + delta_w
        acc.append(accuracy(w, x.T, t))
    return w, acc

def perceptron_rule_batch(w, x, t, epochs=20, learning_rate=0.0001):
    acc = []
    x = x.T
    for i in range(epochs):
        delta_w = 0
        for j in range(len(x)):
            prediction = np.dot(w, x[j])  # w*X
            if not (prediction < 0) == (t[j] < 0):
                error = t[i] - prediction
                delta_w += learning_rate * np.dot(error, np.transpose(x[j]))

        w = w + delta_w
        acc.append(accuracy(w, x.T, t))
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

def evaluate_learning_rates():
    learning_rate = [0.01, 0.001, 0.0001, 0.00001]
    e = 20

    for l_rate in learning_rate:
        db_series = 0
        ds_series = 0
        pb_series = 0
        for i in range(100):
            p, t = generate_data(100)
            w_rand = np.random.randn(3)  # Initializing weights
            d_batch_w, db_acc = delta_rule_batch(w_rand, p, t, epochs=e, learning_rate=l_rate)
            d_seq_w, ds_acc = delta_rule_seq(w_rand, p, t, epochs=e, learning_rate=l_rate)
            p_batch_w, pb_acc = perceptron_rule_batch(w_rand, p, t, epochs=e, learning_rate=l_rate)
            db_series += accuracy(d_batch_w, p, t)
            ds_series += accuracy(d_seq_w, p, t)
            pb_series += accuracy(p_batch_w, p, t)
        print("%s || D-Batch: %s   || D-Seq: %s   || P-Batch: %s " % (l_rate, db_series/100, ds_series/100, pb_series/100))

def main():
    l_rate = 0.001
    e = 20

    p, t = generate_data(100)
    w_rand = np.random.randn(3)  # Initializing weights
    d_batch_w, db_acc = delta_rule_batch(w_rand, p, t, epochs=e, learning_rate=l_rate)
    d_seq_w, ds_acc = delta_rule_seq(w_rand, p, t, epochs=e, learning_rate=l_rate)
    p_batch_w, pb_acc = perceptron_rule_batch(w_rand, p, t, epochs=e, learning_rate=l_rate)

    print("Final accuracy for Delta rule, batch: " + str(accuracy(d_batch_w, p, t)))
    print("Final accuracy for Delta rule, sequential: " + str(accuracy(d_seq_w, p, t)))
    print("Final accuracy for Perceptron rule, batch: " + str(accuracy(p_batch_w, p, t)))
    print(d_batch_w)
    print(d_seq_w)
    print(p_batch_w)
#    plot_sep_bound(p_batch_w, p, t, title="Separation boundary for Single Perceptron using perceptron rule")

    plt.plot(range(len(db_acc)), db_acc, '-', label="Batch Delta rule (%s)" % (accuracy(d_batch_w, p, t)))
    plt.plot(range(len(ds_acc)), ds_acc, '-', label="Sequential Delta rule (%s)" % (accuracy(d_seq_w, p, t)))
    plt.plot(range(len(pb_acc)), pb_acc, '-', label="Batch Perceptron rule (%s)" % (accuracy(p_batch_w, p, t)))
    title = 'Learning curve depending on rule and training mode \n Learning rate = %s Epochs = %s' % (l_rate, e)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch', color='#1C2833')
    plt.ylabel('Ratio of correct classifications', color='#1C2833')
    plt.show()
    plot_sep_bound(d_batch_w, p, t, title="Separation boundary for Single-Layer Perceptron using Delta rule")


def q3():
    l_rate = 0.001
    e = 20

    acc = 0
    for i in range(100):
        p, t = generate_data(100, bias=False)
        w_rand = np.random.randn(2)  # Initializing weights
        d1, d1_acc = delta_rule_batch(w_rand, p, t, epochs=e, learning_rate=l_rate)
        a = accuracy(d1, p, t)
        acc += a
        print("Final accuracy for Delta rule, batch: " + str(a))
        print(d1)
        plt.plot(range(len(d1_acc)), d1_acc, '-')

    title = 'Learning curve without bias '
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
    print(acc/100)

if __name__ == "__main__":
    main()
    #q3()
    #evaluate_learning_rates()
