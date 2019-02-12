import numpy as np
import matplotlib.pyplot as plt

## Generates two classes of data
def generate_data(n = 100, specific=False):
    if specific:
        mA = [1, 0.3]
        mB = [0, -0.1]
        sigmaA = [0.2, 0.2]
        sigmaB = [0.3, 0.3]

        classA_x1_1 = np.random.randn(int(0.5*n), 1) * sigmaA[0] - mA[0]
        classA_x1_2 = np.random.randn(int(0.5*n), 1) * sigmaA[0] + mA[0]
        classA_x1 = np.concatenate((classA_x1_1, classA_x1_2), axis=0)
        classA_x2 = np.random.randn(n, 1) * sigmaA[1] + mA[1]
        classA = np.concatenate((classA_x1, classA_x2), axis=1)
        classB = np.random.randn(n, 2) * sigmaB + mB
    else:
        mA = [0, 0]
        mB = [1, 1]
        sigmaA = [1, 1]
        sigmaB = [2, 2]
        classA = np.random.randn(n, 2) * sigmaA + mA
        classB = np.random.randn(n, 2) * sigmaB + mB

    # Create X and T matrices
    target = np.array([1]*n + [-1]*n)
    pattern = np.concatenate((classA, classB), axis=0)
    pattern = np.insert(pattern, 2, values=np.ones(n*2), axis=1)

    return pattern.T, target

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

def plot_learning_rate(list, string):
    plt.plot(range(len(list)),list, '-',label="Accuracy")
    title='Accuracy for ' + string + ' learning'
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

## Plots separation boundary
def plot_data(w, x, t, title="Graph title"):
    x = x.T
    for i in range(len(t)):
        if t[i] == -1:
            plt.plot(x[i][0], x[i][1], 'o', color='blue')
        else:
            plt.plot(x[i][0], x[i][1], 'x', color='orange')

    lin_x1 = np.linspace(-5, 5, 3)
    lin_x2 = (-w[0]/w[1])*lin_x1 + w[2]/np.linalg.norm(w)

    plt.plot(lin_x1, lin_x2, '-r', label='Separation Boundary')

    plt.title(title)
    plt.xlabel('x1', color='#1C2833')
    plt.ylabel('x2', color='#1C2833')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()


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

def delta_rule_seq(w, x, t, epochs=20, learning_rate=0.0001):
    acc = []
    x = x.T
    for i in range(epochs):
        for j in range(len(t)):
            delta_w = -learning_rate*(np.dot((np.dot(w, x[j]) -t[j]), x[j].T))
            w = w + delta_w
        acc.append(accuracy(w, x.T, t))
    return w, acc

def split_25(patterns, targets):
    patterns = patterns.T
    targets = targets.T

    n = int(len(targets)*0.5) # Number of datapoints per class
    classA = patterns[:n]
    classB = patterns[n:]
    targetA = targets[:n]
    targetB = targets[n:]

    # Retrive a list of random indexes
    row_i = np.random.choice(n, int(n*0.75), replace=False)

    targetA_ss = []
    targetB_ss = []
    classA_ss = []
    classB_ss = []
    for i in row_i:
        targetA_ss.append(targetA[i])
        targetB_ss.append(targetB[i])
        classA_ss.append(classA[i])
        classB_ss.append(classB[i])

    # Put subselections together
    pattern = np.concatenate((classA_ss, classB_ss), axis=0)
    target = np.concatenate((targetA_ss, targetB_ss), axis=0)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return pattern[s].T, target[s].T

def split_50(patterns, targets, classifier):
    patterns = patterns.T
    targets = targets.T

    n = int(len(targets)*0.5) # Number of datapoints per class
    classA = patterns[:n]
    classB = patterns[n:]
    targetA = targets[:n]
    targetB = targets[n:]

    # Retrive a list of random indexes
    row_i = np.random.choice(n, int(n*0.5), replace=False)

    if classifier == 1:
        targetA_ss = []
        classA_ss = []
        for i in row_i:
            targetA_ss.append(targetA[i])
            classA_ss.append(classA[i])
        targetB_ss = targetB
        classB_ss = classB
    else:
        targetB_ss = []
        classB_ss = []
        for i in row_i:
            targetB_ss.append(targetB[i])
            classB_ss.append(classB[i])
        targetA_ss = targetA
        classA_ss = classA

    # Put subselections together
    pattern = np.concatenate((classA_ss, classB_ss), axis=0)
    target = np.concatenate((targetA_ss, targetB_ss), axis=0)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return pattern[s].T, target[s].T

def split_special(patterns, targets):
    patterns = patterns.T
    targets = targets.T

    n = int(len(targets)*0.5) # Number of datapoints per class
    classA = patterns[:n]
    classB = patterns[n:]
    targetA = targets[:n]
    targetB = targets[n:]

    a_div = int(n*0.5)  # Number of datapoints lower or greater than 0
    classA_low = classA[:a_div]     # All points where classA x1 < 0
    classA_high = classA[a_div:]    # All points where classA x1 > 0

    # Retrive a list of random indexes to subselect with
    low_i = np.random.choice(a_div, int(a_div*0.8), replace=False)
    high_i = np.random.choice(a_div, int(a_div*0.2), replace=False)

    classA_low_ss = []
    for i in low_i:
        classA_low_ss.append(classA_low[i])

    classA_high_ss = []
    for i in high_i:
        classA_high_ss.append(classA_high[i])

    # Adjust number of targets to class A
    n_remaining = int(a_div*0.8) + int(a_div*0.2)
    targetA_ss = targetA[:n_remaining]

    # Put subselections together
    classA_ss = np.concatenate((classA_low_ss, classA_high_ss), axis=0)
    pattern = np.concatenate((classA_ss, classB), axis=0)
    target = np.concatenate((targetA_ss, targetB), axis=0)

    # Shuffles data
    s = np.arange(target.shape[0])
    np.random.shuffle(s)

    return pattern[s].T, target[s].T

def plot_sep_bound(w_all, w_list, patterns, targets):
    x = patterns.T
    for i in range(len(targets)):
        if targets[i] == -1:
            plt.plot(x[i][0], x[i][1], 'o', color='blue')
        else:
            plt.plot(x[i][0], x[i][1], 'x', color='orange')

    lin_x1 = np.linspace(-3, 3, 2)
    lin_x2 = (-w_all[0]/w_all[1])*lin_x1 + w_all[2]/np.linalg.norm(w_all)
    plt.plot(lin_x1, lin_x2, '-', label="All data")

    for i in range(len(w_list)):
        w = w_list[i]
        lin_x1 = np.linspace(-3, 3, 2)
        lin_x2 = (-w[0]/w[1])*lin_x1 + w[2]/np.linalg.norm(w)
        plt.plot(lin_x1, lin_x2, '-', label="Subset %s" %(i+1))

    plt.title("Separation bounds for different subsets")
    plt.xlabel('x1', color='#1C2833')
    plt.ylabel('x2', color='#1C2833')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

def accuracy_2(w, x, t):
    classA = 0
    classB = 0
    classA_corr = 0
    classB_corr = 0
    predictions = []
    corr = 0
    x = x.T
    for i in range(len(t)):
        pred = np.dot(w,x[i])             # Takes bias into account
        if pred > 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    for i in range(len(t)):
        if t[i] == 1:
            classA += 1
            if predictions[i] == 1: classA_corr += 1
        else:
            classB += 1
            if predictions[i] == -1: classB_corr += 1
    print(classA)
    print(classA_corr)
    A_acc = (int(classA_corr/classA*100))/100
    B_acc = (int(classB_corr/classB*100))/100
    print(A_acc)
    return "[" + str(A_acc) + " | " + str(B_acc) + "]"


def part1():
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

def part2_1():
    l_rate = 0.0001
    e = 20
    p_all, t_all = generate_data(1000, specific=True)
    w_rand = np.random.randn(3)  # Initializing weights

    # Split up data
    p_25, t_25 = split_25(p_all, t_all)
    p_a, t_a = split_50(p_all, t_all, 1)
    p_b, t_b = split_50(p_all, t_all, -1)
    p_special, t_special = split_special(p_all, t_all)

    # Trains different models using delta rule batch mode
    d_all, db_acc_all = delta_rule_batch(w_rand, p_all, t_all, epochs=e, learning_rate=l_rate)
    d_25, db_acc_25 = delta_rule_batch(w_rand, p_25, t_25, epochs=e, learning_rate=l_rate)
    d_a, db_acc_a = delta_rule_batch(w_rand, p_a, t_a, epochs=e, learning_rate=l_rate)
    d_b, db_acc_b = delta_rule_batch(w_rand, p_b, t_b, epochs=e, learning_rate=l_rate)
    d_special, db_acc_special = delta_rule_batch(w_rand, p_b, t_b, epochs=e, learning_rate=l_rate)

    # Trains different models using perceptron rule batch mode
    pb_all, pb_acc_all = perceptron_rule_batch(w_rand, p_all, t_all, epochs=e, learning_rate=l_rate)
    pb_25, pb_acc_25 = perceptron_rule_batch(w_rand, p_25, t_25, epochs=e, learning_rate=l_rate)
    pb_a, pb_acc_a = perceptron_rule_batch(w_rand, p_a, t_a, epochs=e, learning_rate=l_rate)
    pb_b, pb_acc_b = perceptron_rule_batch(w_rand, p_b, t_b, epochs=e, learning_rate=l_rate)
    pb_special, pb_acc_special = perceptron_rule_batch(w_rand, p_special, t_special, epochs=e, learning_rate=l_rate)


    # Plot separation boundaries
#    plot_data(d_all, p_all, t_all, title="Separation boundary for SLP using Delta rule - All")
#    plot_data(d_25, p_25, t_25, title="Separation boundary for SLP using Delta rule, subset 1")
#    plot_data(d_a, p_a, t_a, title="Separation boundary for SLP using Delta rule, subset 2")
#    plot_data(d_b, p_b, t_b, title="Separation boundary for SLP using Delta rule, subset 3")
    w_list = [d_25, d_a, d_b, d_special]

    # Plot learning curves for batch delta rule
#    plt.plot(range(len(db_acc_all)), db_acc_all, '-', label="All data (%s)" % ((int(accuracy(d_all, p_all, t_all)*100)/100)))
#    plt.plot(range(len(db_acc_25)), db_acc_25, '-', label="Subset 1 (%s)" % ((int(accuracy(d_25, p_25, t_25)*100)/100)))
#    plt.plot(range(len(db_acc_a)), db_acc_a, '-', label="Subset 2 %s" % (accuracy_2(d_a, p_a, t_a)))
#    plt.plot(range(len(db_acc_b)), db_acc_b, '-', label="Subset 3 %s" % (accuracy_2(d_b, p_b, t_b)))
#    plt.plot(range(len(db_acc_special)), db_acc_special, '-', label="Subset 4 %s" % (accuracy_2(d_special, p_special, t_special)))

    # Plot learning curves for batch delta rule
    plt.plot(range(len(pb_acc_all)), pb_acc_all, '-', label="PB All data (%s)" % ((int(accuracy(pb_all, p_all, t_all)*100)/100)))
    plt.plot(range(len(pb_acc_25)), pb_acc_25, '-', label="PB Subset 1 (%s)" % ((int(accuracy(pb_25, p_25, t_25)*100)/100)))
    plt.plot(range(len(pb_acc_a)), pb_acc_a, '-', label="PB Subset 2 %s" % (accuracy_2(pb_a, p_a, t_a)))
    plt.plot(range(len(pb_acc_b)), pb_acc_b, '-', label="PB Subset 3 %s" % (accuracy_2(pb_b, p_b, t_b)))
    plt.plot(range(len(pb_acc_special)), pb_acc_special, '-', label="PB Subset 4 %s" % (accuracy_2(pb_special, p_special, t_special)))

    title = 'Learning curve Perceptron rule  \n Learning rate = %s Epochs = %s' % (l_rate, e)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch', color='#1C2833')
    plt.ylabel('Ratio of correct classifications', color='#1C2833')
    plt.show()

#    plot_sep_bound(d_all, w_list, p_all, t_all)
#    plot_data(d_special, p_special, t_special, title="Separation boundary for SLP using Delta rule \n n = 1000, Subset 4")


if __name__ == "__main__":
    #part1()
    part2_1()