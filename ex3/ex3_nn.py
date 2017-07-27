#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
import scipy, matplotlib, random
import scipy.misc
import matplotlib.pyplot as plot


def pixel2square(row):
    square = row[1:].reshape(20, 20)
    return square.T


# this is the propagate forward
def pf(row, Theta):
    a = row
    for i in range(len(Theta)):
        iTheta = Theta[i]
        z = iTheta.dot(a)
        a = 1. / (1 + np.exp(-z))
        # print(a.shape)
        if i == len(Theta) - 1:
            return a
        a = np.insert(a, 0, 1)


def predictNN(row, Theta):
    p = pf(row, Theta)
    return np.argmax(p) + 1


if __name__ == '__main__':
    data = scipy.io.loadmat('ex3weights.mat')
    Theta1, Theta2 = data['Theta1'], data['Theta2']
    print(Theta1.shape, Theta2.shape)
    data2 = scipy.io.loadmat('ex3data1.mat')
    X, y = data2['X'], data2['y']
    X = np.insert(X, 0, 1, axis=1)
    print(X.shape, y.shape)
    Theta = [Theta1, Theta2]

    n_correct, n_total = 0., 0.
    incorrect_indices = []

    for row in range(X.shape[0]):
        n_total += 1
        if predictNN(X[row], Theta) == int(y[row]):
            n_correct += 1
        else:
            incorrect_indices.append(row)
    print("Training set accuracy: %0.1f%%" % (100 * (n_correct / n_total)))

    for x in range(5):
        i = random.choice(incorrect_indices)
        fig = plot.figure(figsize=(3, 3))
        img = scipy.misc.toimage(pixel2square(X[i]))
        plot.imshow(img, cmap=matplotlib.cm.Greys_r)
        predicted_val = predictNN(X[i], Theta)
        predicted_val = 0 if predicted_val == 10 else predicted_val
        fig.suptitle('Predicted: %d' % predicted_val, fontsize=14, fontweight='bold')
        plot.show()
