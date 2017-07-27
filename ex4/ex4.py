#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy, matplotlib, random
import scipy.misc
import matplotlib.pyplot as plot
import scipy.io  # read .mat file
import scipy.optimize
import itertools
from scipy.special import expit  # Vectorized sigmoid function


def pixel2square(row):
    square = row[1:].reshape(20, 20)
    return square.T


def displayData(source, indices=None, rows=10, columns=10):
    # rows, columns = 10,
    if not indices:
        indices = random.sample(range(source.shape[0]), rows * columns)
    fig = np.zeros((20 * rows, 20 * columns))
    r, c = 0, 0
    for idx in indices:
        if c == columns:
            c = 0
            r += 1
        fig[r * 20:(r + 1) * 20, c * 20:(c + 1) * 20] = pixel2square(source[idx])
        c += 1
    img = scipy.misc.toimage(fig)
    plot.imshow(img, cmap=matplotlib.cm.Greys_r)
    plot.show()


def pf(row, Theta):
    a = row
    z_avi = []
    for i in range(len(Theta)):
        iTheta = Theta[i]
        z = iTheta.dot(a)
        a = 1. / (1 + np.exp(-z))
        z_avi.append((z, a))  # the first one is z and the second is the activations for next layer
        if i == len(Theta) - 1:
            return z_avi
        a = np.insert(a, 0, 1)


def predictNN(row, Theta):
    p = pf(row, Theta)
    return np.argmax(p) + 1


def unrollTheta(Theta):
    # flattened_theta = np.row_stack(mytheta.flatten() for mytheta in Theta).reshape(-1)
    flattened_list = [mytheta.flatten() for mytheta in Theta]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size + 1) * hidden_layer_size + \
                            (hidden_layer_size + 1) * output_layer_size
    return np.array(combined).reshape((len(combined), 1))


def reshape_theta(flattened_theta):
    Theta1 = flattened_theta[:(input_layer_size + 1) * hidden_layer_size].reshape(
        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = flattened_theta[(input_layer_size + 1) * hidden_layer_size:].reshape(
        (output_layer_size, hidden_layer_size + 1))
    return [Theta1, Theta2]


def unrollX(X):
    return np.array(X.flatten()).reshape((m * (input_layer_size + 1), 1))


def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((m, input_layer_size + 1))


def costFunction(flattened_theta, flattenedX, y, myLamdba=0.):
    Theta = reshape_theta(flattened_theta)
    X = reshapeX(flattenedX)
    cost = 0
    for i in range(m):
        h = pf(X[i], Theta)[-1][1]  # the last one
        tmpy = np.zeros((10, 1))
        tmpy[y[i] - 1] = 1  # shape 10,1
        tmpcost = -np.log(h).dot(tmpy) - np.log(1 - h).dot(1 - tmpy)
        cost += tmpcost
    cost = float(cost) / m
    # compute regularized term
    reg = 0
    for itheta in Theta:
        reg += np.sum(itheta * itheta)
    reg *= float(myLamdba) / (2 * m)
    return reg + cost


def sigmoidDradient(z):
    gz = expit(z)
    return (1 - gz) * gz


def rand_initial():
    epsilon_init = 0.12
    theta1_shape = (hidden_layer_size, input_layer_size + 1)
    theta2_shape = (output_layer_size, hidden_layer_size + 1)
    rand_theta = [np.random.random(theta1_shape) * 2 * epsilon_init - epsilon_init, \
                  np.random.random(theta2_shape) * 2 * epsilon_init - epsilon_init]
    return rand_theta


def bp(flattened_theta, flattenedX, y, myLambda=0.):
    Theta = reshape_theta(flattened_theta)
    X = reshapeX(flattenedX)
    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros(((output_layer_size, hidden_layer_size + 1)))
    for i in range(m):
        a1 = X[i]
        temp = pf(a1, Theta)
        z2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1].reshape((10, 1))
        tmpy = np.zeros((10, 1))
        tmpy[y[i] - 1] = 1  # shape 10,1
        delta3 = a3 - tmpy
        delta2 = Theta[1].T[1:, :].dot(delta3) * sigmoidDradient(z2).reshape((-1, 1))  # shape (25,10)(10,1)=(25,1)
        Delta1 += delta2.dot(a1.T.reshape(1, -1))  # shape (25,1)(1,401)=25,401
        a2 = np.insert(a2, 0, 1)
        Delta2 += delta3.dot(a2.reshape((1, -1)))  # shape (10,1)(1,26)=10,26
    D1, D2 = Delta1 / float(m), Delta2 / float(m)
    # regularized:
    D1[:, 1:] = D1[:, 1:] + (float(myLambda) / m) * Theta[0][:, 1:]
    D2[:, 1:] = D2[:, 1:] + (float(myLambda) / m) * Theta[1][:, 1:]
    return unrollTheta([D1, D2]).flatten()


def gradientChecking(Theta, D, X, y, myLambda=0.):
    epsilon = 0.0001
    flattendX = unrollX(X)
    flattend_theta = unrollTheta(Theta)
    flattendD = unrollTheta(D)
    n = len(flattend_theta)
    for i in range(10):
        j = int(random.random() * n)
        epsilon_vec = np.zeros((n, 1))
        epsilon_vec[j] = epsilon
        high = costFunction(flattend_theta + epsilon_vec, flattendX, y, myLambda)
        low = costFunction(flattend_theta - epsilon_vec, flattendX, y, myLambda)
        numerical_gradient = (high - low) / float(2 * epsilon)
        print("No.%d theta Numerical gradient is %f. BP gradient is %f." % (j, numerical_gradient, flattendD[j]))


def fmincg(myLambda=0.):
    init_theta = unrollTheta(rand_initial())
    result = scipy.optimize.fmin_cg(costFunction, x0=init_theta, fprime=bp, \
                                    args=(flattendX, y, myLambda), maxiter=50, disp=True, full_output=True)
    print("Done")
    return reshape_theta(result[0])


def predict(row, Theta):
    p = pf(row, Theta)[-1][1]
    return np.argmax(p)


def computeAccuracy(X, y, Theta):
    n_correct, n_total = 0, m
    for i in range(m):
        if predict(X[i], Theta) + 1 == y[i]:
            n_correct += 1
    print("The accuracy is %0.1f%%" % (100 * (float(n_correct) / n_total)))


if __name__ == '__main__':
    data = scipy.io.loadmat('ex4weights.mat')
    Theta1, Theta2 = data['Theta1'], data['Theta2']
    print(Theta1.shape, Theta2.shape)
    data2 = scipy.io.loadmat('ex4data1.mat')
    X, y = data2['X'], data2['y']
    X = np.insert(X, 0, 1, axis=1)
    print(X.shape, y.shape)
    Theta = [Theta1, Theta2]
    print(X.shape, y.shape)

    # These are some global variables I'm suing to ensure the sizes
    # of various matrices are correct
    # these are NOT including bias nits
    input_layer_size = 400
    hidden_layer_size = 25
    output_layer_size = 10
    m = X.shape[0]

    flattendX = unrollX(X)
    flattend_theta = unrollTheta(Theta)
    flattendD = bp(flattend_theta, flattendX, y)
    D1, D2 = reshape_theta(flattendD)
    gradientChecking(Theta, [D1, D2], X, y)
    computeAccuracy(X, y, Theta)
    # reg_theta=fmincg(myLambda=10.)
    # computeAccuracy(X,y,reg_theta)
    # displayData(source=Theta[0], rows=5, columns=5)
    displayData(source=X)
