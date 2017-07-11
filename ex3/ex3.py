#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy, matplotlib, random
import scipy.misc
import matplotlib.pyplot as plot
import scipy.io  # read .mat file
import scipy.optimize

data = scipy.io.loadmat('ex3data1.mat')
# print(type(data))
# data is a dictionary
X, y = data['X'], data['y']
X = np.insert(X, 0, 1, axis=1)
print(X.shape, y.shape)


def pixel2square(row):
    square = row[1:].reshape(20, 20)
    return square.T

def displayData(indices=None):
    rows, columns = 10, 10
    if not indices:
        indices = random.sample(range(X.shape[0]), rows * columns)
    fig = np.zeros((20 * rows, 20 * columns))
    r, c = 0, 0
    for idx in indices:
        if c == columns:
            c = 0
            r += 1
        fig[r * 20:(r + 1) * 20, c * 20:(c + 1) * 20] = pixel2square(X[idx])
        c += 1
    img = scipy.misc.toimage(fig)
    plot.imshow(img, cmap=matplotlib.cm.Greys_r)
    plot.show()


def h(theta, X):
    z = X.dot(theta)  # shape 5000,1
    return 1. / (1. + np.exp(-z))


#displayData()


# def costFunction(myTheta, X, myy, myLambda=0.):
#     m = X.shape[0]
#     theta = myTheta.T
#     y = myy.T
#     J = 0.
#     hx = h(theta, X)  # shape:5000,1
#     temp1 = np.log(hx) * y + np.log(1 - hx) * (1 - y)
#     J += temp1.sum()
#     J += (1. / (2. * m)) * myLambda * np.sum(theta * theta)
#     return J
#
#
# def gradient(myTheta, X, myy, myLambda=0.):
#     m = X.shape[0]
#     y = myy.T
#     theta = myTheta.T
#     delta_h = (h(theta, X) - y)  # shape 5000,1
#     temp = X.T  # shape 401,5000
#     grad = (1. / m) * temp.dot(delta_h) + (myLambda / m) * theta  # shape 401,1
#     return grad.T

def computeCost(mytheta,myX,myy,mylambda = 0.):
    m = myX.shape[0] #5000
    myh = h(mytheta,myX) #shape: (5000,1)
    term1 = np.log( myh ).dot( -myy.T ) #shape: (5000,5000)
    term2 = np.log( 1.0 - myh ).dot( 1 - myy.T ) #shape: (5000,5000)
    left_hand = (term1 - term2) / m #shape: (5000,5000)
    right_hand = mytheta.T.dot( mytheta ) * mylambda / (2*m) #shape: (1,1)
    return left_hand + right_hand #shape: (5000,5000)

def costGradient(mytheta,myX,myy,mylambda = 0.):
    m = myX.shape[0]
    #Tranpose y here because it makes the units work out in dot products later
    #(with the way I've written them, anyway)
    beta = h(mytheta,myX)-myy.T #shape: (5000,5000)

    #regularization skips the first element in theta
    regterm = mytheta[1:]*(mylambda/m) #shape: (400,1)

    grad = (1./m)*np.dot(myX.T,beta) #shape: (401, 5000)
    #regularization skips the first element in theta
    grad[1:] = grad[1:] + regterm
    return grad #shape: (401, 5000)

def optimizeTheta(theta, X, y, myLambda=0.):
    result = scipy.optimize.fmin_cg(computeCost, x0=theta, fprime=costGradient, args=(X, y, myLambda), maxiter=80,
                                    disp=False,full_output=True)
    return result[0], result[1]


def getTheta():
    # this is a function to get theta for every class
    myLambda = 0;
    initial_theta = np.zeros((1, X.shape[1]))
    Theta = np.zeros((10, X.shape[1]))
    for i in range(10):
        iclass = i if i else 10
        print('Optimizing for %d' % i)
        new_y = np.array([1 if x == iclass else 0 for x in y])  # .reshape((-1,1))
        itheta, iminicost = optimizeTheta(initial_theta, X, new_y, myLambda)
        Theta[i, :] = itheta
    print("done")
    return Theta

def predictOneVsAll(Theta,row):
    hypo = np.zeros((10,1))
    for i in range(10):
        hypo[i]=h(Theta[i].T,row)
    return np.argmax(hypo)

Theta = getTheta()
n_correct, n_total = 0., 0.
incorrect_indices = []
for irow in range(X.shape[0]):
    n_total += 1
    if predictOneVsAll(Theta,X[irow]) == y[irow]:
        n_correct += 1
    else: incorrect_indices.append(irow)
print("Training set accuracy: %0.1f%%" %(100*(n_correct/n_total)))

displayData(incorrect_indices[200:300])