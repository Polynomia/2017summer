#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
batch_size=100
#get data
mnist = mx.test_utils.get_mnist()
sigma1 = float(1/np.e)
sigma2 = float((np.e)**-6)

hidden_1 = 128
hidden_2 = 64
data_in = 784
data_out = 10
w_prior_pi = np.pi
#initialize two iterators
train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batch_size,shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size)

def get_pw_logNormal(x,mu,sigma):
    return  -0.5 * (np.log(2*np.pi) - np.log(sigma) - (x - mu)**2 / (2 * sigma**2))
def get_log_qw_theta(x,mu,sigma):
    return  -0.5 * (np.log(2*np.pi) - mx.sym.log(sigma) - (x - mu)**2 / (2 * sigma**2))


data=mx.sym.Variable('data')
data=mx.sym.flatten(data=data)
yInput = mx.sym.Variable('yInput')

eps1 = mx.sym.random_normal(loc = 0, scale = 1, shape = (data_in,hidden_1))
eps2 = mx.sym.random_normal(loc = 0, scale = 1, shape = (hidden_1,hidden_2))
eps3 = mx.sym.random_normal(loc = 0, scale = 1, shape = (hidden_2,data_out))

mu1 = mx.sym.Variable('mu1',shape=(data_in,hidden_1))
mu2 = mx.sym.Variable('mu2',shape=(hidden_1,hidden_2))
mu3 = mx.sym.Variable('mu3',shape=(hidden_2,data_out))

pho1 = mx.sym.Variable('pho1',shape=(data_in,hidden_1))
pho2 = mx.sym.Variable('pho2',shape=(hidden_1,hidden_2))
pho3 = mx.sym.Variable('pho3',shape=(hidden_2,data_out))

w1 = mu1+mx.sym.log(mx.sym.exp(pho1)+1)*eps1
w2 = mu2+mx.sym.log(mx.sym.exp(pho2)+1)*eps2
w3 = mu3+mx.sym.log(mx.sym.exp(pho3)+1)*eps3

w_total = mx.sym.concat(mx.sym.flatten(w1),mx.sym.flatten(w2),mx.sym.flatten(w3),dim = 0)
sigma_theta = mx.sym.log(1+mx.sym.exp(mx.sym.concat(mx.sym.flatten(pho1),mx.sym.flatten(pho2),mx.sym.flatten(pho3),dim = 0)))
mu_total = mx.sym.concat(mx.sym.flatten(mu1),mx.sym.flatten(mu2),mx.sym.flatten(mu3),dim = 0)



fc1 = mx.sym.dot(data,w1)
act1 = mx.sym.Activation(data = fc1, act_type = "relu")
fc2 = mx.sym.dot(act1,w2)
act2 = mx.sym.Activation(data = fc2, act_type = "relu")
fc3 = mx.sym.dot(act2,w3)
mlp = mx.sym.SoftmaxOutput(data = fc3, name = "softmax")

log_qw_theta = mx.sym.sum(get_log_qw_theta(w_total,mu_total,sigma_theta))
log_pw = mx.sym.sum()

mlp_model = mx.mod.Module(symbol=mlp,context=mx.cpu())
# #declare the nn
# data=mx.sym.Variable('data')
# data=mx.sym.flatten(data=data)
# eps1 = mx.sym.randpm_normal(0,1,(1,))
# #layers
# fc1=mx.sym.FullyConnected(data=data,num_hidden=128)
# act1=mx.sym.Activation(data=fc1,act_type="relu")
#
#
# fc2=mx.sym.FullyConnected(data=act1,num_hidden=64)
# act2=mx.sym.Activation(data=fc2,act_type="relu")
# fc3=mx.sym.FullyConnected(data=act2,num_hidden=10)
# mlp=mx.sym.SoftmaxOutput(data=fc3,name='softmax')
#
# logging.getLogger().setLevel(logging.DEBUG)
#
# mlp_model = mx.mod.Module(symbol=mlp,context=mx.cpu())
# mlp_model.fit(train_iter,eval_data=val_iter,optimizer='sgd',optimizer_params={'learning_rate':0.1},
#               eval_metric='acc',
#               batch_end_callback=mx.callback.Speedometer(batch_size,100),
#               num_epoch=10)
#
# test_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size)
# acc=mx.metric.Accuracy()
#
# mlp_model.score(test_iter,acc)
# mx.viz.plot_network(symbol=mlp).view()
# print(acc)
