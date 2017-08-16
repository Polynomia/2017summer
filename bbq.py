#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import random
ctx = mx.cpu()
batch_size = 100

# get data
mnist = mx.test_utils.get_mnist()
sigma1 = float(1 / np.e)
sigma2 = float((np.e) ** -6)

hidden_1 = 128
hidden_2 = 64
data_in = 784
data_out = 10
w_prior_pi = 0.25
# initialize two iterators
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

print(mnist['train_label'].shape, mnist['train_data'].shape)
print(train_iter.provide_data, train_iter.provide_label)


def get_log_pw(x, mu, sigma):
    return -0.5 * (np.log(2 * np.pi) - np.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2))


def get_log_qw_theta(x, mu, sigma):
    return -0.5 * (np.log(2 * np.pi) - mx.sym.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2))


data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
yInput = mx.sym.Variable('yInput')

# eps1_w = mx.sym.random_normal(loc=0, scale=1)
# eps2_w = mx.sym.random_normal(loc=0, scale=1)
# eps3_w = mx.sym.random_normal(loc=0, scale=1)
#
# eps1_b = mx.sym.random_normal(loc=0, scale=1)
# eps2_b = mx.sym.random_normal(loc=0, scale=1)
# eps3_b = mx.sym.random_normal(loc=0, scale=1)

# eps1_w = np.random.standard_normal((hidden_1,data_in))
# eps2_w = np.random.standard_normal((hidden_2,hidden_1))
# eps3_w = np.random.standard_normal((data_out,hidden_2))
#
# eps1_b = np.random.standard_normal((hidden_1,))
# eps2_b = np.random.standard_normal((hidden_2,))
# eps3_b = np.random.standard_normal((data_out,))

eps1_w = mx.sym.var('eps1_w',init=mx.init.Constant(random.gauss(0,1)))
eps2_w = mx.sym.var('eps2_w',init=mx.init.Constant(random.gauss(0,1)))
eps3_w = mx.sym.var('eps3_w',init=mx.init.Constant(random.gauss(0,1)))

eps1_b = mx.sym.var('eps1_b',init=mx.init.Constant(random.gauss(0,1)))
eps2_b = mx.sym.var('eps2_b',init=mx.init.Constant(random.gauss(0,1)))
eps3_b = mx.sym.var('eps3_b',init=mx.init.Constant(random.gauss(0,1)))

mu1_w = mx.sym.Variable('mu1_w', init=mx.init.Zero())
mu2_w = mx.sym.Variable('mu2_w', init=mx.init.Zero())
mu3_w = mx.sym.Variable('mu3_w', init=mx.init.Zero())

mu1_b = mx.sym.Variable('mu1_b', init=mx.init.Zero())
mu2_b = mx.sym.Variable('mu2_b', init=mx.init.Zero())
mu3_b = mx.sym.Variable('mu3_b', init=mx.init.Zero())

pho1_w = mx.sym.Variable('pho1_w', init=mx.init.Normal(1))
pho2_w = mx.sym.Variable('pho2_w', init=mx.init.Normal(1))
pho3_w = mx.sym.Variable('pho3_w', init=mx.init.Normal(1))

pho1_b = mx.sym.Variable('pho1_b', init=mx.init.Normal(1))
pho2_b = mx.sym.Variable('pho2_b', init=mx.init.Normal(1))
pho3_b = mx.sym.Variable('pho3_b', init=mx.init.Normal(1))

w1_w = mu1_w + mx.sym.log(mx.sym.exp(pho1_w) + 1) * eps1_w
w2_w = mu2_w + mx.sym.log(mx.sym.exp(pho2_w) + 1) * eps2_w
w3_w = mu3_w + mx.sym.log(mx.sym.exp(pho3_w) + 1) * eps3_w

w1_b = mu1_b + mx.sym.log(mx.sym.exp(pho1_b) + 1) * eps1_b
w2_b = mu2_b + mx.sym.log(mx.sym.exp(pho2_b) + 1) * eps2_b
w3_b = mu3_b + mx.sym.log(mx.sym.exp(pho3_b) + 1) * eps3_b


fc1 = mx.sym.FullyConnected(data=data, weight=w1_w, bias=w1_b, num_hidden=hidden_1)
act1 = mx.sym.Activation(data=fc1, act_type="relu")
fc2 = mx.sym.FullyConnected(data=act1, weight=w2_w, bias=w2_b, num_hidden=hidden_2)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
fc3 = mx.sym.FullyConnected(data=act2, weight=w3_w, bias=w3_b, num_hidden=data_out)
#mlp = mx.sym.softmax(data=fc3, name="softmax")
mlp = mx.sym.SoftmaxOutput(data = fc3,label = yInput,name='softmax')

log_qw_theta = mx.sym.sum(get_log_qw_theta(w1_w, mu1_w, mx.sym.log(mx.sym.exp(pho1_w) + 1))) \
               + mx.sym.sum(get_log_qw_theta(w1_b, mu1_b, mx.sym.log(mx.sym.exp(pho1_b) + 1))) \
               + mx.sym.sum(get_log_qw_theta(w2_w, mu2_w, mx.sym.log(mx.sym.exp(pho2_w) + 1))) \
               + mx.sym.sum(get_log_qw_theta(w2_b, mu2_b, mx.sym.log(mx.sym.exp(pho2_b) + 1))) \
               + mx.sym.sum(get_log_qw_theta(w3_w, mu3_w, mx.sym.log(mx.sym.exp(pho3_w) + 1))) \
               + mx.sym.sum(get_log_qw_theta(w3_b, mu3_b, mx.sym.log(mx.sym.exp(pho3_b) + 1)))
log_pw = mx.sym.sum(w_prior_pi * get_log_pw(w1_w, 0, sigma1) + (1 - w_prior_pi) * get_log_pw(w1_w, 0, sigma2)) \
         + mx.sym.sum(w_prior_pi * get_log_pw(w1_b, 0, sigma1) + (1 - w_prior_pi) * get_log_pw(w1_b, 0, sigma2)) \
         + mx.sym.sum(w_prior_pi * get_log_pw(w2_w, 0, sigma1) + (1 - w_prior_pi) * get_log_pw(w2_w, 0, sigma2)) \
         + mx.sym.sum(w_prior_pi * get_log_pw(w2_b, 0, sigma1) + (1 - w_prior_pi) * get_log_pw(w2_b, 0, sigma2)) \
         + mx.sym.sum(w_prior_pi * get_log_pw(w3_w, 0, sigma1) + (1 - w_prior_pi) * get_log_pw(w3_w, 0, sigma2)) \
         + mx.sym.sum(w_prior_pi * get_log_pw(w3_b, 0, sigma1) + (1 - w_prior_pi) * get_log_pw(w3_b, 0, sigma2))
p_dw = mx.sym.sum(mx.sym.softmax_cross_entropy(mlp, yInput))

f = log_qw_theta - log_pw - p_dw
loss = mx.sym.MakeLoss(f)

group = mx.symbol.Group([loss, mx.sym.BlockGrad(mlp)])
bbp_model = mx.mod.Module(symbol=group, data_names=('data',),
                          label_names=('yInput',), context=ctx)

bbp_model.bind(data_shapes=[('data', (batch_size, 1, 28, 28))], label_shapes=[('yInput', (batch_size,))])
bbp_model.init_params(initializer=mx.init.Normal(1))
bbp_model.init_optimizer(optimizer='sgd',optimizer_params=(('learning_rate',0.001),))
#mx.viz.plot_network(mlp).view()
pred = []
for batch in train_iter:
    acc=0
    bbp_model.forward(batch,is_train=True)
    for array in bbp_model.get_outputs()[1].asnumpy():
        pred.append(np.argmax(array))
    for x,y in zip(batch.label[0].asnumpy(),pred):
        if x == y:
            acc+=1
    print(acc/100)
    bbp_model.backward()
    bbp_model.update()
    break

