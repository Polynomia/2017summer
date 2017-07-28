#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mxnet as mx
import logging

batch_size=100
#get data
mnist = mx.test_utils.get_mnist()

#initialize two iterators
train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batch_size,shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size)

#declare the nn
data=mx.sym.Variable('data')
data=mx.sym.flatten(data=data)

#layers
fc1=mx.sym.FullyConnected(data=data,num_hidden=128)
act1=mx.sym.Activation(data=fc1,act_type="relu")
fc2=mx.sym.FullyConnected(data=act1,num_hidden=64)
act2=mx.sym.Activation(data=fc2,act_type="relu")
fc3=mx.sym.FullyConnected(data=act2,num_hidden=10)
mlp=mx.sym.SoftmaxOutput(data=fc3,name='softmax')

logging.getLogger().setLevel(logging.DEBUG)

mlp_model = mx.mod.Module(symbol=mlp,context=mx.cpu())
mlp_model.fit(train_iter,eval_data=val_iter,optimizer='sgd',optimizer_params={'learning_rate':0.1},
              eval_metric='acc',
              batch_end_callback=mx.callback.Speedometer(batch_size,100),
              num_epoch=10)

test_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size)
acc=mx.metric.Accuracy()

mlp_model.score(test_iter,acc)

print(acc)
