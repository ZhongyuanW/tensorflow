# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 01:05:48 2017

@author: Reader
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print("train images shape:",mnist.train.images.shape,",train label shape:",mnist.train.labels.shape)
print("test images shape:",mnist.test.images.shape,",test label shape:",mnist.test.labels.shape)
print("validation images shape:",mnist.validation.images.shape,",validation label shape:",mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch_xc,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xc,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))



