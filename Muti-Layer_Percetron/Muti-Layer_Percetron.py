# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 00:17:22 2017

@author: Reader
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
sess = tf.InteractiveSession()

# 1.定义网络结构
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu((tf.matmul(x,W1) + b1))
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)

y = tf.nn.softmax(tf.matmul(hidden1_drop,W2) + b2)

# 2.定于损失函数和选择优化器
y_ = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 3.训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    loss,_ = sess.run([cross_entropy,train_step],feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
    if(i%100 == 0):
        print("Epoch:%04d"%(i),"  loss={:.9f}".format(loss))

# 4.测试
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("mnist test accracy:",accracy.eval({x:mnist.test.images,y_:mnist.test.labels,
                    keep_prob:1.0}))