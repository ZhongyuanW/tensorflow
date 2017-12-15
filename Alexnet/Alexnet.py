# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:02:20 2017

@author: Reader
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batchs = 100
 
def print_activations(t):
        print(t.op.name,' ',t.get_shape().as_list())
        
def inference(images):
    parameters = []
    #第1卷积层
    #Conv 11x11s4,64
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],
                     dtype=tf.float32,stddev=1e-1),name='weights')

        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                             trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv1)    
    
    #lrn层和pool层
    #Max Pool 3x3s2
    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='VALID',name='pool1')
    print_activations(pool1)

    #第2卷积层
    #Conv 5x5s1,192
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],
                     dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),
                             trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv2)
    
    #lrn层和pool层
    #Max Pool 3x3s2
    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='VALID',name='pool2')
    print_activations(pool2)
    
    #第3卷积层
    #Conv 3x3s1,384
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],
                     dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),
                             trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv3)
    
    #第4卷积层
    #Conv 3x3s1,256
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],
                     dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),
                             trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv4)
    
    #第5卷积层
    #Conv 3x3s1,256
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],
                     dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),
                             trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
    print_activations(conv5)
    
    #pool层
    #Max Pool 3x3s2
    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='VALID',name='pool5')
    print_activations(pool5)
    return pool5,parameters
    
def fc(pool5):
    parameters = []
    #flatten
    reshape = tf.reshape(pool5,[-1,6*6*256])
    dim = reshape.get_shape()[1].value  #获取数据扁平化之后的长度
    
    # 第1层全连接层
    weight1 = tf.Variable(tf.truncated_normal(shape=[dim,4096],stddev=0.04))
    bias1 = tf.Variable(tf.constant(0.1,shape=[4096]))
    f1 = tf.nn.relu(tf.matmul(reshape,weight1) + bias1)
    fc1 = tf.nn.dropout(f1, 0.1,name='fc1')
    parameters += [weight1,bias1]
    print_activations(fc1)
    
    # 第2层全连接层
    weight2 = tf.Variable(tf.truncated_normal(shape=[4096,4096],stddev=0.04))
    bias2 = tf.Variable(tf.constant(0.1,shape=[4096]))
    f2 = tf.nn.relu(tf.matmul(fc1,weight2) + bias2)
    fc2 = tf.nn.dropout(f2, 0.1,name='fc2')
    parameters += [weight2,bias2]
    print_activations(fc2)
    
    # 第3层全连接层
    weight3 = tf.Variable(tf.truncated_normal(shape=[4096,1000],stddev=1/192.0))
    bias3 = tf.Variable(tf.constant(0.0,shape=[1000]))
    fc3 = tf.add(tf.matmul(fc2,weight3),bias3,name='fc3')
    parameters += [weight3,bias3]
    print_activations(fc3)
    softmax = tf.nn.softmax(fc3)
    
    #返回全连接层和softmax
    return fc3,softmax,parameters
    
#计算耗时
def time_tensorflow_run(session,target,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squrared = 0.0
    
    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s:step %d, duration = %.3f'%
                      (datetime.now(),i - num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squrared += duration*duration
    mn = total_duration / num_batchs
    vr = total_duration_squrared / num_batchs - mn*mn
    
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch'%
          (datetime.now(),info_string,num_batchs,mn,sd))


def run_benchmark(name="fc"):
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))        
        pool5,parameters_conv = inference(images)
        fc3,softmax,parameters_fc = fc(pool5)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        print("\nthe %s layers cost:"%(name))
        if name == "conv":
            time_tensorflow_run(sess,pool5,"Forward")
            objective = tf.nn.l2_loss(pool5)
            grad = tf.gradients(objective,parameters_conv)
            time_tensorflow_run(sess,grad,"Forward-backward")
        else:
            time_tensorflow_run(sess,fc3,"Forward")
            objective = tf.nn.l2_loss(fc3)
            grad = tf.gradients(objective,parameters_fc)
            time_tensorflow_run(sess,grad,"Forward-backward")

if __name__ == '__main__':
    run_benchmark()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    