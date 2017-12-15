# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:10:38 2017

@author: Reader
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datetime import datetime
import math
import time
import tensorflow as tf

def print_activations(t):
        print(t.op.name,' ',t.get_shape().as_list())

#卷积层创建函数
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    #得到输入的通道数
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        #初始化conv层kernel
        kernel = tf.get_variable(scope+'w',
           shape=[kw,kh,n_in,n_out],dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),
                            padding='SAME')
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable=True,name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        print_activations(activation)
        return activation
#全连接层创建函数
def fc_op(input_op,name,n_out,p):
    #得到输入的通道数
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
            shape=[n_in,n_out],dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],
                                         dtype=tf.float32),name='b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p += [kernel,biases]
        print_activations(activation)
        return activation
        

#池化层创建函数        
def mpool_op(input_op,name,kh,kw,dh,dw):
    pool = tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],
                          strides=[1,dh,dw,1],
                          padding='SAME',
                          name=name)
    print_activations(pool)
    return pool

def inference_op(input_op,keep_prob):
    #保存的参数
    p = []
    
    #第一段卷积网络
    conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,
                      dw=1,p=p)
    conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,
                      dw=1,p=p)
    pool1 = mpool_op(conv1_2,name='pool1',kh=2,kw=2,dh=2,dw=2)

    #第二段卷积网络
    conv2_1 = conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,
                      dw=1,p=p)
    conv2_2 = conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,
                      dw=1,p=p)
    pool2 = mpool_op(conv2_2,name='pool2',kh=2,kw=2,dh=2,dw=2)

    #第三段卷积网络
    conv3_1 = conv_op(pool2,name='conv3_1',kh=3,kw=3,n_out=256,dh=1,
                      dw=1,p=p)
    conv3_2 = conv_op(conv3_1,name='conv3_2',kh=3,kw=3,n_out=256,dh=1,
                      dw=1,p=p)
    conv3_3 = conv_op(conv3_2,name='conv3_3',kh=3,kw=3,n_out=256,dh=1,
                      dw=1,p=p)
    pool3 = mpool_op(conv3_3,name='pool3',kh=2,kw=2,dh=2,dw=2)
    
    #第四段卷积网络
    conv4_1 = conv_op(pool3,name='conv4_1',kh=3,kw=3,n_out=512,dh=1,
                      dw=1,p=p)
    conv4_2 = conv_op(conv4_1,name='conv4_2',kh=3,kw=3,n_out=512,dh=1,
                      dw=1,p=p)
    conv4_3 = conv_op(conv4_2,name='conv4_3',kh=3,kw=3,n_out=512,dh=1,
                      dw=1,p=p)
    pool4 = mpool_op(conv4_3,name='pool4',kh=2,kw=2,dh=2,dw=2)

    #第五段卷积网络
    conv5_1 = conv_op(pool4,name='conv5_1',kh=3,kw=3,n_out=512,dh=1,
                      dw=1,p=p)
    conv5_2 = conv_op(conv5_1,name='conv5_2',kh=3,kw=3,n_out=512,dh=1,
                      dw=1,p=p)
    conv5_3 = conv_op(conv5_2,name='conv5_3',kh=3,kw=3,n_out=512,dh=1,
                      dw=1,p=p)
    pool5 = mpool_op(conv5_3,name='pool5',kh=2,kw=2,dh=2,dw=2)

    #扁平化
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name='resh1')
    
    #全连接层 - 1
    fc6 = fc_op(resh1,name='fc6',n_out=4096,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
    
    #全连接层 - 2
    fc7 = fc_op(fc6_drop,name='fc7',n_out=4096,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name='fc7_drop')

    #输出层
    fc8 = fc_op(fc7_drop,name='fc8',n_out=1000,p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    
    return predictions,softmax,fc8,p

#计算耗时
def time_tensorflow_run(session,target,feed,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squrared = 0.0
    
    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        session.run(target,feed_dict=feed)
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
        keep_prob = tf.placeholder(tf.float32)
        prediction,softmax,fc8,p = inference_op(images,keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,prediction,{keep_prob:1.0},"Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective,p)
        time_tensorflow_run(sess,grad,{keep_prob:0.5},"Forward-backward")

if __name__ == "__main__":
    batch_size = 32
    num_batchs = 100
    run_benchmark()
















































    