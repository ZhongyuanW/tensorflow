# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:36:09 2017

@author: Reader
"""

import numpy as np
import tensorflow as tf

def xavier_init(fan_in,fan_out,constant = 1):
    #fan_in:输入结点的数量
    #fan_out:输出结点的数量
    low = -constant * np.sqrt(6.0 / (fan_in+fan_out))
    high = constant * np.sqrt(6.0 / (fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,
                             maxval=high,dtype=tf.float32)

class AdditiveGassianNosieAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),scale=0.1):
        #n_input:输入变量数
        #n_hidden：隐含层节点数
        #transfer_function:隐含层激活函数，默认为softplus
        #optimzer:优化器，默认为Adam
        #scale:高斯噪声系数，默认为0.1
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.train_scale = scale
        
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                            self.x + scale * tf.random_normal((n_input,)),
                            self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                  self.weights['w2']),self.weights['b2'])
        
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                              self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _initialize_weights(self):
        #初始化权重
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                  dtype=tf.float32))
        all_weights['w2'] = tf.Variable(xavier_init(self.n_hidden,
                                                    self.n_input))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                  dtype=tf.float32))
        return all_weights
        
    def partial_fit(self,X):
        #得到损失值并且训练模型（训练阶段）
        cost,opt = self.sess.run((self.cost,self.optimizer),
             feed_dict = {self.x:X,self.scale:self.train_scale})
        return cost
    
    def calc_total_cost(self,X):
        #得到损失值（测试阶段）
        return self.sess.run(self.cost,feed_dict = {self.x:X,
             self.scale:self.train_scale
        })
    
    def transform(self,X):
        #得到隐含层的特征
        return self.sess.run(self.hidden,feed_dict = {self.x:X,
             self.scale:self.train_scale
        })
        
    def generate(self,hidden = None):
        #自定义隐含层的特征，然后复原的原始数据
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict = {self.hidden:hidden})
        
    def reconstruct(self,X):
        #整体运行一次一遍吗器，包含了transfer和generate两个模块
        return self.sess.run(self.reconstruct,feed_dict={self.x:X,
             self.scale:self.train_scale
        })
        
    def getWeights(self):
        #得到隐含层w1的权重
        return self.sess.run(self.weights['w1'])
    def getBiases(self):
        #得到隐含层b1的系数
        return self.sess.run(self.weights['b1'])
    
    