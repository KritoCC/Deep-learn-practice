# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:29:40 2020

@author: Krito
"""
import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # 加这一行是为了兼容tensorflow版本1
hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()
print(sess.run(hello))
print(keras.__version__)