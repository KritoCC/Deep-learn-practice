# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:47:48 2020

@author: Krito
"""

import numpy as np

x = np.array(12)
x
x.ndim

a = np.array([2, 3, 4])
c = np.array([[1.0, 2.0], [3.0, 4.0]])
e = np.array([[[5, 2, 3, 4],
               [4, 1, 3, 4],
               [2, 2, 2, 4]],
              [[5, 5, 5, 4],
               [5, 4, 4, 4],
               [5, 5, 5, 4]],
              [[2, 1, 1, 4],
               [5, 24, 14, 4],
               [5, 52, 51, 4]]])
e.ndim
# ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度。


arr1 = np.array([2, 3])
arr2 = np.array([4, 5])
np.dot(arr1, arr2)

arr3 = np.array([2, 3, 4])
arr4 = np.array([5, 6, 7])
np.dot(arr3, arr4)

# 广播
x = np.full((32, 1), 10)
# 两种创建（1，）Numpy向量的方法,直接使用np.array（10）无法实现
# 法1
y = [10]
y = np.array(y)


# # 法2
# y = np.full((1,), 10)

def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


naive_add_matrix_and_vector(x, y)
# python assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。
# 可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。
# x.shape以元组形式，返回数组的维数

# copy的用法
x = {
    'usrname': 'admin',
    'machines': [1, 2, 3]
}
y = x.copy();
print(y)
print(x)
print('\n')
y['machines'].remove(3)
print(y)
print(x)
print('\n')
y['usrname'] = 'test'
y['machines'] = [3, 4, 5]
print(y)
print(x)
# 当x使用copy()方法后，只是将x中的key创建了一个副本给y。
# 但是x中的value并没有进行副本的创建，所以y中与x中的变量名相同的key，其实指向同一个vaule的地址。
# 这样当使用赋值的方法
# y[‘usrname’] = ‘test’
# 就会改变y中key指向的地址，但是x中key指向的地址不变，因为x与y中的key并不是同一个量。
# 当使用修改的方法情况下
# y[‘machines’].remove(3)
# y中key与x中的key指向同一个地址，这种情况下，修改y中lkey所指向value就会引起x中同名的key的vaule的变化。
# Python中如果想要需要创建key和value的副本，需要使用deepcopy()方法

# 逐元素的maximum运算应用于两个形状不同的张量
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)

# 张量点积
# 两个向量之间的点积是一个标量
x = np.full((10,), 10)
y = np.full((10,), 10)


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


naive_vector_dot(x, y)

# 一个矩阵x和一个向量y做点积，返回值是一个向量
x = np.full((32, 10), 10)
y = np.full((10,), 10)


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


naive_matrix_vector_dot(x, y)


def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)  # 复用前面写过的代码，从中看出矩阵-向量点积与向量点积之间的关系
    return z


naive_matrix_vector_dot(x, y)

# 两个矩阵之间的点积
x = np.full((32, 32), 10)
y = np.full((32, 32), 10)


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):  # 遍历x的所有行
        for j in range(y.shape[1]):  # 遍历y的所有列
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] += naive_vector_dot(row_x, column_y)
    return z


naive_matrix_dot(x, y)

# 张量变形
x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)
x = x.reshape((6, 1))
x

x = x.reshape(2, 3)
x
# 转置
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:28:38 2020

@author: Krito
"""
import keras
from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print(np.__doc__) 查看模块的作用说明、简介
# help(imdb.load_data) 查看某个函数的用法
train_data[0]  # 序列最开始的的数字1不表示任何意义，数字2表示已经超出我们限定的词语范围
train_labels[0]
max([max(sequence) for sequence in train_data])

word_index = imdb.get_word_index()
# 将字典中的key和value颠倒
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])# .items() 将字典中的key和value都列出来
# 解码评论。 请注意，索引偏移了3，因为0、1和2是“填充”，“序列开始”和“未知”的保留索引,因此需要减去3
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]) #join 用于字符串连接；get() 函数返回指定键的值，如果值不在字典中返回默认值


# 采用one-hot编码，将其转换为0和1的向量。‘存在的数对应1，不存在的数对应0’
# 例如，这将意味着将序列[3，5]转换为一个10,000维向量，除了索引3和5（是1）之外，所有向量均为0。
# 然后，您可以将能够处理浮点矢量数据的dense层用作网络中的第一层。
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = vectorize_sequences(train_labels).astype('float32')
y_test = vectorize_sequences(test_labels).astype('float32')