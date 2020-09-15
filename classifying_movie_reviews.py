# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:28:38 2020

@author: Krito
"""
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import copy


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
        # print(i, sequences)
        # print(results)
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss = losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 'bo' is for 'blue dot'
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Valdation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Valdation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()

acc

a = np.mean(acc)


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,batch_size=512)
results = model.evaluate(x_test, y_test)
results

model.predict(x_test)

# 混淆矩阵
# 混淆矩阵是数据科学、数据分析和机器学习中总结分类模型预测结果的情形分析表，
# 以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总。
# 以二元分类问题为例，数据集存在肯定类别和否定类别两类记录，而分类模型对记录分类可能作出阳性判断（判断记录属于肯定类别）或阴性判断（判断记录属于否定类别）两种判断。
# 混淆矩阵是一个2 × 2的情形分析表，显示以下四组记录的数目：作出正确判断的肯定记录（真阳性TP）、作出错误判断的肯定记录（假阴性FN）、作出正确判断的否定记录（真阴性TN）以及作出错误判断的否定记录（假阳性FP）。

# 准确率precision = TP/(TP+FP) = A
# 召回率recall = TP/(TP+FN) = B
# 调和平均f1-score = 1/((1/A)+(1/B))/2

prediction = model.predict_classes(x_test)
prediction.shape
prediction

prediction.reshape(-1) # 从（25000，1）的二维张量变为（25000，）的一维张量
prediction.reshape(-1).shape
y_test.shape
print(y_test, prediction.reshape(-1))

# 建立混淆矩阵
pd.crosstab(y_test, prediction.reshape(-1),
            rownames=['label'], colnames=['predict'])

# 计算准确率、召回率以及调和平均值
print(classification_report(y_test,prediction))
help(classification_report)


# 如果随机去猜，看一下正确率是多少
test_copy = copy.copy(y_test)
np.random.shuffle(test_copy) # 顺序打乱
float(np.sum(np.array(y_test) == np.array(test_copy))) / len(y_test)
