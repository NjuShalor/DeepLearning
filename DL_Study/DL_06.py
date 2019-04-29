#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DL_06.py
@Time    :   2019/04/29 16:10:07
@Author  :   Shalor
@Desc    :   用卷积神经网络进行手写体识别
'''

# Here is the code
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# 直接调用GoogleNet的接口
# from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.Variable(
        tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float32))
    return b


def model():
    """
    自定义卷积模型
    """
    # 1、建立数据的占位符 x=[None,784],y_true=[None,10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])  # 占位符中不确定的用None
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2、第一层卷积层:卷积:5*5*1,32个filter,stride=1、激活:relu、池化:2*2 strides=2
    with tf.variable_scope("conv1"):
        # 随机初始化权重,初始化偏置[32]
        w_conv1 = weight_variables([5, 5, 1, 32
                                    ])  # 这是filter的参数，大小为[5,5],1个通道数，32个filter
        b_conv1 = bias_variables([32])

        # 对x的形状进行改变[None,784]--->[None,28,28,1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])  # 注意，在reshape中不确定的用-1

        # 卷积操作 [None,28,28,1]--->[None,28,28,32]
        x_conv1 = tf.nn.conv2d(
            x_reshape, w_conv1, [1, 1, 1, 1], padding="SAME") + b_conv1
        x_relu1 = tf.nn.relu(x_conv1)

        # 池化操作:2*2,strides=2,[None,28,28,32]--->[None,14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME")

    # 3、第二层卷积层:卷积:5*5*32,64个filter,stride=1、激活:relu、池化:2*2 strides=2
    with tf.variable_scope("conv2"):
        # 随机初始化权重,初始化偏置[64]
        w_conv2 = weight_variables([5, 5, 32, 64
                                    ])  # 这是filter的参数，大小为[5,5],1个通道数，32个filter
        b_conv2 = bias_variables([64])

        # 卷积操作 [None,14,14,32]--->[None,14,14,64]
        x_conv2 = tf.nn.conv2d(x_pool1, w_conv2, [1, 1, 1, 1],
                               padding="SAME") + b_conv2
        x_relu2 = tf.nn.relu(x_conv2)

        # 池化操作:2*2,strides=2,[None,14,14,64]--->[None,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME")

    # 4、全连接层:[None,7,7,64]--->[None,7*7*64]*[7*7*64,10]+[10]=[None,10]
    with tf.variable_scope("fc"):
        # 随机初始化权重和偏置
        w_fc = weight_variables([7 * 7 * 64, 10])
        b_fc = bias_variables([10])

        # 将输出形状进行改变
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行矩阵计算得出每个样本的取值
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def conv_fc():
    # 1、获取真实的数据
    mnist = input_data.read_data_sets("G://PythonCode//DL_Study//Testdata//",
                                      one_hot=True)

    # 2、调用自定义的卷积模型
    x, y_true, y_predict = model()

    # 3、进行交叉熵损失计算
    with tf.variable_scope("soft_cross"):
        # 计算平均交叉熵损失
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                    logits=y_predict))

    # 4、梯度下降让损失最小
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 5、计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义初始化变量OP
    init_op = tf.global_variables_initializer()

    # 开启会话进行运行
    with tf.Session() as sess:
        # 运行初始化变量OP
        sess.run(init_op)

        # 循环训练
        for i in range(1000):
            # 取出真实的目标值和特征值
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 运行train_op训练
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            # # 写入每步运行的值
            # summary = sess.run(merged,
            #                     feed_dict={
            #                         x: mnist_x,
            #                         y_true: mnist_y
            #                     })
            # filewriter.add_summary(summary, i)

            print(
                "训练第%d步，准确率为:%f" %
                (i, sess.run(accuracy, feed_dict={
                    x: mnist_x,
                    y_true: mnist_y
                })))

    return None


if __name__ == "__main__":
    conv_fc()
