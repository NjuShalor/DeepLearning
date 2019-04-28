#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DL_01.py
@Time    :   2019/04/23 20:59:52
@Author  :   Shalor
@Desc    :   tensorflow的基础操作和入门知识
'''

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 这个是消除警告，警告提示我们用TensorFlow源码可以提高运行效率

# Here is the Part1
"""
# 实现一个加法运算
a = tf.constant(2)
b = tf.constant(3)

sum1 = tf.add(a, b)

var1 = 2
var2 = 3
var3 = var1 + var2  # 注意var3只是一个普通的int变量，是python中的，它在会话中不能run()
var4 = a + var1  # var4和var4都可以在会话中run()，因为+被tensorflow重载了
var5 = var1 + a

# 自己可以定义一个图
# 注意自己定义个图g和下面的sess分配的内存是互不干扰的
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11)
    print(c.graph)

# 默认的这张图，相当于给程序分配一段内存
graph = tf.get_default_graph()

print(graph)

print(sum1)

# with tf.Session() as sess:是上下文管理器，调用完了之后自动调用sess.close()，不用显示调用
# with tf.Session(
#         graph=g
# ) as sess:  # 注意Session(graph = g)中可以指定开启哪张图的对话，比如开启g之后，可以print(sess.run(c))
#     # print(sess.run(sum1))
#     print(sess.run(c))
#     print(a.graph)
#     print(sum1.graph)
#     print(sess.graph)

# tf.placehoder()是一个占位操作，训练时实时传入数据，用feed_dict进行键值传入
plt = tf.placeholder(tf.float32, [2, 3])
plt2 = tf.placeholder(tf.float32, [None, 3])  # 注意，如果用None表示，表名不确定，实时传入时可以任意行

# config=tf.ConfigProto(log_device_placement=True)加上这个参数后，会输出这个图里所有的OP信息
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum1))
    print(sess.run(var4))
    print(sess.run(var5))
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)
    print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6]]}))
    print(sess.run(plt2, feed_dict={plt2: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
    print(a.graph) # 输出a的图信息
    print(a.shape) # 输出a的形状
    print(a.op) # 输出a节点信息
"""

# Here is Part2

# tensor的形状：0维：(), 1维:(3), 2维:(2,3), 3维:(2,3,4)
# tensor形状的改变分为两种:1、静态形状改变(直接改变原数据形状，不生成新的tensor)；2、动态形状改变(生成新的tensor)
# 对于静态形状来说，一旦形状确定后，就不能修改形状了，但是动态形状可以
# 静态形状修改不能跨维度修改，只能同维度修改，动态形状可以

plt = tf.placeholder(tf.float32, [None, 3])
print(plt)  # 不确定的维度数会输出?
plt.set_shape([3, 3])
print(plt)
# plt.set_shape([9,1]) # 这个操作会报错，因为确定形状后不能再次改变了
plt_reshape = tf.reshape(plt, [9, 1, 1])  # 注意：动态形状改变时，元素数量要匹配
print(plt_reshape)

# 设置一定形状的常量时，可以用ones和zeros来生成固定值的常量，也可以用正态分布来生成数据
ones = tf.ones([4, 5], tf.float32)  # 生成全是1的4*5的tensor
print(ones)
zeros = tf.zeros([4, 5], tf.float32)  # 生成全是0的4*5的tensor
print(zeros)
# tf.random_normal是生成正态分布的数据，mean,stddev分别是正态分布的均值和方差，不写默认为0和1
random1 = tf.random_normal([2, 3], mean=0, stddev=1.0, dtype=tf.float32)
print(random1)
# tf.truncated_normal是截断的正态分布，只返回两个方差之内的数据
random2 = tf.truncated_normal([2, 3])
print(random2)

# tf.concat可以合并两个tensor
c1 = tf.concat([ones, zeros], axis=0)  # axis=0是按行拼接，行增多
c2 = tf.concat([ones, zeros], axis=1)  # axis=1是按列拼接，列增多

# tf.cast()是一个万能的转换类型的API
before = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
now = tf.cast(before, dtype=tf.float32)

with tf.Session() as sess:
    print(ones.eval())
    print(zeros.eval())
    print(random1.eval())
    print(random2.eval())
    print(c1.eval())
    print(c2.eval())
