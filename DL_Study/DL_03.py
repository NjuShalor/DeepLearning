#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DL_03.py
@Time    :   2019/04/26 14:58:13
@Author  :   Shalor
@Desc    :   1、模拟队列进行读取数据
             2、利用线程，实现异步操作
             3、读取CSV文件
'''

# Here is the code
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# # Part1
# # 模拟一下同步先处理数据，然后才能取数据训练

# # 1、首先定义队列
# Q = tf.FIFOQueue(3, tf.float32) # 队列长为3，数据类型为float32
# # 放入一些数据
# # 注意：如果是[0.1,0.2,0.3]会报错，因为会把这个当做一个tensor来整体处理
# # 所以我们要写成[[0.1,0.2,0.3],]这个形式，这样才会把它当做列表来传入数据
# enq_many = Q.enqueue_many([
#     [0.1, 0.2, 0.3],
# ])

# # 2、定义一些处理数据、取数据的过程--->取出数据，+1，入队列
# out_q = Q.dequeue()  # 从队列中取出一个数据
# data = out_q + 1  # 将这个取出的数据加一
# en_q = Q.enqueue(data)  # 再将这个数据放入队列

# with tf.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)

#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)

#     # 训练数据
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))

# # Part2
# # 模拟异步处理，子线程：存入样本；主线程：读取样本

# # 1、定义一个队列（比如长度为1000）
# Q = tf.FIFOQueue(1000, tf.float32)

# # 2、定义要做的事情，循环，值+1，放入队列
# var = tf.Variable(0.0, tf.float32)
# # 实现一个自增加,tf.assign_add,并且每次的值存入队列
# data = tf.assign_add(var, tf.constant(1.0))
# en_q = Q.enqueue(data)

# # 3、定义队列管理器OP，指定有多少个子线程，子线程该干什么事情
# qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
# # 注意：[op1,op2,...]是指定子线程要进行操作的OP，比如上述[en_q]表示子线程要操作使得数据入队列
# # [op1,op2,...]*2表示有两个子线程

# # 初始化变量的OP
# init_op = tf.global_variables_initializer()

# with tf.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)

#     # 开启线程管理器
#     coord = tf.train.Coordinator()

#     # 真正开启子线程，以上只是定义了子线程的任务，但是要显示开启
#     threads = qr.create_threads(sess, coord=coord, start=True)

#     # 主线程：不断读取数据进行训练
#     for i in range(200):
#         print(sess.run(Q.dequeue()))

#     # 回收子线程
#     coord.request_stop()
#     coord.join(threads)

#     # 注意：如果没有线程管理器，那么等回话结束后，资源会自动释放，
#     # 但是子线程不会结束，因此会报错，所以要添加线程管理器来结束子线程
#     # 输出的结果不是顺序增大的----->是因为线程是由CPU管理的，有随机性

# Here is Part3
# 进行文件读取(csv文件)


def csvread(filelist):
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2、构造csv阅读器读取队列数据（按一行来读）
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)  # key为文件名，value为文件内容

    # 3、对每行的内容进行解码
    # record_defaults:指定样本的每一列的类型以及指定每一列的默认值
    # 比如：[["None"],[4.0]]----->表示两列分别为string和float类型，默认值为None和4.0
    records = [["None"], ["None"]]
    example, label = tf.decode_csv(value, record_defaults=records)

    # 4、想要读取多个数据，就需要批处理
    # batch_size是每批次数据的大小（即每批次数据有多少个），这个是自己定义的，跟原始数据无关
    # capacity是队列长度
    example_batch, label_batch = tf.train.batch([example, label],
                                                batch_size=9,
                                                num_threads=2,
                                                capacity=32)
    print(example_batch, label_batch)

    return example_batch, label_batch


if __name__ == "__main__":
    # 找到文件，放入列表 （注意列表中为：路径+名字 的格式）
    file_name = os.listdir("G:/PythonCode/DL_Study/Testdata")

    file_list = [os.path.join("G:/PythonCode/DL_Study/Testdata", file) for file in file_name]

    # print(file_list)
    example_batch, label_batch = csvread(file_list)

    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取的内容
        print(sess.run([example_batch, label_batch]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)
