#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DL_02.py
@Time    :   2019/04/24 11:19:37
@Author  :   Shalor
@Desc    :   1、tensorflow的可视化学习(event序列文件的生成以及tensorboard的使用)
             2、自己生成线性回归的数据，做一个线性回归模型
             3、保存加载模型
             4、自定义命令行参数，指定运行时哪些参数要被自定义
'''

# # Here is the Part1
# import tensorflow as tf
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# # 变量OP
# # 1、变量OP能够持久化保存，普通张量OP是不行的
# # 2、当定义一个变量OP的时候，一定要在会话当中去运行初始化
# # 3、name参数：在tensorboard使用的时候显示名字的，可以将相同OP名字的进行区分
# a = tf.constant(2.0, name="a")
# b = tf.constant(3.0, name="b")
# c = tf.add(a, b, name="add")
# d = tf.add(a, c, name="add2")
# var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0),
#                   name="variable")
# print(a, var)

# # 要想输出变量Variable，必须显示的进行初始化，如果不显示初始化是没用的
# init_op = tf.global_variables_initializer()

# with tf.Session() as sess:
#     # 必须运行初始化OP
#     sess.run(init_op)

#     # 把程序的图结构写入事件文件，graph:把指定的图写进事件文件
#     # 注意到：第一个参数只写到要保存的路径，不涉及文件名
#     filewriter = tf.summary.FileWriter("G://PythonCode//DL_Study/",
#                                        graph=sess.graph)

#     # 在终端中输入 tensorboard --logdir="G://PythonCode//DL_Study/" 即可得到一个地址，在浏览器中打开就行
#     print(sess.run([c, d, var]))

# Here is Part2
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# 定义命令行参数，可以指定程序运行时的参数有哪些
# 1、首先要指定哪些参数需要在运行的时候去指定
# 2、程序当中去获取命令行参数
# 定义时三个参数分别表示：名字+默认值+说明
tf.app.flags.DEFINE_integer("max_step", 1000, "模型训练的步数")
tf.app.flags.DEFINE_string("model_dir", "G:/PythonCode/DL_Study/Model/model", "模型文件的加载路径")

# 定义一个量来接收这些命令行参数
FLAGS = tf.app.flags.FLAGS  # 要使用哪个参数，就直接用FLAGS.max_step/FLAGS.model_dir


def myregression():
    # tf.variable_scope("...")是变量作用域，使得代码可读性更强，更清晰
    with tf.variable_scope("data"):
        # 1、准备数据，100个样本1个特征值，x是[100,1],y是目标值[100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2、建立线性回归模型，1个特征，1个权重，1个偏置
        # 随机给一个权重和偏置的值，让他去计算损失，然后在当前状态下优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0),
                             name="w")
        bias = tf.Variable(0.0, name='b')

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 3、建立损失函数，计算均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 4、梯度下降优化损失，learning_rate不要太大
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 收集要在tensorboard中显示的变量
    tf.summary.scalar("loss", loss)  # 第一个"loss"是在后台中显示的名称，第二个loss是要收集的变量
    tf.summary.histogram("weight", weight)
    # 合并收集的变量
    merge = tf.summary.merge_all()

    # 5、定义一个初始化变量的OP
    init_op = tf.global_variables_initializer()

    # 6、定义保存模型的OP
    saver = tf.train.Saver()

    # 7、通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重：%f, 偏置为:%f" % (weight.eval(), bias.eval()))

        # 建立事件文件，用tensorboard进行观察
        filewriter = tf.summary.FileWriter("G://PythonCode//DL_Study//",
                                           graph=sess.graph)

        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的参数处开始
        if os.path.exists("G:/PythonCode/DL_Study/Model/checkpoint"):
            saver.restore(sess, FLAGS.model_dir)

        # 循环训练运行优化
        for i in range(FLAGS.max_step):
            sess.run(train_op)

            # 每次训练时，都要将收集的变量运行并写入事件文件
            summary = sess.run(merge)
            filewriter.add_summary(summary, i)

            print("第%f次运行后参数权重：%f, 偏置为:%f" % (i, weight.eval(), bias.eval()))

        # 保存模型
        saver.save(sess, FLAGS.model_dir)  # 第一个是保存哪个会话，第二个要精确到文件名，跟事件文件不同

    return None


if __name__ == "__main__":
    myregression()
