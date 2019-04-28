#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DL_04.py
@Time    :   2019/04/27 15:39:29
@Author  :   Shalor
@Desc    :   1、图片文件的读取
             2、二进制文件的读取，以及将数据存进tfrecords
             3、读取tfrecords文件并且解析、解码
'''

# Here is the code
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# # Here is Part1
# # 读取图片文件
# def picread(filelist):
#     # 1、构造文件队列
#     file_queue = tf.train.string_input_producer(filelist)

#     # 2、构造图片阅读器(默认按照一张进行读取)
#     reader = tf.WholeFileReader()

#     key, value = reader.read(file_queue)
#     # print(value)此时输出的图片信息，shape=()，什么都没有，因此要进行解码

#     # 3、对读取的图片数据进行解码
#     image = tf.image.decode_png(value)  # 解码后是uint8类型
#     print(image)

#     # 4、对图片大小进行处理（处理成同一大小）
#     image_resize = tf.image.resize_images(image,
#                                           [1080, 1920])  # 转换大小后变成float32类型
#     print(image_resize)  # 此时通道数还未固定，显示？
#     # 注意：一定要将图片形状进行固定，这样才能进行批处理
#     image_resize.set_shape([1080, 1920, 3])

#     # 5、进行批处理
#     image_batch = tf.train.batch([image_resize],
#                                  batch_size=16,
#                                  num_threads=1,
#                                  capacity=16)

#     print(image_batch)

#     return None

# if __name__ == "__main__":
#     # 1、找到文件，放入列表，格式为：路径名+文件名
#     file_name = os.listdir("D://nju//dota2//DOTA2_PNG//")

#     filelist = [
#         os.path.join("D://nju//dota2//DOTA2_PNG//", file) for file in file_name
#     ]

#     # print(filelist)
#     picread(filelist)

# Here is Part2
# 二进制文件读取

# 定义cifar的数据文件地址等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir",
                           "G://E//testdata//cifar-10-batches-bin//", "文件的目录")

# 注意这个是存进tfrecords文件的文件名（路径+文件名）
tf.app.flags.DEFINE_string("tfrecords_dir", "./cifar.tfrecords",
                           "存进tfrecords的文件")


class CifarRead(object):
    """
    完成二进制文件的读取并写入tfrecords，读取tfrecords
    """

    def __init__(self, filelist):
        self.filelist = filelist
        # 定义读取的图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        # 二进制下每张图片的的字节数
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer(self.filelist)

        # 2、构造二进制文件读取器，读取内容，
        reader = tf.FixedLengthRecordReader(self.bytes)  # 可以直接写成3073
        key, value = reader.read(file_queue)

        # 3、解码内容
        label_image = tf.decode_raw(value, tf.uint8)  # 注意：解码出来的是label+image

        # 分割图片和标签数据
        # 可以将label值直接转换成int32型的
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]),
                        tf.int32)
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])
        # print(label, image)

        # 5、可以将图片的特征数据进行形状的改变:[3072]--->[32,32,3]
        image_reshape = tf.reshape(image,
                                   [self.height, self.width, self.channel])

        # print(label, image_reshape)

        # 6、对数据进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label],
                                                  batch_size=16,
                                                  num_threads=1,
                                                  capacity=16)

        return image_batch, label_batch

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值和目标值存进tfrecords
        """
        # 1、构造tfrecords存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_dir)

        # 2、循环的将所有样本写入文件，每张图片样本都要构造example协议
        for i in range(10):
            # 取出第i个样本的特征值和目标值
            image = image_batch[i].eval().tostring(
            )  # 将tensor先转成值然后在转成string类型
            label = int(
                label_batch[i].eval()[0])  # label_batch是一个二维tensor，因此需要索引切片

            # 构造一个样本的example协议块
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "image":
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[image])),
                    "label":
                    tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[label]))
                }))

            # 写入单独的样本
            writer.write(example.SerializeToString()
                         )  # 注意：example是类字典格式，不能直接写入文件，要用API进行转换

        # 关闭文件
        writer.close()

        return None

    def read_from_tfrecords(self):
        """
        读取tfrecords文件
        """
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.tfrecords_dir])

        # 2、构造文件阅读器，读取内容example,value=一个样本的序列化example
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        # 3、解析example,注意这里只有tf.int64类型，没有tf.int32类型
        features = tf.parse_single_example(value,
                                           features={
                                               "image":
                                               tf.FixedLenFeature([],
                                                                  tf.string),
                                               "label":
                                               tf.FixedLenFeature([], tf.int64)
                                           })
        # print(features["image"], features["label"])

        # 4、解码内容:如果读取的内容格式是string需要解码，如果是int64,float32则不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)
        label = features["label"]
        # print(image, label)

        # 5、固定image的形状，进行批处理
        image_reshape = tf.reshape(image,
                                   [self.height, self.width, self.channel])

        image_batch, label_batch = tf.train.batch([image_reshape, label],
                                                  batch_size=16,
                                                  num_threads=1,
                                                  capacity=16)

        return image_batch, label_batch


if __name__ == "__main__":
    # 1、找到文件，放入列表
    file_name = os.listdir(FLAGS.cifar_dir)
    filelist = [
        os.path.join(FLAGS.cifar_dir, file) for file in file_name
        if file[-6] == "_"
    ]

    # 实例化对象
    cf = CifarRead(filelist)

    # image_batch, label_batch = cf.read_and_decode()

    image_batch, label_batch = cf.read_from_tfrecords()

    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取的内容
        print(sess.run([image_batch, label_batch]))

        # # 将数据写入tfrecords文件
        # print("开始写入")
        # cf.write_to_tfrecords(image_batch, label_batch)
        # print("结束写入")

        # 回收子线程
        coord.request_stop()
        coord.join(threads)
