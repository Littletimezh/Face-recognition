# coding: utf-8
# Author：Li

import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import cv2
import numpy as np

image_size = 64  # TFrecords文件的样本数据集的size：64*64; 原图像像素大小为


# 相关路径
# .tfrecords文件的存储路径
tf_save_path = "my_tfrecords\\face_train.tfrecords"
# 样本分类文件夹名字存储labels.txt
labels_path = "labels.txt"
# 样本原图相关路径
image_path = "data_gray\\"
# 反验证tfrecords数据集输出图片路径
tf_out_path = "tf_out\\"


# function:读取样本分类文件夹的名字
print("\n/****************************/")
print("\n  face生成.tfrecords训练数据集 ~~~")
# 读取分类标签文本
f = open(labels_path)  # 通过文本的方式读取分类文件夹名称

class_id_cnt = 0  # 分类计数

# 定义变长数组，保存分类文件夹的名字，python的数组也是从 0 开始的
classes_read = []

print("\n读取样本的分类号：")
while True:
    line = f.readline()
    if line:
        class_id_cnt = class_id_cnt + 1  # 记录分类总数
        line = line.strip()
        classes_read.append(line)  # 给变长数组添加分类的文件夹名字
        print(class_id_cnt, ")", "--", classes_read[class_id_cnt - 1])
    else:
        break
f.close()
print("\n")


# function：生成 TfRecords文件 -- 包括样本及对应的标签号
writer = tf.python_io.TFRecordWriter(tf_save_path)  #
picture_cnt = 0  # 记录样本总量计数

for index, name in enumerate(classes_read):
    class_path = image_path + name + '\\'""  # 每个分类文件夹路径
    print("第 ", index, " 类 开始转换~~~")
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每一个图片的地址
        # opencv 显示
        if (picture_cnt % 60 == 0):  # 每60帧显示1次
            img_cv = cv2.imread(img_path)  # opencv读取图片路径必须全英文
            cv2.namedWindow("image_tfrecords", 0)  # 定义显示窗口
            cv2.imshow("image_tfrecords", img_cv)  # 显示图片
            cv2.waitKey(1)  # 等待1ms，接着继续往下执行
        # 样本数据写入 tfrecords
        img = Image.open(img_path)  # 通过PIL的Image方式打开图片
        img = img.resize((image_size, image_size))  # 图片resize 样本为 28*28
        img_raw = img.tobytes()  # 将28*28图片转化为二进制格式
        # example对象对样本标签和图像数据进行封装，label编码成int64，img_raw编码成2进制
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串,写入tfrecords文件
        picture_cnt = picture_cnt + 1  # 样本计数
writer.close()  # 将所有样本写入tfrecords文件后关闭操作

cv2.destroyAllWindows()  # 销毁opencv显示窗口

print(" TfRecords文件生成 ：样本图总数为 ", picture_cnt)


# function:读取 tfrecords文件功能函数
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),  # 64int
            'img_raw': tf.FixedLenFeature([], tf.string),  # 图片是 string类型
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # 数据解码
    img = tf.reshape(img, [image_size, image_size, 1])  # reshape和原图像素的长、宽、通道一致
    label = tf.cast(features['label'], tf.int64)  # 获取对应的标签
    return img, label


# 加载tfrecords文件并进行文件解析
image2, labels2 = read_and_decode(tf_save_path)


# function:反向验证.tfrecords格式数据集的样本和标签对应关系， 生成对应的图像，并输出tf_out文件夹
with tf.Session() as sess:  # 开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    # 启动QueueRunner
    threads = tf.train.start_queue_runners(coord=coord)

    Classes_Cnt = np.zeros([class_id_cnt], np.int32)  # 记录每个分类的样本个数

    for i in range(picture_cnt):
        example, class_num = sess.run([image2, labels2])  # 在会话中取出image和label
        if i % 30 == 0:  # 每30帧显示1次
            cv2.namedWindow("image_out", 0)
            cv2.imshow("image_out", example)
            cv2.waitKey(1)
        out_file = tf_out_path + str(i) + '_''Label_' + str(class_num) + '.jpg'  # 存储和灰度图像路径
        cv2.imwrite(out_file, example)  # 存储灰度图像
        #         print("分类号 : ",l)
        Classes_Cnt[class_num] = Classes_Cnt[class_num] + 1
    # print(example, l)
    coord.request_stop()
    coord.join(threads)

    # 打印每类样本的样本个数
    for i in range(class_id_cnt):
        print("分类号", i, " = ", Classes_Cnt[i], " 个样本")

    cv2.destroyAllWindows()  # 销毁opencv显示窗口
    sess.close()
    print("\n3）TfRecords测试 转换成功 ！ ")
    print("well done!")
