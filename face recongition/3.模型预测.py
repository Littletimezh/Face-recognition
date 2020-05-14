# coding: utf-8
# Author：Li
# 实现将已有模型进行加载，且实现对样本数据集 逐一 one by one方式的预测
import tensorflow as tf
import numpy as np
import cv2
import sys



image_size = 64  # 样本 size：28*28
picture_cnt_test = 1710  # 用于预测的样本量

print("/****************************/")
print(" face cnn estimation ~~~")

# In[2]:

# function: 相关路径
train_path = "my_tfrecords\\face_train.tfrecords"  # 测试数据集 .tfrecords路径
meta_path = "face_cnn_model\\face_cnn_model.ckpt.meta"  # 训练好的网络模型结构文件路径
model_param_path = "face_cnn_model\\face_cnn_model.ckpt"  # 训练好的网络模型参数文件路径


# function:加载解码.tfrecords数据集
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # 样本解码
    img = tf.reshape(img, [image_size, image_size, 1])  # 将一个样本 reshape，为单通道 size*size
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 归一化
    label = tf.cast(features['label'], tf.int64)  # 获取labels标签
    return img, label


image_test, labels_test = read_and_decode(train_path)  # 解码tfrecords文件，获取测试数据集

# In[5]:

# function : 加载训练好模型，进行预测测试
print(" cnn 开始模型测试预测~~~~~~ \n")
graph = tf.get_default_graph()
saver = tf.train.import_meta_graph(meta_path)  # 导入模型图结构

with tf.Session() as sess:  # 开始一个会话
    saver.restore(sess, model_param_path)  # 加载模型
    print("\nModel restored.")

    # 加载模型 tensor
    x = graph.get_tensor_by_name('images:0')  # 默认name后加上“:0”，表示第0个接口
    keep_prob = graph.get_tensor_by_name('Fc1/my_keep_prob:0')
    prediction = graph.get_tensor_by_name('Fc2/softmax/my_prediction:0')

    coord = tf.train.Coordinator()  # 协同启动的线程
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)  # 启动线程运行队列

    cnt_corrt = 0.  # 正确预测计数
    cnt_error = 0.  # 错误预测计数

    for i in range(picture_cnt_test):

        example, class_num = sess.run([image_test, labels_test])  # 在会话中取出image和label
        example_test = example.reshape((1, image_size, image_size, 1))  # reshape 为 4D 张量

        pre = sess.run(prediction, feed_dict={x: example_test, keep_prob: 1.0})
        #         print("pre",pre)
        class_pre = sess.run(tf.arg_max(pre, 1))

        if class_pre == class_num:
            cnt_corrt = cnt_corrt + 1.  # 记录预测正确的样本数量
        else:
            cnt_error = cnt_error + 1.  # 记录预测错误的样本数量
        # print("预测与真实结果不符合~~~~~")

        if (i != 0 and i % 100 == 0):
            print(i, ")准确率 = ", cnt_corrt / (i + 1), "(预测,真实)=(", class_pre, class_num, ")", "错误个数：", cnt_error, "\n")

            str_i = str(class_pre)
            example_show = np.reshape(example, [image_size, image_size, 1])  # 用于opencv绘图
            cv2.putText(example_show, str_i, (0, 8), 0, 0.25, (100, 100, 190), 1)  # 显示预测的标签号在图片上
            cv2.namedWindow("image", 1)  # 定义显示窗口
            cv2.imshow("image", example_show)  # 显示
            cv2.waitKey(1)  # 等待 1 ms

    print("\n模型预测测试准确率 = ", cnt_corrt / (picture_cnt_test))

    cv2.destroyAllWindows()  # 销毁opencv窗口
    coord.request_stop()  # 关闭线程
    coord.join(threads)
    sess.close()
print("\n cnn 模型预测样本数据测试成功 ！ ")




