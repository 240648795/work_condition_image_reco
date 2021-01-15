# -*- coding : utf-8-*-
import tensorflow as tf
import numpy as np


if __name__ == '__main__':

    # 输入张量为3×3的二维矩阵
    M = np.array([
        [[1], [-1], [0]],
        [[-1], [2], [1]],
        [[0], [2], [-2]]
    ])
    # 定义卷积核权重和偏置项。由权重可知我们只定义了一个2×2×1的卷积核
    filter_weight = tf.Variable('weights', [2, 2, 1, 1], initializer=tf.constant_initializer([
        [1, -1],
        [0, 2]]))
    biases = tf.Variable('biases', [1], initializer=tf.constant_initializer(1))

    # 调整输入格式符合TensorFlow要求
    M = np.asarray(M, dtype='float32')
    M = M.reshape(1, 3, 3, 1)

    # 计算输入张量通过卷积核和池化滤波器计算后的结果
    x = tf.placeholder('float32', [1, None, None, 1])

    # 我们使用了带Padding，步幅为2的卷积操作，因为filter_weight的深度确定了卷积核的数量
    conv = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, biases)

    # 使用带Padding，步幅为2的平均池化操作
    pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 执行计算图
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        convoluted_M = sess.run(bias, feed_dict={x: M})
        pooled_M = sess.run(pool, feed_dict={x: M})

        print("convoluted_M: \n", convoluted_M)
        print("pooled_M: \n", pooled_M)