import tensorflow as tf
from tensorflow import Tensor


class Inputs(object):
    def __init__(self, img1: Tensor, img2: Tensor, img3: Tensor):
        self.img1 = img1
        self.img2 = img2
        self.img3 = img3

class Model(object):
    def __init__(self, inputs: Inputs):
        # self.inputs = inputs
        self.anchor_embeddings, self.positive_embeddings, self.negative_embeddings = self.embed(inputs)
        self.loss = self.calculate_loss(self.anchor_embeddings, self.positive_embeddings, self.negative_embeddings)
        self.opt_step = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)

    def embed(self, inputs: Inputs):
        self.w1 = tf.Variable(tf.truncated_normal([3, 3, 3, 20], stddev = 0.1))
        self.b1 = tf.Variable(tf.constant(0.0, shape = [20]))
        anchor_1 = tf.nn.relu(tf.nn.conv2d(inputs.img1/255.0, self.w1, strides = [1, 1, 1, 1], padding = 'SAME') + self.b1)
        positive_1 = tf.nn.relu(tf.nn.conv2d(inputs.img2/255.0, self.w1, strides = [1, 1, 1, 1], padding = 'SAME') + self.b1)
        negative_1 = tf.nn.relu(tf.nn.conv2d(inputs.img3/255.0, self.w1, strides = [1, 1, 1, 1], padding = 'SAME') + self.b1)

        self.w2 = tf.Variable(tf.truncated_normal([3, 3, 20, 40], stddev = 0.1))
        self.b2 = tf.Variable(tf.constant(0.0, shape = [40]))
        anchor_2 = tf.nn.relu(tf.nn.conv2d(anchor_1, self.w2, strides = [1, 1, 1, 1], padding = 'SAME') + self.b2)
        positive_2 = tf.nn.relu(tf.nn.conv2d(positive_1, self.w2, strides = [1, 1, 1, 1], padding = 'SAME') + self.b2)
        negative_2 = tf.nn.relu(tf.nn.conv2d(negative_1 , self.w2, strides = [1, 1, 1, 1], padding = 'SAME') + self.b2)

        anchor_2_max = tf.nn.max_pool(anchor_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        positive_2_max = tf.nn.max_pool(positive_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        negative_2_max = tf.nn.max_pool(negative_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self.w3 = tf.Variable(tf.truncated_normal([3, 3, 40, 60], stddev = 0.1))
        self.b3 = tf.Variable(tf.constant(0.0, shape = [60]))
        anchor_3 = tf.nn.relu(tf.nn.conv2d(anchor_2_max, self.w3, strides = [1, 1, 1, 1], padding = 'SAME') + self.b3)
        positive_3 = tf.nn.relu(tf.nn.conv2d(positive_2_max, self.w3, strides = [1, 1, 1, 1], padding = 'SAME') + self.b3)
        negative_3 = tf.nn.relu(tf.nn.conv2d(negative_2_max , self.w3, strides = [1, 1, 1, 1], padding = 'SAME') + self.b3)

        self.w4 = tf.Variable(tf.truncated_normal([3, 3, 60, 80], stddev = 0.1))
        self.b4 = tf.Variable(tf.constant(0.0, shape = [80]))
        anchor_4 = tf.nn.relu(tf.nn.conv2d(anchor_3, self.w4, strides = [1, 1, 1, 1], padding = 'SAME') + self.b4)
        positive_4 = tf.nn.relu(tf.nn.conv2d(positive_3, self.w4, strides = [1, 1, 1, 1], padding = 'SAME') + self.b4)
        negative_4 = tf.nn.relu(tf.nn.conv2d(negative_3 , self.w4, strides = [1, 1, 1, 1], padding = 'SAME') + self.b4)

        anchor_4_max = tf.nn.max_pool(anchor_4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        positive_4_max = tf.nn.max_pool(positive_4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        negative_4_max = tf.nn.max_pool(negative_4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self.w5 = tf.Variable(tf.truncated_normal([3, 3, 80, 100], stddev = 0.1))
        self.b5 = tf.Variable(tf.constant(0.0, shape = [100]))
        anchor_5 = tf.nn.relu(tf.nn.conv2d(anchor_4_max, self.w5, strides = [1, 1, 1, 1], padding = 'SAME') + self.b5)
        positive_5 = tf.nn.relu(tf.nn.conv2d(positive_4_max, self.w5, strides = [1, 1, 1, 1], padding = 'SAME') + self.b5)
        negative_5 = tf.nn.relu(tf.nn.conv2d(negative_4_max , self.w5, strides = [1, 1, 1, 1], padding = 'SAME') + self.b5)

        self.w6 = tf.Variable(tf.truncated_normal([3, 3, 100, 100], stddev = 0.1))
        self.b6 = tf.Variable(tf.constant(0.0, shape = [100]))
        anchor_6 = tf.nn.relu(tf.nn.conv2d(anchor_5, self.w6, strides = [1, 1, 1, 1], padding = 'SAME') + self.b6)
        positive_6 = tf.nn.relu(tf.nn.conv2d(positive_5, self.w6, strides = [1, 1, 1, 1], padding = 'SAME') + self.b6)
        negative_6 = tf.nn.relu(tf.nn.conv2d(negative_5 , self.w6, strides = [1, 1, 1, 1], padding = 'SAME') + self.b6)

        anchor_6_max = tf.nn.max_pool(anchor_6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        positive_6_max = tf.nn.max_pool(positive_6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        negative_6_max = tf.nn.max_pool(negative_6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self.w7 = tf.Variable(tf.truncated_normal([3, 3, 100, 100], stddev = 0.1))
        self.b7 = tf.Variable(tf.constant(0.0, shape = [100]))
        anchor_7 = tf.nn.relu(tf.nn.conv2d(anchor_6_max, self.w7, strides = [1, 1, 1, 1], padding = 'SAME') + self.b7)
        positive_7 = tf.nn.relu(tf.nn.conv2d(positive_6_max, self.w7, strides = [1, 1, 1, 1], padding = 'SAME') + self.b7)
        negative_7 = tf.nn.relu(tf.nn.conv2d(negative_6_max , self.w7, strides = [1, 1, 1, 1], padding = 'SAME') + self.b7)

        self.w8 = tf.Variable(tf.truncated_normal([3, 3, 100, 100], stddev = 0.1))
        self.b8 = tf.Variable(tf.constant(0.0, shape = [100]))
        anchor_8 = tf.nn.relu(tf.nn.conv2d(anchor_7, self.w8, strides = [1, 1, 1, 1], padding = 'SAME') + self.b8)
        positive_8 = tf.nn.relu(tf.nn.conv2d(positive_7, self.w8, strides = [1, 1, 1, 1], padding = 'SAME') + self.b8)
        negative_8 = tf.nn.relu(tf.nn.conv2d(negative_7 , self.w8, strides = [1, 1, 1, 1], padding = 'SAME') + self.b8)

        anchor_8_max = tf.nn.max_pool(anchor_8, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        positive_8_max = tf.nn.max_pool(positive_8, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        negative_8_max = tf.nn.max_pool(negative_8, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        anchor_8_max_flat = tf.reshape(anchor_8_max, [-1, 8 * 8 * 100])  # (., 8*8*100)
        positive_8_max_flat = tf.reshape(positive_8_max, [-1, 8 * 8 * 100])  # (., 8*8*100)
        negative_8_max_flat = tf.reshape(negative_8_max, [-1, 8 * 8 * 100])  # (., 8*8*100)

        self.w9 = tf.Variable(tf.truncated_normal([8*8*100, 2000], stddev = 0.1))
        self.b9 = tf.Variable(tf.constant(0.0, shape = [2000]))
        anchor_9 = tf.nn.relu(tf.matmul(anchor_8_max_flat, self.w9) + self.b9) # (.,2000)
        positive_9 = tf.nn.relu(tf.matmul(positive_8_max_flat, self.w9) + self.b9)  # (.,2000)
        negative_9 = tf.nn.relu(tf.matmul(negative_8_max_flat, self.w9) + self.b9)  # (.,2000)

        self.w10 = tf.Variable(tf.truncated_normal([2000, 1000], stddev = 0.1))
        self.b10 = tf.Variable(tf.constant(0.0, shape = [1000]))
        anchor_10 = tf.nn.relu(tf.matmul(anchor_9, self.w10) + self.b10) # (.,1000)
        positive_10 = tf.nn.relu(tf.matmul(positive_9, self.w10) + self.b10)  # (.,1000)
        negative_10 = tf.nn.relu(tf.matmul(negative_9, self.w10) + self.b10)  # (.,1000)

        self.w11 = tf.Variable(tf.truncated_normal([1000, 128], stddev = 0.1))
        self.b11 = tf.Variable(tf.constant(0.0, shape = [128]))
        anchor_11 = tf.matmul(anchor_10, self.w11) + self.b11 # (.,128)
        positive_11 = tf.matmul(positive_10, self.w11) + self.b11  # (.,128)
        negative_11 = tf.matmul(negative_10, self.w11) + self.b11  # (.,128)

        anchor_embeddings = tf.nn.l2_normalize(anchor_11, axis = 1)
        positive_embeddings = tf.nn.l2_normalize(positive_11, axis = 1)
        negative_embeddings = tf.nn.l2_normalize(negative_11, axis = 1)

        return anchor_embeddings, positive_embeddings, negative_embeddings

    def calculate_loss(self, anchors: Tensor, positives: Tensor, negatives: Tensor):
        anchors_to_positives_dist =  tf.square(tf.norm(anchors - positives, axis = 1))
        anchors_to_negatives_dist = tf.square(tf.norm(anchors - negatives, axis = 1))

        alpha = 0.2
        loss_i  = tf.math.maximum(0.0, anchors_to_positives_dist + alpha - anchors_to_negatives_dist)

        return tf.reduce_sum(loss_i)