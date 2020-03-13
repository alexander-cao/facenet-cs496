# training set size is 9295
# test set size is 4543

""" import packages """
import numpy as np
import tensorflow as tf


""" load weights """
w1 = np.load('w1.npy')
b1 = np.load('b1.npy')

w2 = np.load('w2.npy')
b2 = np.load('b2.npy')

w3 = np.load('w3.npy')
b3 = np.load('b3.npy')

w4 = np.load('w4.npy')
b4 = np.load('b4.npy')

w5 = np.load('w5.npy')
b5 = np.load('b5.npy')

w6 = np.load('w6.npy')
b6 = np.load('b6.npy')

w7 = np.load('w7.npy')
b7 = np.load('b7.npy')

w8 = np.load('w8.npy')
b8 = np.load('b8.npy')

w9 = np.load('w9.npy')
b9 = np.load('b9.npy')

w10 = np.load('w10.npy')
b10 = np.load('b10.npy')

w11 = np.load('w11.npy')
b11 = np.load('b11.npy')

""" trained facenet class """
class facenet(object):
    def __init__(self):

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 3], name = "x") # data

        """ create network """
        self._create_network()

        """ initializing the tensorflow variables """
        init = tf.global_variables_initializer()

        """ launch session """
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        self.w1 = tf.Variable(w1)
        self.b1 = tf.Variable(b1)
        embedding_1 = tf.nn.relu(tf.nn.conv2d(self.x/255.0, self.w1, strides = [1, 1, 1, 1], padding = 'SAME') + self.b1)

        self.w2 = tf.Variable(w2)
        self.b2 = tf.Variable(b2)
        embedding_2 = tf.nn.relu(tf.nn.conv2d(embedding_1, self.w2, strides = [1, 1, 1, 1], padding = 'SAME') + self.b2)

        embedding_2_max = tf.nn.max_pool(embedding_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self.w3 = tf.Variable(w3)
        self.b3 = tf.Variable(b3)
        embedding_3 = tf.nn.relu(tf.nn.conv2d(embedding_2_max, self.w3, strides = [1, 1, 1, 1], padding = 'SAME') + self.b3)

        self.w4 = tf.Variable(w4)
        self.b4 = tf.Variable(b4)
        embedding_4 = tf.nn.relu(tf.nn.conv2d(embedding_3, self.w4, strides = [1, 1, 1, 1], padding = 'SAME') + self.b4)

        embedding_4_max = tf.nn.max_pool(embedding_4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self.w5 = tf.Variable(w5)
        self.b5 = tf.Variable(b5)
        embedding_5 = tf.nn.relu(tf.nn.conv2d(embedding_4_max, self.w5, strides = [1, 1, 1, 1], padding = 'SAME') + self.b5)

        self.w6 = tf.Variable(w6)
        self.b6 = tf.Variable(b6)
        embedding_6 = tf.nn.relu(tf.nn.conv2d(embedding_5, self.w6, strides = [1, 1, 1, 1], padding = 'SAME') + self.b6)

        embedding_6_max = tf.nn.max_pool(embedding_6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self.w7 = tf.Variable(w7)
        self.b7 = tf.Variable(b7)
        embedding_7 = tf.nn.relu(tf.nn.conv2d(embedding_6_max, self.w7, strides = [1, 1, 1, 1], padding = 'SAME') + self.b7)

        self.w8 = tf.Variable(w8)
        self.b8 = tf.Variable(b8)
        embedding_8 = tf.nn.relu(tf.nn.conv2d(embedding_7, self.w8, strides = [1, 1, 1, 1], padding = 'SAME') + self.b8)

        embedding_8_max = tf.nn.max_pool(embedding_8, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        embedding_8_max_flat = tf.reshape(embedding_8_max, [-1, 8 * 8 * 100])  # (., 8*8*100)

        self.w9 = tf.Variable(w9)
        self.b9 = tf.Variable(b9)
        embedding_9 = tf.nn.relu(tf.matmul(embedding_8_max_flat, self.w9) + self.b9) # (.,2000)

        self.w10 = tf.Variable(w10)
        self.b10 = tf.Variable(b10)
        embedding_10 = tf.nn.relu(tf.matmul(embedding_9, self.w10) + self.b10) # (.,1000)

        self.w11 = tf.Variable(w11)
        self.b11 = tf.Variable(b11)
        embedding_11 = tf.matmul(embedding_10, self.w11) + self.b11 # (.,128)

        self.embedding = tf.nn.l2_normalize(embedding_11, axis = 1)

    def transform(self, data):
        """Transform data by mapping it into the embedding space."""
        return self.sess.run(self.embedding, feed_dict = {self.x: data})

model = facenet()

train_images = np.load('train-images.npy')
train_z = np.zeros((9295, 128))

for i in range(int(9295/55)):

    x_batch = train_images[i*55:(i+1)*55, :, :, :]

    z_batch = model.transform(x_batch)
    
    train_z[i*55:(i+1)*55, :] = z_batch
    
    print(i)

np.save('train-embeddings', train_z)




test_images = np.load('test-images.npy')
test_z = np.zeros((4543, 128))

for i in range(int(4543 / 59)):
    x_batch = test_images[i * 59:(i + 1) * 59, :, :, :]

    z_batch = model.transform(x_batch)

    test_z[i * 59:(i + 1) * 59, :] = z_batch

    print(i)

np.save('test-embeddings', test_z)

