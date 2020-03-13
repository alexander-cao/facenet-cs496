""" import packages """
import numpy as np
import tensorflow as tf
import os
import glob

read_dictionary = np.load('people-class_num-dictionary.npy', allow_pickle = 'TRUE').item()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

pubfig83_train_path = 'recognizer/pubfig83-train'

labels_train = np.zeros(9295)
image_counter = 0

for person_folder in os.listdir(pubfig83_train_path):
    person_photos = glob.glob(pubfig83_train_path + os.path.sep + person_folder + os.path.sep + '*.jpg')
    for j in range(len(person_photos)):

        img_file = tf.read_file(person_photos[j])
        img = tf.image.decode_image(img_file)
        img.set_shape([None, None, 3])
        img_resized = tf.image.resize_images(img, [128, 128])
        img_resized = tf.reshape(img_resized, [1, 128, 128, 3])

        labels_train[image_counter] = read_dictionary[person_folder]
        
        if image_counter == 0:
            x_train_tf = img_resized
        else:
            x_train_tf = tf.concat([x_train_tf, img_resized], 0)

        image_counter += 1
        print(image_counter)



x_train_npy = sess.run(x_train_tf)

np.save('train-images.npy', x_train_npy)
np.save('train-labels.npy', labels_train)



pubfig83_test_path = 'recognizer/pubfig83-test'

labels_test = np.zeros(4543)
image_counter = 0

for person_folder in os.listdir(pubfig83_test_path):
    person_photos = glob.glob(pubfig83_test_path + os.path.sep + person_folder + os.path.sep + '*.jpg')
    for j in range(len(person_photos)):
        img_file = tf.read_file(person_photos[j])
        img = tf.image.decode_image(img_file)
        img.set_shape([None, None, 3])
        img_resized = tf.image.resize_images(img, [128, 128])
        img_resized = tf.reshape(img_resized, [1, 128, 128, 3])

        labels_test[image_counter] = read_dictionary[person_folder]

        if image_counter == 0:
            x_test_tf = img_resized
        else:
            x_test_tf = tf.concat([x_test_tf, img_resized], 0)

        image_counter += 1
        print(image_counter)

x_test_npy = sess.run(x_test_tf)

np.save('test-images.npy', x_test_npy)
np.save('test-labels.npy', labels_test)

