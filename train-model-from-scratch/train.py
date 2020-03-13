from recognizer.triplet_generator import TripletGenerator
from recognizer.tf_dataset import Dataset
from recognizer.model import Model
import tensorflow as tf
import pylab as plt
import numpy as np


def main():
    generator = TripletGenerator()
    # print 2 outputs from our generator just to see that it works:
    iter = generator.get_next_triplet()
    for i in range(2):
        print(next(iter))
    ds = Dataset(generator)
    model_input = ds.next_element
    model = Model(model_input)

    # train for 100k steps
    with tf.Session() as sess:
        # sanity test: plot out the first resized images:
        (img1, img2, img3) = sess.run([model_input.img1, model_input.img2, model_input.img3])

        # img1, img2, and img3 are BATCHES of images. plot out the first one
        plt.subplot(1, 3, 1)
        plt.imshow(img1[0].astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.imshow(img2[0].astype(np.uint8))
        plt.subplot(1, 3, 3)
        plt.imshow(img3[0].astype(np.uint8))
        plt.savefig('example-triplet.png')
        plt.close()

        # intialize the model
        sess.run(tf.global_variables_initializer())
        # run 100k optimization steps
        for step in range(100000):
            (_, current_loss) = sess.run([model.opt_step, model.loss])
            # print(f"step {step+1} loss {current_loss}")
            print("step: ", '%04d' % (step+1), " loss = ", "{:.9f}".format(current_loss))

        mat = sess.run(model.w1)
        np.save('w1.npy', mat)
        mat = sess.run(model.b1)
        np.save('b1.npy', mat)

        mat = sess.run(model.w2)
        np.save('w2.npy', mat)
        mat = sess.run(model.b2)
        np.save('b2.npy', mat)

        mat = sess.run(model.w3)
        np.save('w3.npy', mat)
        mat = sess.run(model.b3)
        np.save('b3.npy', mat)

        mat = sess.run(model.w4)
        np.save('w4.npy', mat)
        mat = sess.run(model.b4)
        np.save('b4.npy', mat)

        mat = sess.run(model.w5)
        np.save('w5.npy', mat)
        mat = sess.run(model.b5)
        np.save('b5.npy', mat)

        mat = sess.run(model.w6)
        np.save('w6.npy', mat)
        mat = sess.run(model.b6)
        np.save('b6.npy', mat)

        mat = sess.run(model.w7)
        np.save('w7.npy', mat)
        mat = sess.run(model.b7)
        np.save('b7.npy', mat)

        mat = sess.run(model.w8)
        np.save('w8.npy', mat)
        mat = sess.run(model.b8)
        np.save('b8.npy', mat)

        mat = sess.run(model.w9)
        np.save('w9.npy', mat)
        mat = sess.run(model.b9)
        np.save('b9.npy', mat)

        mat = sess.run(model.w10)
        np.save('w10.npy', mat)
        mat = sess.run(model.b10)
        np.save('b10.npy', mat)

        mat = sess.run(model.w11)
        np.save('w11.npy', mat)
        mat = sess.run(model.b11)
        np.save('b11.npy', mat)

if __name__ == '__main__':
    main()