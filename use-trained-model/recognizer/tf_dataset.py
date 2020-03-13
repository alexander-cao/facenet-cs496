import tensorflow as tf
from .triplet_generator import TripletGenerator
from .model import Inputs


class Dataset(object):
    img1_resized = 'img1_resized'
    img2_resized = 'img2_resized'
    img3_resized = 'img3_resized'

    def __init__(self, generator = TripletGenerator()):
        self.next_element = self.build_iterator(generator)

    def build_iterator(self, triplet_gen: TripletGenerator):
        batch_size = 128
        prefetch_batch_buffer = 5

        dataset = tf.data.Dataset.from_generator(triplet_gen.get_next_triplet, output_types = {TripletGenerator.anchor: tf.string, TripletGenerator.positive: tf.string, TripletGenerator.negative: tf.string})
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        return Inputs(element[self.img1_resized], element[self.img2_resized], element[self.img3_resized])

    def _read_image_and_resize(self, triplet_element):
        target_size = [128, 128]
        # read images from disk
        img1_file = tf.read_file(triplet_element[TripletGenerator.anchor])
        img2_file = tf.read_file(triplet_element[TripletGenerator.positive])
        img3_file = tf.read_file(triplet_element[TripletGenerator.negative])
        img1 = tf.image.decode_image(img1_file)
        img2 = tf.image.decode_image(img2_file)
        img3 = tf.image.decode_image(img3_file)

        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img1.set_shape([None, None, 3])
        img2.set_shape([None, None, 3])
        img3.set_shape([None, None, 3])

        # resize to model input size
        img1_resized = tf.image.resize_images(img1, target_size)
        img2_resized = tf.image.resize_images(img2, target_size)
        img3_resized = tf.image.resize_images(img3, target_size)

        triplet_element[self.img1_resized] = img1_resized
        triplet_element[self.img2_resized] = img2_resized
        triplet_element[self.img3_resized] = img3_resized

        return triplet_element