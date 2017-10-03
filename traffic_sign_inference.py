import os
import glob
import numpy as np
import cv2
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize


def preprocess_test_image(path):
    image = imread(path)
    image = imresize(image, (32, 32, 3))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape((32, 32, 1))
    image = (image - 128) / 128
    return image


def get_images():
    images = []
    names = []
    for path in glob.glob('./data/other_data/*.png'):
    #for path in glob.glob('./data/random_x_sample/*.png'):
        image = preprocess_test_image(path)
        fname = os.path.basename(path)
        images.append(image)
        names.append(fname)
    images = np.array(images)

    return images, names


def main():
    images = get_images()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./saved_models/lenet.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./saved_models/'))

        # access tensors by name
        x = tf.get_default_graph().get_tensor_by_name("x:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        logits = tf.get_default_graph().get_tensor_by_name("logits:0")

        # Load the images
        images, names = get_images()

        # run some operations
        predictions = sess.run(logits, feed_dict = {x: images, keep_prob: 1.0})

    results = dict(zip(names, [np.argmax(l) for l in predictions]))

    return results


if __name__ == '__main__':
    results = main()


