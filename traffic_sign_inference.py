import os
import glob
import pickle
import random
import numpy as np
import argparse

import cv2
import pandas as pd
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt

import tensorflow as tf

#
# Methods to load and preprocess images are defined here but not really used until the next cell
#
def preprocess_test_image(path, imtype='png', prt_fcn=None):
    image = imread(path)
    image = imresize(image, (32, 32, 3))
    if imtype == 'png':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.GaussianBlur(image,(15,15),0)
    
    image = image.reshape((32, 32, 1))
    image = (image - 128) / 128
    return image


def get_images(glob_pattern=None, preprocess=True):
    if glob_pattern is None:
        glob_pattern = './data/other_images/*.png'

    images = []
    names = []
    for longname in glob.glob(glob_pattern):
        if preprocess:
            image = preprocess_test_image(longname)
        else:
            image = imread(longname)
        fname = os.path.basename(longname)
        images.append(image)
        names.append(fname)
    images = np.array(images)

    return images, names


def plot_images(preprocess=False):
    images, file_names = get_images(preprocess=preprocess)
    for i in range(len(images)):
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title('Name: %s' % file_names[i])
        plt.show()


def load_signnames():
    """Load the id to string name mapping into a dict.
    """
    id2names={}
    with open('signnames.csv') as f:
        d = f.readlines()
    for r in d[1:]:
        class_id, sign_name = r.strip().split(',')
        id2names[int(class_id)] = sign_name
    return id2names

def infer(glob_pattern=None):
    # Load the images
    images, file_names = get_images(glob_pattern)

    # The correct class for each image is encoded in the filename
    y = []
    for fname in file_names:
        y.append(fname.split('.')[-2].split('_')[-1])
    y = np.array(y)
        
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./saved_models/lenet.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./saved_models/'))

        # access tensors by name
        x = tf.get_default_graph().get_tensor_by_name("x:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        logits = tf.get_default_graph().get_tensor_by_name("logits:0")
        one_hot_y = tf.get_default_graph().get_tensor_by_name("one_hot_y:0")
        correct_prediction_op = tf.get_default_graph().get_tensor_by_name("correct_op:0")
        accuracy_operation_op = tf.get_default_graph().get_tensor_by_name("accuracy_op:0")

        # run the graph
        scores = sess.run(logits, feed_dict = {x: images, keep_prob: 1.0})
        
        # Check the accuracy
        x = sess.run(tf.one_hot(np.argmax(scores, axis=1), 43))
        y = sess.run(tf.one_hot(y, 43))

        correctness = sess.run(correct_prediction_op, feed_dict={logits: x, one_hot_y: y})
        accuracy = sess.run(accuracy_operation_op, feed_dict={correct_prediction_op: correctness})

    return file_names, accuracy, scores


def main():
    #plot_images(preprocess=False)

    id2names = load_signnames()
    files, accuracy, scores = infer()

    for i in range(len(files)):
        print("File: {:<31}: Actual: {:>2} --> Predicted: {} ({})".format(
            files[i], files[i].split('.')[-2].split('_')[-1], np.argmax(scores[i]), id2names[np.argmax(scores[i])]))

    print()
    print("Accuracy for the newly downloaded images:          %s" % str(int(accuracy * 100)) + '%')
    print()

    with tf.Session() as sess:
        topk = sess.run(tf.nn.top_k(tf.nn.softmax(scores), k=5))

    for i in range(len(files)):
        file = files[i]
        probs = topk.values[i]
        idxs = topk.indices[i]

        print("##### Top 5 Softmax for {}:".format(file))
        print()
        print("| {:<11} | {:<45} |".format("Probability", "Prediction"))
        print("|:{:11}-|:{:<40}-|".format("-"*11, "-"*45))

        for j in range(len(probs)):
            print("| {:11.4f} | {:<40} ({:2d}) |".format(probs[j], id2names[idxs[j]], idxs[j]))

        print("\n\n")


if __name__ == '__main__':
    main()




