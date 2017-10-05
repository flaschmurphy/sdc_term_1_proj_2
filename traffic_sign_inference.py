import os
import glob
import numpy as np
import cv2
import pandas as pd

from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

def preprocess_test_image(path):
    image = imread(path)
    image = imresize(image, (32, 32, 3))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape((32, 32, 1))
    image = (image - 128) / 128
    return image


def get_images(glob_pattern='./data/other_images/*.png', preprocess=True):
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


def plot_images():
    images, file_names = get_images(preprocess=False)
    for image in images:
        plt.imshow(image.squeeze(), cmap="gray")
        plt.show()

def infer(glob_pattern=None):
    # Load the images
    if glob_pattern is None:
        images, file_names = get_images()
    else:
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
        x = sess.run(tf.one_hot(np.argmax(scores, axis=1), len(file_names)))
        y = sess.run(tf.one_hot(y, len(file_names)))

        correctness = sess.run(correct_prediction_op, feed_dict={logits: scores, one_hot_y: y})
        accuracy = sess.run(accuracy_operation_op, feed_dict={correct_prediction_op: correctness})

    results = dict(zip(file_names, [np.argmax(l) for l in scores]))

    return results, accuracy, scores

def main():

    print("\n\nWARNING: This script needs to be updated with the latest code from the Jupyter notebook\n\n")

    results_newimages = infer()
    for file in results_newimages[0].keys():
        print("File: {:<20}: Actual: {:>2} --> Predicted: {}".format(\
                file, file.split('.')[-2].split('_')[-1], results_newimages[0][file]))

    # Also check the accuracy for the test images that we saved to disk earlier from the training data.
    results_testimages = infer(glob_pattern='./data/random_x_sample/*.png')

    print("Accuracy for the newly downloaded images:          %s" % str(int(results_newimages[1] * 100)) + '%')
    print("Accuracy for the previously saved training images: %s" % str(int(results_testimages[1] * 100)) + '%')

    # Print out the top five softmax probabilities
    with tf.Session() as sess:
        topk = sess.run(tf.nn.top_k(tf.nn.softmax(results_newimages[2]), k=5))
    print("Top 5 Softmax probabilities: \n{}".format(topk))

if __name__ == '__main__':
    main()



