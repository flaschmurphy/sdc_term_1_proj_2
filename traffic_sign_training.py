import os
import glob
import pickle
import random
import numpy as np
import argparse

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize, imsave

import tensorflow as tf

# Data was downloaded from this source: 
# https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
#@todo: implement a method to download this automatically if not present and remove raw data from source control


def load_data():
    """Load training, validation and test data
    """
    training_file = 'data/train.p'
    validation_file= 'data/valid.p'
    testing_file = 'data/test.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    X_train = train['features']
    y_train = train['labels']
    X_valid = valid['features']
    y_valid = valid['labels']
    X_test = test['features']
    y_test = test['labels']
    return X_train, y_train, X_valid, y_valid, X_test, y_test


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


def print_stats():
    """Print some basic stats obtained from the raw data.
    """
    n_train = X_train.shape[0]
    n_valid= X_valid.shape[0]
    n_test = X_test.shape[0]

    shapes = set([i.shape for i in X_train])
    assert len(shapes) == 1, 'Not all images have the same shape in the training data'
    image_shape = shapes.pop()

    n_classes = len(set(y_train))
    assert len(y_train) == len(X_train), 'Number of classes (y) must equal number of rows in input (X)'

    print("Number of training examples is: {}".format(n_train))
    print("Number of validation examples is: {}".format(n_valid))
    print("Number of testing examples is: {}".format(n_test))
    print("Image data shape is: {}".format(image_shape))
    print("Number of unique classes is: {}".format(n_classes))
    
    # Print a summary of the classes in y and their occurrence counts
    id2names = load_signnames()
    classes_wkeys = pd.Series([(i, id2names[i]) for i in y_train])
    print()
    print("Summary of classes and their frequency:")
    print(classes_wkeys.value_counts())
    print()
    

def get_rand_image():
    """Get a random image from the training data.
    """
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    return image, index


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


def plot_rand_sample(id2names):
    """Plot some samples for visual inspection. Running this cell multiple times will 
    produce different random samples from the training data.
    """
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20,20))
    for ax in np.array(axes).flatten():
        image, index = get_rand_image()
        class_id = y_train[index]
        ax.set_title('Idx: %d; Class: %d\nName: %s' % (index, class_id, id2names[class_id]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        s = fig.add_subplot(ax)
        s.imshow(image)
    plt.show()
    plt.clf()


def plot_hist_summary(y_train, y_valid, y_test):
    """Plot a histogram of the training, validation and test sets.
    """
    train_classes = pd.Series(y_train)
    train_classes.plot.hist(alpha=0.5, bins=43)
    
    train_classes = pd.Series(y_valid)
    train_classes.plot.hist(alpha=0.5, bins=43)
    plt.show()
    plt.clf()


def rgb2gray(image_set):
    x = np.array([cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in image_set]).reshape([-1, 32, 32, 1]) 
    return x


def normalize(image_set):
    x = image_set.copy().astype(np.float64)
    x = (x - 128) / 128
    return x


def preprocess(image_set):
    x = rgb2gray(image_set)
    x = normalize(x)
    return x


def LeNet(x, keep_prob):
    """Implement LeNet for image recognition
    """
    # Convolutional Layer. 32x32x1 --> 30x30x6 --> 15x15x6
    logits = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID') + b1
    logits = tf.nn.elu(logits)
    logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer: 15x15x6 --> 13x13x16 --> 6x6x16
    logits = tf.nn.conv2d(logits, w2, strides=[1, 1, 1, 1], padding='VALID') + b2
    logits = tf.nn.elu(logits)
    logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer: 6x6x16 --> 6x6x32
    logits = tf.nn.conv2d(logits, w3, strides=[1, 1, 1, 1], padding='VALID') + b3
    logits = tf.nn.elu(logits)
    
    # Flatten. 6x6x32 --> 1152
    logits = tf.contrib.layers.flatten(logits)

    # Fully Connected Layer. 1152 --> 120
    logits = tf.matmul(logits, w4) + b4
    logits = tf.nn.dropout(logits, keep_prob=keep_prob)
    logits = tf.nn.elu(logits)

    # Fully Connected Layer. 120 --> 84 
    logits = tf.matmul(logits, w5) + b5
    logits = tf.nn.dropout(logits, keep_prob=keep_prob)
    logits = tf.nn.elu(logits)

    # Fully Connected Layer. 84 --> 43
    logits = tf.matmul(logits, w6, name='logits') + b6
    
    return logits


def evaluate(X_data, y_data, accuracy_operation):
    """Evaluate the accuracy of X_data wrt y_data
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def init_weights():
    """Initialize weights and biases
    """
    global w1, w2, w3, w4, w5, w6, b1, b2, b3, b4, b5, b6
    mu = 0
    sigma = 0.1
    w1 = tf.Variable(tf.truncated_normal([3, 3, 1, 6], mean = mu, stddev = sigma))
    b1 = tf.Variable(tf.zeros([6]))

    w2 = tf.Variable(tf.truncated_normal([3, 3, 6, 16], mean = mu, stddev = sigma))
    b2 = tf.Variable(tf.zeros([16]))

    w3 = tf.Variable(tf.truncated_normal([1, 1, 16, 32], mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros([32]))

    w4 = tf.Variable(tf.truncated_normal([6*6*32, 120], mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros([120]))

    w5 = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros([84]))

    w6 = tf.Variable(tf.truncated_normal([84, 43], mean = mu, stddev = sigma))
    b6 = tf.Variable(tf.zeros([43]))

    
def train():
    """Main method - run the training including back prop and print out of results
    """
    global x, y, keep_prob, \
            X_train, y_train, X_valid, y_valid, X_test, y_test, \
            w1, w2, w3, w4, w5, w6, b1, b2, b3, b4, b5, b6, \
            logits, accuracy_operation, train_history, valid_history, \
            test_accuracy

    # Load the raw traffic sign data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    train_classes = pd.Series(y_train)
    valid_classes = pd.Series(y_valid)
    test_classes = pd.Series(y_test)

    # Save some images to be used later to manually check inference accuracy
    #save_random_sample()
    
    # Pre-process the images
    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid)
    X_test = preprocess(X_test)

    # x and y placeholders
    x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="x")
    y = tf.placeholder(tf.int32, (None), name="y")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    #
    # Tensorflow operations
    #
    ## Initialize weights operation
    init_weights()
    ## One hot vector for y. Note: 43 is the number of classes in the data.
    one_hot_y = tf.one_hot(y, 43, name='one_hot_y')
    ## Forward prop
    logits = LeNet(x, keep_prob=keep_prob)
    ## Error
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    ## Loss & Optimizer
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
    ## Back prop
    training_operation = optimizer.minimize(loss_operation)
    ## Evaluate
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1), name='correct_op')
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
    #
    
    # Run the training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print()
        print("Training...")
        print()

        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        epoch_number, train_history, valid_history = 0, [], []

        print("  Epoch# | Train | Valid")
        print("---------+-------+--------")

        for i in range(EPOCHS):
            epoch_number += 1
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})
                
            train_accuracy = evaluate(X_train, y_train, accuracy_operation)
            valid_accuracy = evaluate(X_valid, y_valid, accuracy_operation)
            
            # Only to be checked once model design is finalized
            test_accuracy = evaluate(X_test, y_test, accuracy_operation)
            
            train_history.append(train_accuracy)
            valid_history.append(valid_accuracy)
            
            print("  {:>6} | {:<.3f} | {:<.3f} ".format(i+1, train_accuracy, valid_accuracy))
        
        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')
        saver.save(sess, './saved_models/lenet')
        print("Model saved")
        print()
        

# Plot the training vs validation accuracy over time
def plot_accuracy_hist():
    """ Plot the training vs validation accuracy over time
    """
    plt.figure(figsize=(20,10))
    plt.title('Train vs. Validation Accuracy')
    ax = plt.subplot(111)
    _ = ax.plot(train_history, '--', color='blue', label='training')
    _ = ax.plot(valid_history, color='red', label='validation')
    _ = ax.legend()
    plt.show()


def main():
    global X_train, y_train, X_valid, y_valid, X_test, y_test,\
            EPOCHS, BATCH_SIZE, RATE, KEEP_PROB

    EPOCHS = 100
    BATCH_SIZE = 1024
    RATE = 0.0025
    KEEP_PROB = 0.5

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    print_stats()

    id2names = load_signnames()

    plot_rand_sample(id2names)
    plot_hist_summary(y_train, y_valid, y_test)

    plt.imshow(X_train[1243].squeeze())
    plt.title('Original Image')
    plt.show()

    # Pre-process all the images
    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid)
    X_test = preprocess(X_test)

    plt.imshow(X_train[1243].squeeze(), cmap="gray")
    plt.title('Normalized Grayscale Image')
    plt.show()

    train()

    plot_accuracy_hist()

    print("Final test accuracy: {:.2f}".format(test_accuracy))


if __name__ == '__main__':
    main()





