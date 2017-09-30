import pickle
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
import os
import glob
from scipy.ndimage import imread
from scipy.misc import imresize
from pprint import pprint as pp

def load_data():
    global X_train, y_train, X_valid, y_valid, X_test, y_test
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

def print_stats():
    n_train = X_train.shape[0]
    n_valid= X_valid.shape[0]
    n_test = X_test.shape[0]

    shapes = set([i.shape for i in X_train])
    assert len(shapes) == 1, 'Not all images have the same shape in the training data'
    image_shape = shapes.pop()

    n_classes = len(set(y_train))

    print("Number of training examples is: {}".format(n_train))
    print("Number of validation examples is: {}".format(n_valid))
    print("Number of testing examples is: {}".format(n_test))
    print("Image data shape is: {}".format(image_shape))
    print("Number of unique classes is: {}".format(n_classes))

    assert len(y_train) == len(X_train), 'Number of classes (y) must equal number of rows in input (X)'

def get_rand_image():
    """Get a random image from the training data."""
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    return image, index

def load_signnames():
    """Load the id to string name mapping into a dict."""
    id2names={}
    with open('signnames.csv') as f:
        d = f.readlines()
    for r in d[1:]:
        class_id, sign_name = r.strip().split(',')
        id2names[int(class_id)] = sign_name
    return id2names

#def plot_rand_sample(id2names):
#    """Plot some samples for visual inspection. Running this cell multiple times will 
#    produce different random samples from the training data."""
#    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20,20))
#    for ax in np.array(axes).flatten():
#        image, index = get_rand_image()
#        class_id = y_train[index]
#        ax.set_title('Idx: %d; Class: %d\nName: %s' % (index, class_id, id2names[class_id]))
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        s = fig.add_subplot(ax)
#        s.imshow(image)
#    plt.show()

def normalize(X_train):
    x = X_train.copy().astype(np.float64)
    x = (x - 128) / 128
    return x

def rgb2gray(arr):
    return np.array([cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in arr]).reshape([-1, 32, 32, 1]) 

def LeNet(x, keep_prob=1):    
    mu = 0
    sigma = 0.1
    
    # Convolutional Layer. 32x32x1 --> 30x30x6 --> 15x15x6
    weights = tf.Variable(tf.truncated_normal([3, 3, 1, 6], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros([6]))
    logits = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
    logits = tf.nn.elu(logits)
    logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer: 15x15x6 --> 13x13x16 --> 6x6x16
    weights = tf.Variable(tf.truncated_normal([3, 3, 6, 16], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros([16]))
    logits = tf.nn.conv2d(logits, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
    logits = tf.nn.elu(logits)
    logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer: 6x6x16 --> 6x6x32
    weights = tf.Variable(tf.truncated_normal([1, 1, 16, 32], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros([32]))
    logits = tf.nn.conv2d(logits, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
    logits = tf.nn.elu(logits)
    
    # Flatten. 6x6x32 --> 1152
    logits = tf.contrib.layers.flatten(logits)

    # Fully Connected Layer. 1152 --> 120
    weights = tf.Variable(tf.truncated_normal([6*6*32, 120], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros([120]))
    logits = tf.matmul(logits, weights) + bias
    logits = tf.nn.dropout(logits, keep_prob=keep_prob)
    logits = tf.nn.elu(logits)

    # Fully Connected Layer. 120 --> 84 
    weights = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros([84]))
    logits = tf.matmul(logits, weights) + bias
    logits = tf.nn.dropout(logits, keep_prob=keep_prob)
    logits = tf.nn.elu(logits)

    # Fully Connected Layer. 84 --> 43
    weights = tf.Variable(tf.truncated_normal([84, 43], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros([43]))
    logits = tf.matmul(logits, weights) + bias
    
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def preprocess_test_image(path):
    image = imread(path, mode='RGB')
    image = imresize(image, (32, 32, 3))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape((32, 32, 1))
    image = (image - 128) / 128
    return image


def main():
    global X_train, y_train, X_valid, y_valid, X_test, y_test, BATCH_SIZE, accuracy_operation, x, y
    load_data()

    id2names = load_signnames()
    train_classes = pd.Series(y_train)
    valid_classes = pd.Series(y_valid)
    classes_wkeys = pd.Series([(i, id2names[i]) for i in y_train])
    print(classes_wkeys.value_counts())
    
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    X_test = rgb2gray(X_test)
    
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    X_test = normalize(X_test)
    
    EPOCHS = 200
    BATCH_SIZE = 1024
    RATE = 0.005
    KEEP_PROB = 0.7
    
    x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="x_placeholder")
    y = tf.placeholder(tf.int32, (None), name="y_placeholder")
    one_hot_y = tf.one_hot(y, 43)
    
    logits = LeNet(x, keep_prob=KEEP_PROB)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = RATE)
    training_operation = optimizer.minimize(loss_operation)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        epoch_number, train_history, valid_history = 0, [], []
        
        print("Training...")
        print()
        print("  Epoch# | Train | Valid")
        print("--------------------------")
        for i in range(EPOCHS):
            epoch_number += 1
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
            train_accuracy = evaluate(X_train, y_train)
            valid_accuracy = evaluate(X_valid, y_valid)
            
            train_history.append(train_accuracy)
            valid_history.append(valid_accuracy)
            
            print("  {:>6} | {:<.3f} | {:<.3f} ".format(i+1, train_accuracy, valid_accuracy))
        
        if not os.path.exists('./saved_models'):
            os.makedir('./saved_models')
        saver.save(sess, './saved_models/lenet')
        print("Model saved")
    
    y_lookup = {
        '001.jpg': 22,     # Bumpy Road
        '002.jpg': 5,      # Speed limit (80km/h)
        '003.jpg': 1,      # Speed limit (30km/h)
        '004.jpg': 2,      # Speed limit (50km/h)
        '005.jpg': 14,     # Stop
        '006.jpg': 14,     # Stop
        '007.jpg': 4,      # Speed limit (70km/h)
    }
    
    for path in glob.glob('./data/other_data/*.jpg'):
        image = preprocess_test_image(path)
        fname = (os.path.basename(path))
        
        actuals = []
        results = []
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())    
            onehot = sess.run(one_hot_y, feed_dict={x: [image], y: [y_lookup[fname]]})
    
        prediction = np.where(onehot[0] == 1)[0][0]    
        print("{} <==> Actual: {:20s} ({:2d}) <==> Predicted: {:20s} ({:2d})".format(
            fname, 
            id2names[y_lookup[fname]],
            y_lookup[fname],
            id2names[prediction],
            prediction)
        )
        
        actuals.append(y_lookup[fname])
        results.append(prediction)
    
    percentage_accuracy = (sum([a == b for a, b in zip(actuals, results)]) / float(len(actuals))) * 100
    
    print("Final Accuracy: {:.0f}%".format(percentage_accuracy))
    
    init = tf.global_variables_initializer()
    for path in glob.glob('./data/other_data/*.jpg'):
        image = preprocess_test_image(path)
        fname = os.path.basename(path)
    
        with tf.Session() as sess:
            sess.run(init)
            result = sess.run(tf.nn.softmax(one_hot_y), feed_dict={x: [image], y: [y_lookup[fname]]})
            topk = sess.run(tf.nn.top_k(result, k=5))
         
        print('{}     ==> TopK Vals: {}, Index: {} \nPrediction: ==> {}'.format(
            os.path.basename(path), topk.values, topk.indices, id2names[0]))
        
        #plt.imshow(imread(path, mode='RGB'))
        #plt.show()
    

if __name__ == "__main__":
    main()
    

