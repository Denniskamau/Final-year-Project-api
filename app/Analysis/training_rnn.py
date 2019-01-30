from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from six.moves import urllib

import os
import re
import math
import random
import zipfile
import tarfile
import numpy as np
import collections
import tensorflow as tf
import matplotlib as mp
import matplotlib.pyplot as plt


DOWNLOADED_FILENAME = 'ImdbReviews.tar.gz'
def download_file(url_path):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path,DOWNLOADED_FILENAME)

    print('Found and verified file from this path: ', url_path)
    print('DOwnloaded file: ', DOWNLOADED_FILENAME)

# Extract and clean up the review
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")

def get_reviews(dirName,positive=True):
    label =1 if positive else 0

    reviews = []
    labels = []
    for fileName in os.listdir(dirName):
        if fileName.endswith('.txt'):
            with open (dirName + fileName, 'r+') as f:
                review = f.read()
                review = review.lower().replace("<br />", " ")
                review = re.sub(TOKEN_REGEX, '', review)

                 #return a turple of the review text and a label for whether it is positive or
        # negative
                reviews.append(review)
                labels.append(label)

    return reviews, labels


def extract_labels_data():
    # if the file has not already been extracted
    if not os.path.exists('aclImdb'):
        with tarfile.open(DOWNLOADED_FILENAME) as tar:
            tar.extractall()
            tar.close()
    positive_reviews, positive_labels = get_reviews('aclImdb/train/pos/',positive = True)
    negative_reviews, negative_labels = get_reviews('aclImdb/train/pos/',positive = False)

    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels

    return data, labels

URL_PATH = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
download_file(URL_PATH)

data,labels = extract_labels_data()

MAX_SEQUENCE_LENGTH = 250

words = np.load('wordsList.npy')#load all the words in the vocabulary
#glove has a vocabulary of 400,000 words.

#mapping words to uniqe indexes
review_ids = np.load('idsMatrix.npy')
#set up the array for the training data
x_data = review_ids
y_output = np.array(labels)

vocabulary_size = len(words)

#shuffle b4 feeding for training
np.random.seed(22)
shuffled_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffled_indices]
y_shuffled = y_output[shuffled_indices]

TRAIN_DATA =5500
TOTAL_DATA =6000

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]

#Training work is done time to define the neural network
tf.reset_default_graph()
#define the inputs
x = tf.placeholder(tf.int32, [None,MAX_SEQUENCE_LENGTH])#numeric rep of reviews
y = tf.placeholder(tf.int32,[None])#corresponding labels

num_epochs = 10
batch_size = 25
embedding_size = 50
max_label = 2

saved_embeddings = np.load('wordVectors.npy')
embeddings = tf.nn.embedding_lookup(saved_embeddings,x)

lstmCell= tf.contrib.rnn.BasicLSTMCell(embedding_size)
#Prevent overfitting the model to the input cell
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

_, (encodding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)#feed to softmax prdiction model


logits = tf.layers.dense(encodding, max_label, activation=None)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_entropy)

prediction = tf.equal(tf.argmax(logits,1),tf.cast(y,tf.int64))

accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

#optimizer to minimise loss
optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    init.run()
    highest_test_acc = 0
    highest_session = None
    for epoch in range(num_epochs):
        num_batches = int(len(train_data) // batch_size) + 1

        for i in range (num_batches):
            min_ix = i * batch_size
            max_ix = np.min([len(train_data),((i+1)* batch_size)])

            x_train_batch  = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]

            train_dict = {x: x_train_batch, y: y_train_batch}
            session.run(train_step, feed_dict=train_dict)

            train_loss,train_acc = session.run([loss,accuracy], feed_dict=train_dict)

        test_dict = {x: test_data, y:test_target}

        test_loss, test_acc = session.run([loss,accuracy],feed_dict=test_dict)
        print('Epoch:{}, Test Loss: {:.2}, Test Acc: {:.5}' .format(epoch + 1,test_loss,test_acc))


        if test_acc > highest_test_acc:
            highest_test_acc = test_acc
            print('highest accuracy is ', highest_test_acc)
            highest_session = session

    # save highest session
    saver = tf.train.Saver()
    save_path = saver.save(highest_session, "/home/dennis/Desktop/projects/RNN/models/pretrained_lstm.ckpt")
    print("saved to %s" % save_path)



#Training work is done time to define the neural network
tf.reset_default_graph()
#define the inputs
x = tf.placeholder(tf.int32, [None,MAX_SEQUENCE_LENGTH])#numeric rep of reviews
y = tf.placeholder(tf.int32,[None])#corresponding labels

num_epochs = 10
batch_size = 25
embedding_size = 50
max_label = 2

saved_embeddings = np.load('wordVectors.npy')
embeddings = tf.nn.embedding_lookup(saved_embeddings,x)

lstmCell= tf.contrib.rnn.BasicLSTMCell(embedding_size)
#Prevent overfitting the model to the input cell
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

_, (encodding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)#feed to softmax prdiction model


logits = tf.layers.dense(encodding, max_label, activation=None)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_entropy)

prediction = tf.equal(tf.argmax(logits,1),tf.cast(y,tf.int64))

accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

#optimizer to minimise loss
optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    init.run()
    highest_test_acc = 0
    highest_session = None
    for epoch in range(num_epochs):
        num_batches = int(len(train_data) // batch_size) + 1

        for i in range (num_batches):
            min_ix = i * batch_size
            max_ix = np.min([len(train_data),((i+1)* batch_size)])

            x_train_batch  = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]

            train_dict = {x: x_train_batch, y: y_train_batch}
            session.run(train_step, feed_dict=train_dict)

            train_loss,train_acc = session.run([loss,accuracy], feed_dict=train_dict)

        test_dict = {x: test_data, y:test_target}

        test_loss, test_acc = session.run([loss,accuracy],feed_dict=test_dict)
        print('Epoch:{}, Test Loss: {:.2}, Test Acc: {:.5}' .format(epoch + 1,test_loss,test_acc))


        if test_acc > highest_test_acc:
            highest_test_acc = test_acc
            print('highest accuracy is ', highest_test_acc)
            highest_session = session

    # save highest session
    saver = tf.train.Saver()
    save_path = saver.save(highest_session, "/home/dennis/Desktop/projects/RNN/models/pretrained_lstm.ckpt")
    print("saved to %s" % save_path)

