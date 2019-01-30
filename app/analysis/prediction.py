import re
import numpy as np
import tensorflow as tf
#Define some hyperparameters

numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 30

#load the data structures
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')

# create the graphs
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#value, _ = tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)
#weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
weight = tf.Variable(tf.truncated_normal([lstmUnits,numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

#load in the network
sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.import_meta_graph('/home/dennis/Desktop/projects/RNN/models/pretrained_lstm.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('/home/dennis/Desktop/projects/RNN/models/'))
#saver.restore(sess, tf.train.latest_checkpoint('/home/dennis/Desktop/projects/RNN/models/pretrained_lstm.ckpt'))


#format the input text/date


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix

inputText = "this was a good day everyone was very very happy,we really enjoyed ourselves with our friends.Our friends love use for this.It was such a charitable day."
inputMatrix = getSentenceMatrix(inputText)
print ("input matrix", inputMatrix)

predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# predictedSentiment[0] represents output score for positive sentiment
# predictedSentiment[1] represents output score for negative sentiment
print('sentiment',predictedSentiment)
if (predictedSentiment[0] > predictedSentiment[1]):
    print ("Positive Sentiment")
else:
    print ("Negative Sentiment")

secondInputText = "That movie was the best one I have ever seen."
secondInputMatrix = getSentenceMatrix(secondInputText)
predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
if (predictedSentiment[0] > predictedSentiment[1]):
    print ("Positive Sentiment")
else:
    print ("Negative Sentiment")