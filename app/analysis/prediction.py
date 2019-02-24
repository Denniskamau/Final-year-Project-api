import re
import os
import numpy as np
import tensorflow as tf
#Define some hyperparameters
class Prediction():

    numDimensions = 300
    maxSeqLength = 250
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 30

    #load the data structures
    data_dir = os.path.dirname(__file__)
    print(data_dir)
    wordlist_data = "wordsList.npy"
    wordvector_data = "wordVectors.npy"
    # abs_file_path = os.path.join(data_dir,wordlist_data)
    wordsList = np.load(os.path.join(data_dir,wordlist_data)).tolist()
    print("loaded wordlist")
    wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
    wordVectors = np.load(os.path.join(data_dir,wordvector_data))

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
    pretrain_path  = "pretrained_lstm.ckpt-40000.meta"
    saver = tf.train.import_meta_graph( os.path.join(data_dir,'models/pretrained_lstm.ckpt-50000.meta') )
    saver.restore(sess,tf.train.latest_checkpoint(os.path.join(data_dir,'models/')))
    #saver.restore(sess, tf.train.latest_checkpoint('/home/dennis/Desktop/projects/RNN/models/pretrained_lstm.ckpt'))


    #format the input text/date



    def cleanSentences(self,string):
        # Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())

    def getSentenceMatrix(self,sentence):
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        sentenceMatrix = np.zeros([self.batchSize,self.maxSeqLength], dtype='int32')
        cleanedSentence = self.cleanSentences(sentence)
        split = cleanedSentence.split()
        for indexCounter,word in enumerate(split):
            try:
                sentenceMatrix[0,indexCounter] = self.wordsList.index(word)
            except ValueError:
                sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
        return sentenceMatrix

    def receiveAnalysisParameter(self,inputParam):
        inputMatrix = self.getSentenceMatrix(inputParam)
        predictedSentiment = self.sess.run(self.prediction, {self.input_data: inputMatrix})[0]
        # predictedSentiment[0] represents output score for positive sentiment
        # predictedSentiment[1] represents output score for negative sentiment
        percentage = {"pos":"","neg":""}
        percentage["pos"]= str(predictedSentiment[0])
        percentage["neg"]= str(predictedSentiment[1])

        if (predictedSentiment[0] > predictedSentiment[1]):
            data = {
                "message": "Positive Sentiment",
                "percentage": percentage,
                "tweet":inputParam
            }
            return data
        else:
            data = {
                "message":"Negative Sentiment",
               "tweet":inputParam,
               "percentage":percentage
            }
            return data

