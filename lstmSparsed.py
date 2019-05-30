import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import numpy as np
import matplotlib.pyplot as plt


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, "rb"))


# folderInputs = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/'
folderInputs = 'F:/atil/inputsFrom_20190504_181608/'
inputs = []
inputDictionary = []
maxSampleLen = 0  # timestep is how long is the file(samples in the file)
labelList = dict()
maxLenSparsed = 1000

# prepare max length and file names
print('Initial Scan. Classes:', end='')
for rootPath, directories, files in os.walk(folderInputs):
    for filename in files:
        if '.inp2' in filename:
            inputFile = loadData(rootPath + filename)
            inputShape = np.shape(inputFile)
            if inputShape[1:] == (4, 2, 9):  # todo - ai : fix all files, extract again
                maxSampleLen = max(maxSampleLen, len(inputFile))
                label = filename[:2]
                if label not in labelList:
                    print(os.linesep+label, end='')
                    labelCount = len(labelList)
                    for l in labelList:
                        labelList[l].append(0)
                    labelList[label] = (labelCount * [0])
                    labelList[label].append(1)
                print('.', end='', flush=True)
                inputs.append(filename)
print('')
print('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
print('Max file length:', maxSampleLen, 'samples')
print('Total of', len(labelList), 'classes')

stepSize = int(np.ceil(maxSampleLen / maxLenSparsed))
maxSampleLen = maxLenSparsed

np.random.shuffle(inputs)

# Training Parameters
learningRate = 0.001
# train with 80% files, test with 20%
totalOfInputs = len(inputs)
trainingSteps = int(totalOfInputs * 0.8)
testSteps = totalOfInputs - trainingSteps
trainingEpoch = 100

numInput = [2, 9]  # input data size (9 inst freqs, 9 inst mags)
channels = 4  # recorded with 4 microphones
timeSteps = maxSampleLen

# number of units in RNN cell
numHidden = 128
numClasses = len(labelList)  # total number of classification classes (ie. people)

# tf Graph input
x = tf.placeholder("float", [None, timeSteps, channels*numInput[0]*numInput[1]])  # feature set, flattened
y = tf.placeholder("float", [None, numClasses])
seqLen = tf.placeholder(tf.int32)

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([numHidden, numClasses]))
}
biases = {
    'out': tf.Variable(tf.random_normal([numClasses]))
}


def rnnFunc(xParam, weiParam, biaParam):

    xParam = tf.unstack(xParam, timeSteps, 1)

    lstmCell = rnn.BasicLSTMCell(numHidden)

    # generate prediction
    outputs, states = rnn.static_rnn(lstmCell, xParam, sequence_length=seqLen, dtype=tf.float32)

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weiParam['out']) + biaParam['out']


logits = rnnFunc(x, weights, biases)
prediction = tf.nn.softmax(logits)

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

# model evaluation
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    # run initializer
    sess.run(init)
    # print('session initialized')
    loses = []
    accur = []
    # Training
    for e in range(0, trainingEpoch):
        loss_total = 0
        acc_total = 0
        for step in range(0, trainingSteps):
            filename = inputs[step]
            inFile = loadData(folderInputs + filename)
            inFile = [inFile[x] for x in range(0, len(inFile), stepSize)]
            sequenceLength = len(inFile)
            diff = maxSampleLen - sequenceLength
            batchX = np.pad(inFile, ((0, diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            batchX = np.reshape(batchX, (1, maxSampleLen, channels*numInput[0]*numInput[1]))
            batchY = labelList[filename[:2]]
            batchY = np.reshape(batchY, (1, numClasses))

            _, acc, mbloss, onehot_pred = sess.run([optimizer, accuracy, loss, logits],
                                                   feed_dict={x: batchX, y: batchY, seqLen: sequenceLength})

            loss_total += mbloss
            acc_total += acc

        print("Epoch " + str(e+1) + " (", str(trainingSteps) + " training steps), Epoch Loss= " +
              "{:.4f}".format(loss_total) + ", Training Accuracy= " +
              "{:.3f}".format(acc_total/trainingSteps))

        loses.append(loss_total)
        accur.append(acc_total/trainingSteps)

    # Testing
    acc_total = 0
    for inpFl in inputs[-testSteps:]:
        inFile = loadData(folderInputs + inpFl)
        inFile = [inFile[x] for x in range(0, len(inFile), stepSize)]
        sequenceLength = len(inFile)
        diff = maxSampleLen - sequenceLength
        batchX = np.pad(inFile, ((0, diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        batchX = np.reshape(batchX, (1, maxSampleLen, channels*numInput[0]*numInput[1]))
        batchY = labelList[inpFl[:2]]
        batchY = np.reshape(batchY, (1, numClasses))

        acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, seqLen: sequenceLength})
        acc_total += acc
    print("Testing Accuracy:", acc_total/testSteps)

    plt.plot(loses)
    plt.show()
    plt.plot(accur)
    plt.show()
