import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import numpy as np
import matplotlib.pyplot as plt


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, "rb"))


folderInputs = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/'
# folderInputs = 'E:/atil/BreathDataset/Processed/inputsFrom_20190504_181608/'
inputs = []
inputDictionary = []
maxSampleLen = 0  # timestep is how long is the file(samples in the file)
labelList = dict()

# prepare max length and file names
for rootPath, directories, files in os.walk(folderInputs):
    for filename in files:
        if '.inp2' in filename:
            inputFile = loadData(rootPath + filename)
            maxSampleLen = max(maxSampleLen, len(inputFile))
            label = filename[:2]
            if label not in labelList:
                # print('new label:', label)
                labelCount = len(labelList)
                for l in labelList:
                    labelList[l].append(0)
                labelList[label] = (labelCount * [0])
                labelList[label].append(1)
            # labels.append(labelList[label])
            # lengths.append(len(inputFile))
            # inputs.append(inputFile)
            inputs.append(filename)
print('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
print('Max file length:', maxSampleLen, 'samples')
print('Total of', len(labelList), 'classes')

# todo - ai : remove maxlen setting to original
maxSampleLen = 10

np.random.shuffle(inputs)
# print(inputs)

# print(labelList)

# for i in range(len(inputs)):
#     diff = maxSampleLen - len(inputs[i])
#     inputs[i] = np.pad(inputs[i], ((0, diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
# print(np.shape(inputs))

# # todo - ai : remove maxlen setting to original
# for i in range(len(inputs)):
#     maxSampleLen = 10
#     inputs[i] = inputs[i][:maxSampleLen]
# print(np.shape(inputs))

# shape is (fileCount, timeSteps(max sampleCount), channelCount, 2(instf and mag), imfCount)

# Training Parameters
learningRate = 0.001
# todo - ai : create inputs for all breath examples after constructing network structure
# train with 8 files, test with 2 (total of 10 files for draft code)
totalOfInputs = len(inputs)
trainingSteps = int(totalOfInputs * 0.8)
testSteps = totalOfInputs - trainingSteps

numInput = [2, 9]  # input data size (9 inst freqs, 9 inst mags)
channels = 4  # recorded with 4 microphones
timeSteps = maxSampleLen

# number of units in RNN cell
numHidden = 128
# todo - ai : increase classes to total number of people after constructing nw structure
numClasses = len(labelList)  # total number of classification classes (ie. people)

# # todo - ai : move these lines up
# # flatten features
# for i in range(len(inputs)):
#     inputs[i] = inputs[i].reshape(-1, channels*numInput[0]*numInput[1])
# print(np.shape(inputs))

# tf Graph input
x = tf.placeholder("float", [None, timeSteps, channels*numInput[0]*numInput[1]])
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

    # print(np.shape(xParam))
    # Manipulate x here for correct shape, if not
    # xParam = np.reshape(xParam, (-1, channels*numInput[0]*numInput[1]))
    xParam = tf.unstack(xParam, timeSteps, 1)

    # print(np.shape(xParam))
    # xParam = np.expand_dims(xParam, axis=0)
    lstmCell = rnn.BasicLSTMCell(numHidden)

    # generate prediction
    outputs, states = rnn.static_rnn(lstmCell, xParam, sequence_length=seqLen, dtype=tf.float32)

    # todo - ai : debug output len
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

# for i in range(len(inputs)):
#     inputDictionary.append({'label': labels[i], 'length': lengths[i], 'values': inputs[i]})
# np.random.shuffle(inputDictionary)
#
# inputs = []
# labels = []
# lengths = []
#
# for idd in inputDictionary:
#     inputs.append(idd['values'])
#     labels.append(idd['label'])
#     lengths.append(idd['length'])
#
# print(np.shape(inputs))
# print(np.shape(labels))
# print(np.shape(lengths))

# initializing the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    # acc_total = 0
    # loss_total = 0

    # run initializer
    sess.run(init)
    # print('session initialized')
    loses = []
    accur = []
    for e in range(0, 100):
        loss_total = 0
        acc_total = 0
        for step in range(0, trainingSteps):
            filename = inputs[step]
            inFile = loadData(folderInputs + filename)
            batchX = inFile[:maxSampleLen]
            batchX = np.reshape(batchX, (1, maxSampleLen, channels*numInput[0]*numInput[1]))
            batchY = labelList[filename[:2]]
            batchY = np.reshape(batchY, (1, numClasses))
            # Manipulate x here for correct shape, if not

            _, acc, mbloss, onehot_pred = sess.run([optimizer, accuracy, loss, logits],
                                                   feed_dict={x: batchX, y: batchY, seqLen: maxSampleLen})

            loss_total += mbloss
            acc_total += acc

        print("Epoch " + str(e+1) + " (", str(trainingSteps) + " training steps), Epoch Loss= " +
              "{:.4f}".format(loss_total) + ", Training Accuracy= " +
              "{:.3f}".format(acc_total/trainingSteps))

        loses.append(loss_total)
        accur.append(acc_total/trainingSteps)

        # print("Optimization Finished!")
    acc_total = 0
    for inpFl in inputs[-testSteps:]:
        inFile = loadData(folderInputs + inpFl)
        batchX = inFile[:maxSampleLen]
        batchX = np.reshape(batchX, (1, maxSampleLen, channels*numInput[0]*numInput[1]))
        batchY = labelList[inpFl[:2]]
        batchY = np.reshape(batchY, (1, numClasses))

        acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, seqLen: maxSampleLen})
        acc_total += acc
    print("Testing Accuracy:", acc_total/testSteps)

    plt.plot(loses)
    plt.show()
    plt.plot(accur)
    plt.show()
