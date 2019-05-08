import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import numpy as np

# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, "rb"))


folderInputs = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/'
inputs = []
lengths = []
maxSampleLen = 0  # timestep is how long is the file(samples in the file)

for rootPath, directories, files in os.walk(folderInputs):
    for filename in files:
        if '.inp2' in filename:
            # print('Extracting Features of:', filename, '\t\t @', rootPath)
            filepath = rootPath + filename
            inputFile = loadData(filepath)
            maxSampleLen = max(maxSampleLen, len(inputFile))
            lengths.append(len(inputFile))
            inputs.append(inputFile)
print('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)

for i in range(len(inputs)):
    diff = maxSampleLen - len(inputs[i])
    inputs[i] = np.pad(inputs[i], ((0, diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
print(np.shape(inputs))

# shape is (fileCount, timeSteps(max sampleCount), channelCount, 2(instf and mag), imfCount)

# Training Parameters
learningRate = 0.001
# todo - ai : create inputs for all breath examples after constructing network structure
# train with 8 files, test with 2 (total of 10 files for draft code)
totalOfInputs = len(inputs)
trainingSteps = 8
testSteps = totalOfInputs - trainingSteps

numInput = [2, 9]  # input data size (9 inst freqs, 9 inst mags)
channels = 4  # recorded with 4 microphones
timeSteps = maxSampleLen

# number of units in RNN cell
numHidden = 512
# todo - ai : increase classes to total number of people after constructing nw structure
numClasses = 2  # total number of classification classes (ie. people)

# tf Graph input
x = tf.placeholder("float", [None, timeSteps, channels, numInput[0], numInput[1]])
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

    # Manipulate x here for correct shape, if not
    xParam = tf.unstack(xParam, axis=1)
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

# model evaluation
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    # acc_total = 0
    # loss_total = 0

    # run initializer
    sess.run(init)

    for step in range(0, trainingSteps):
        batchX = inputs[step]
        batchY = list(range(0, numClasses))

        # Manipulate x here for correct shape, if not

        _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, loss, logits],
                                             feed_dict={x: batchX, y: batchY, seqLen: lengths[step]})

        # loss_total += loss
        # acc_total += acc

        print("Step " + str(step+1) + ", Minibatch Loss= " +
              "{:.4f}".format(loss) + ", Training Accuracy= " +
              "{:.3f}".format(acc))

        print("Optimization Finished!")

