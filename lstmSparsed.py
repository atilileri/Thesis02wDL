import pickle
import tensorflow as tf
import os
import numpy as np
import sys
from datetime import datetime


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, "rb"))


def rnnFunc(xParam, weiParam, biaParam, cell):

    # generate prediction
    outputs, states = tf.nn.dynamic_rnn(cell, xParam, dtype=tf.float32)

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weiParam['out']) + biaParam['out']


# enable for file printing
# sys.stdout = open('out.txt', 'a')

stepSize = 1000
f = open('conf.txt', 'r')
for parameterLine in f.read().splitlines():
    parameters = eval(parameterLine)

    folderInputs = parameters['inputFolder']
    learningRate = parameters['learningRate']
    numHidden = parameters['numHidden']
    numLayer = parameters['numLayer']
    trainingEpoch = parameters['trainingEpoch']
    optimizerOpt = parameters['optimizerOpt']
    featureMode = parameters['featureMode']
    cellType = parameters['cellType']

    print('================== lstmSparsed with stepSize:', stepSize, '================================================')
    print('Parameters:', parameters)
    print('==================', datetime.now().strftime('%Y.%m.%d %H:%M:%S'), '===========================', flush=True)

    inputs = []
    maxSampleLen = 0  # timestep is how long is the file(samples in the file)
    labelList = dict()

    if featureMode == 'All':
        numInput = [2, 9]  # input data size (9 inst freqs, 9 inst mags)
    elif featureMode in ['Freqs', 'Mags', 'FirstFreq', 'FirstMag']:
        numInput = [1, 9]
    else:
        numInput = None
        print('ERROR: Unrecognized parameter featureMode is: (', parameters['featureMode'], ')')

    if featureMode in ['Freqs', 'Mags', 'All']:
        channels = 4  # recorded with 4 microphones
    elif featureMode in ['FirstFreq', 'FirstMag']:
        channels = 1
    else:
        channels = None
        print('ERROR: Unrecognized parameter featureMode is: (', parameters['featureMode'], ')')

    flattenedFeatures = channels*numInput[0]*numInput[1]

    # prepare max length and file names
    print('Initial Scan. Classes:', end='')
    for rootPath, directories, files in os.walk(folderInputs):
        for filename in sorted(files):
            if '.inp2' in filename:
                inputFile = loadData(rootPath + filename)
                inputShape = np.shape(inputFile)
                if inputShape[1:] == (4, 2, 9):  # todo - ai : fix all files, extract again
                    label = filename[:2]
                    if label not in labelList:
                        print('')
                        print(label, end='')
                        labelCount = len(labelList)
                        for l in labelList:
                            labelList[l].append(0)
                        labelList[label] = (labelCount * [0])
                        labelList[label].append(1)
                    print('.', end='', flush=True)
                    inputs.append(filename)
                else:
                    print('?', end='')

    print('')
    print('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
    print('Total of', len(labelList), 'classes')

    np.random.shuffle(inputs)

    # train with 80% files, test with 20%
    totalOfInputs = len(inputs)
    trainingSteps = int(totalOfInputs * 0.8)
    testSteps = totalOfInputs - trainingSteps
    print(trainingSteps, 'steps or training,', testSteps, 'steps for test')

    numClasses = len(labelList)  # total number of classification classes (ie. people)

    tf.reset_default_graph()
    # tf Graph input
    x = tf.placeholder("float", [None, None, flattenedFeatures])  # feature set, flattened
    y = tf.placeholder("float", [None, numClasses])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([numHidden, numClasses]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([numClasses]))
    }

    cells = list()
    for c in range(numLayer):
        if 'lstm' == cellType:
            cells.append(tf.nn.rnn_cell.LSTMCell(numHidden))
        elif 'gru' == cellType:
            cells.append(tf.nn.rnn_cell.GRUCell(numHidden))
        else:
            print('ERROR: Unrecognized parameter cellType is: (', parameters['cellType'], ')')
            break

    networkStc = tf.nn.rnn_cell.MultiRNNCell(cells)
    logits = rnnFunc(x, weights, biases, networkStc)
    prediction = tf.nn.softmax(logits)

    # loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    if 'Adam' == optimizerOpt:
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)
    elif 'Grad' == optimizerOpt:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)
    else:
        optimizer = None
        print('ERROR: Unrecognized parameter optimizerOpt is: (', parameters['optimizerOpt'], ')')

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
            print("Epoch " + str(e + 1) + " /", trainingEpoch, " (", str(trainingSteps) + " training steps)", end='',
                  flush=True)
            epochStart = datetime.now()
            loss_total = 0
            acc_total = 0
            for step in range(0, trainingSteps):
                filename = inputs[step]
                inFile = loadData(folderInputs + filename)

                if 'Freqs' == featureMode:
                    inFile = inFile[:, :, 0, :]
                elif 'Mags' == featureMode:
                    inFile = inFile[:, :, 1, :]
                elif 'FirstFreq' == featureMode:
                    inFile = inFile[:, 0, 0, :]
                elif 'FirstMag' == featureMode:
                    inFile = inFile[:, 0, 1, :]

                inFile = [inFile[x] for x in range(0, len(inFile), stepSize)]
                batchX = np.reshape(inFile, (1, len(inFile), flattenedFeatures))
                batchY = labelList[filename[:2]]
                batchY = np.reshape(batchY, (1, numClasses))

                _, acc, mbloss, onehot_pred = sess.run([optimizer, accuracy, loss, logits],
                                                       feed_dict={x: batchX, y: batchY})

                loss_total += mbloss
                acc_total += acc

            loses.append(loss_total)
            accur.append(acc_total/trainingSteps)

            duration = datetime.now() - epochStart
            dur_in_secs = duration.total_seconds()
            days = divmod(dur_in_secs, 86400)  # Get days (without [0]!)
            hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
            minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
            seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds
            print(", Epoch Loss= " + "{:.4f}".format(loss_total) +
                  ", Training Accuracy= " + "{:.3f}".format(acc_total/trainingSteps) +
                  " Duration= %d days, %d hours, %d minutes, %d seconds" % (days[0], hours[0], minutes[0], seconds[0]),
                  flush=True)

        # Testing
        acc_total = 0
        for inpFl in inputs[-testSteps:]:
            inFile = loadData(folderInputs + inpFl)

            if 'Freqs' == featureMode:
                inFile = inFile[:, :, 0, :]
            elif 'Mags' == featureMode:
                inFile = inFile[:, :, 1, :]
            elif 'FirstFreq' == featureMode:
                inFile = inFile[:, 0, 0, :]
            elif 'FirstMag' == featureMode:
                inFile = inFile[:, 0, 1, :]

            inFile = [inFile[x] for x in range(0, len(inFile), stepSize)]
            batchX = np.reshape(inFile, (1, len(inFile), flattenedFeatures))
            batchY = labelList[inpFl[:2]]
            batchY = np.reshape(batchY, (1, numClasses))

            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
            acc_total += acc
        print("Testing Accuracy:", acc_total/testSteps)
        print('')
        print('')

        # plt.plot(loses)
        # plt.show()
        # plt.plot(accur)
        # plt.show()
f.close()
