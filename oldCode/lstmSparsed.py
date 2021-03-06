import pickle
import tensorflow as tf
import os
import numpy as np
import sys
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, 'rb'))


def rnnFunc(xParam, weiParam, biaParam, cell):
    # generate prediction
    outputs, states = tf.nn.dynamic_rnn(cell, xParam, dtype=tf.float32)

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weiParam['out']) + biaParam['out']


def sparseIt(inputToBeSparsed, stepSz):
    if stepSz > 1:
        inputToBeSparsed = inputToBeSparsed[::stepSz]
    return inputToBeSparsed


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def featureSlicer(inputToBeSliced, featureM):
    if 'Freqs' == featureM:
        inputToBeSliced = inputToBeSliced[:, :, 0, :]
    elif 'Mags' == featureM:
        inputToBeSliced = inputToBeSliced[:, :, 1, :]
    elif 'FirstFreq' == featureM:
        inputToBeSliced = inputToBeSliced[:, 0, 0, :]
    elif 'FirstMag' == featureM:
        inputToBeSliced = inputToBeSliced[:, 0, 1, :]
    # else is 'All' == featureM. No slicing needed
    return inputToBeSliced


def durationFormatter(dur):
    dur_in_secs = dur.total_seconds()
    days = divmod(dur_in_secs, 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds
    return '%d days, %d hours, %d minutes, %d seconds' % (days[0], hours[0], minutes[0], seconds[0])


# prepare max length and file names
def fileReader(folder, shuffle=True, pad=False):
    labelListLocal = dict()
    inputFilesLocal = list()
    filenamesLocal = list()
    inputsLocal = list()
    print('Initial Scan. Classes:', end='')
    for rootPath, directories, files in os.walk(folder):
        for flname in sorted(files):
            if '.inp2' in flname:
                inputFile = loadData(rootPath + flname)
                inputShape = np.shape(inputFile)
                if inputShape[1:] == (4, 2, 9):  # todo - ai : fix all files, extract again
                    label = flname[:2]
                    if label not in labelListLocal:
                        print('')
                        print(label, end='')
                        labelCount = len(labelListLocal)
                        for l in labelListLocal:
                            labelListLocal[l].append(0)
                        labelListLocal[label] = (labelCount * [0])
                        labelListLocal[label].append(1)
                    print('.', end='', flush=True)
                    inputFilesLocal.append(inputFile[:])
                    filenamesLocal.append(flname)
                else:
                    print('?', end='')
    if pad:
        inputFilesLocal = pad_sequences(inputFilesLocal, dtype='float64', padding='post')

    # todo - ai : do this better way, more pythonic way
    for idx in range(len(inputFilesLocal)):
        inputsLocal.append((filenamesLocal[idx], inputFilesLocal[idx]))

    if shuffle:
        np.random.shuffle(inputsLocal)

    return inputsLocal, labelListLocal


# enable for file printing
# sys.stdout = open('out.txt', 'a')

f = open('conf.txt', 'r')
confList = f.read().splitlines()
f.close()
totalConfigurationCount = len(confList)
print('Total of %d configurations will be run' % totalConfigurationCount)
for cIdx in range(len(confList)):
    parameters = eval(confList[cIdx])

    folderInputs = parameters['inputFolder']
    learningRate = parameters['learningRate']
    numHidden = parameters['numHidden']
    numLayer = parameters['numLayer']
    trainingEpoch = parameters['trainingEpoch']
    optimizerOpt = parameters['optimizerOpt']
    featureMode = parameters['featureMode']
    cellType = parameters['cellType']
    stepSize = parameters['stepSize']
    printStep = parameters['printStep']

    print('============ Config: %d/%d -> lstmSparsed with stepSize: %d ==============================================' %
          (cIdx+1, totalConfigurationCount, stepSize))
    print('Parameters:', parameters)
    print('==================', datetime.now().strftime('%Y.%m.%d %H:%M:%S'), '===========================', flush=True)

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

    inputs, labelList = fileReader(folderInputs)
    
    print('')
    print('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
    print('Total of', len(labelList), 'classes')

    # train with 80% files, test with 20%
    totalOfInputs = len(inputs)
    trainingSteps = int(totalOfInputs * 0.8)
    testSteps = totalOfInputs - trainingSteps
    print(trainingSteps, 'steps or training,', testSteps, 'steps for test')

    numClasses = len(labelList)  # total number of classification classes (ie. people)

    reset_graph()
    # tf Graph input
    x = tf.placeholder('float', [None, None, flattenedFeatures])  # feature set, flattened
    y = tf.placeholder('float', [None, numClasses])

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
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
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
            print('')
            print('Epoch ' + str(e + 1) + '/' + str(trainingEpoch), '(' + str(trainingSteps) +
                  ' training steps) started.', flush=True)
            epochStart = datetime.now()
            loss_total = 0
            acc_total = 0
            for step in range(0, trainingSteps):
                stepStart = datetime.now()
                filename = inputs[step][0]
                inFile = inputs[step][1]

                inFile = featureSlicer(inFile, featureMode)

                inFile = sparseIt(inFile, stepSize)

                batchX = np.reshape(inFile, (1, len(inFile), flattenedFeatures))
                batchY = labelList[filename[:2]]
                batchY = np.reshape(batchY, (1, numClasses))

                _, acc, mbloss, onehot_pred = sess.run([optimizer, accuracy, loss, logits],
                                                       feed_dict={x: batchX, y: batchY})

                # print some steps in epoch (first, after -printStep- interval and last steps)
                if 0 == ((step+1) % printStep) or 0 == step or trainingSteps == step+1:
                    print('--Step %d/%d:' % (step+1, trainingSteps),
                          'Duration= %s' % durationFormatter(datetime.now()-stepStart),
                          'Shape= %s' % str(batchX.shape),
                          'File= %s' % filename, flush=True)
                loss_total += mbloss
                acc_total += acc

            loses.append(loss_total)
            accur.append(acc_total/trainingSteps)

            print('Epoch ' + str(e + 1) + ' Results: Loss= ' + '{:.4f}'.format(loss_total) +
                  ', Training Accuracy= ' + '{:.3f}'.format(acc_total/trainingSteps) +
                  ' Duration= %s' % durationFormatter(datetime.now() - epochStart), flush=True)

        # Testing
        acc_total = 0
        for inpFl in inputs[-testSteps:]:
            filename = inpFl[0]
            inFile = inpFl[1]

            inFile = featureSlicer(inFile, featureMode)

            inFile = sparseIt(inFile, stepSize)

            batchX = np.reshape(inFile, (1, len(inFile), flattenedFeatures))
            batchY = labelList[filename[:2]]
            batchY = np.reshape(batchY, (1, numClasses))

            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
            acc_total += acc
        print('Testing Accuracy:', acc_total/testSteps)
        print('')
        print('')

        # plt.plot(loses)
        # plt.show()
        # plt.plot(accur)
        # plt.show()
