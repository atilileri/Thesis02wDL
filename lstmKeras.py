import pickle
# import tensorflow as tf
import os
import numpy as np
import sys
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.convolutional import Conv1D
from keras import optimizers
from keras import losses
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
from collections import Counter


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, 'rb'))


# Saves variable data to file
def saveData(data, path):
    pickle.dump(data, open(path, "wb"))


# todo - ai : check if commented lines affect anything
def reset_graph():
    # if 'sess' in globals() and sess:
    #     sess.close()
    backend.clear_session()


# prepare input for 2 models
def fileReader2Model(folder, stepSz, shuffle=True, pad=True):
    labelListLocal = dict()
    inputFilesLocal = list()
    filenamesLocal = list()
    print('Initial Scan. Classes:', end='')
    for rootPath, directories, files in os.walk(folder):
        for flname in sorted(files):
            if '.inp2' in flname:
                inputFile = loadData(rootPath + flname)
                inpShape = np.shape(inputFile)
                if inpShape[1:] == (4, 2, 9):
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

                    # decimate by stepSize
                    if stepSz > 1:
                        inputFile = inputFile[::stepSz]

                    # append to return lists
                    inputFilesLocal.append(inputFile)
                    filenamesLocal.append(flname)
                else:
                    print('?', end='')
    if pad:
        inputFilesLocal = pad_sequences(inputFilesLocal, dtype='float64', padding='post')

    # todo - ai : do this better way, more pythonic way
    inputsLocal = list()
    for idx in range(len(inputFilesLocal)):
        inputsLocal.append([filenamesLocal[idx], inputFilesLocal[idx]])

    if shuffle:
        np.random.shuffle(inputsLocal)

    # maxLength = max([len(j) for j in inputFilesLocal])

    filenamesLocal = [k[0] for k in inputsLocal]

    inputFilesLocal = [l[1] for l in inputsLocal]

    # split by features
    inputFilesMags = list()
    inputFilesFreqs = list()
    for inp in inputFilesLocal:
        inputFileMag = inp[:, :, 1, :]
        inputFileFreq = inp[:, :, 0, :]

        # flatten
        inpShape = np.shape(inputFileMag)
        inputFileMag = np.reshape(inputFileMag, (inpShape[0], -1))
        inpShape = np.shape(inputFileFreq)
        inputFileFreq = np.reshape(inputFileFreq, (inpShape[0], -1))

        # append to their arrays
        inputFilesMags.append(inputFileMag)
        inputFilesFreqs.append(inputFileFreq)

    return inputFilesMags, inputFilesFreqs, filenamesLocal, labelListLocal


# prepare max length and file names
def fileReader(folder, stepSz, channelM, featureM, shuffle=True, pad=True):
    labelListLocal = dict()
    inputFilesLocal = list()
    filenamesLocal = list()
    print('Initial Scan. Classes:', end='')
    for rootPath, directories, files in os.walk(folder):
        for flname in sorted(files):
            if '.inp2' in flname:
                inputFile = loadData(rootPath + flname)
                inpShape = np.shape(inputFile)
                if inpShape[1:] == (4, 2, 9):
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

                    # decimate by stepSize
                    if stepSz > 1:
                        inputFile = inputFile[::stepSz]

                    # decimate by features
                    if 'Mags' == featureM:
                        inputFile = inputFile[:, :, np.newaxis, 1, :]
                    elif 'Freqs' == featureM:
                        inputFile = inputFile[:, :, np.newaxis, 0, :]
                    elif 'All' == featureM:
                        pass
                    else:
                        print('!!!!!!!!!!!! FeatureMode ERROR', featureM)

                    # decimate by channels
                    if 'First' == channelM:
                        inputFile = inputFile[:, np.newaxis, 0, :, :]
                    elif 'Seperated' == channelM:
                        seperated = list()
                        for c in range(4):
                            seperated.append(inputFile[:, np.newaxis, c, :, :])
                        inputFile = seperated
                    elif 'All' == channelM:
                        pass
                    else:
                        print('!!!!!!!!!!!! channelMode ERROR', channelM)

                    # flatten
                    inpShape = np.shape(inputFile)
                    if 'Seperated' == channelM:
                        for c in range(len(inputFile)):
                            inputFile[c] = np.reshape(inputFile[c], (inpShape[1], -1))
                    else:
                        inputFile = np.reshape(inputFile, (inpShape[0], -1))

                    # append to return lists
                    if 'Seperated' == channelM:
                        for c in inputFile:
                            inputFilesLocal.append(c)
                            filenamesLocal.append(flname)
                    else:
                        inputFilesLocal.append(inputFile)
                        filenamesLocal.append(flname)
                else:
                    print('?', end='')
    if pad:
        inputFilesLocal = pad_sequences(inputFilesLocal, dtype='float64', padding='post')

    # todo - ai : do this better way, more pythonic way
    inputsLocal = list()
    for idx in range(len(inputFilesLocal)):
        inputsLocal.append([filenamesLocal[idx], inputFilesLocal[idx]])

    if shuffle:
        np.random.shuffle(inputsLocal)

    # maxLength = max([len(j) for j in inputFilesLocal])

    filenameListLocal = [k[0] for k in inputsLocal]

    inputFileListLocal = [l[1] for l in inputsLocal]

    return inputFileListLocal, filenameListLocal, labelListLocal


def trainTestModel(xTrain, xTest, yTrain, yTest, numCls):
    reset_graph()

    trainShape = np.shape(xTrain)
    testShape = np.shape(xTest)
    print('Train Batch:', trainShape)
    print('Test Batch:', testShape)

    # create the model
    model = Sequential()
    # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(8, 48, strides=48, input_shape=trainShape[1:]))
    model.add(Activation('relu'))
    model.add(Conv1D(16, 24, strides=24))
    model.add(Activation('sigmoid'))
    # model.add(Conv1D(32, 24, strides=12))
    # model.add(Activation('sigmoid'))
    # model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(24, return_sequences=True))
    model.add(LSTM(12, return_sequences=False))
    model.add(Dense(numCls, activation='softmax'))
    # model.add(Activation('softmax'))

    ##
    # Optimizer selection (Paper Refs: https://keras.io/optimizers/ and
    # https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/)
    ##
    opt = optimizers.rmsprop(lr=0.003)
    # opt = optimizers.sgd(lr=0.01, nesterov=False)  # works well with shallow networks
    # opt = optimizers.adamax(lr=0.002)
    # opt = optimizers.nadam(lr=0.002)
    # opt = optimizers.adam(lr=0.001)

    ##
    # Loss function selection (Paper Refs: https://keras.io/losses/ and
    # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
    ##
    # sparse_categorical_crossentropy uses integers for labels instead of one-hot vectors.
    # Saves memory when numClasses is big. Other than that identical to categorical_crossentropy, use that instead.
    # los = losses.sparse_categorical_crossentropy

    # los = losses.kullback_leibler_divergence
    los = losses.categorical_crossentropy

    model.compile(loss=los, optimizer=opt, metrics=['accuracy'])
    print('Optimizer:', opt)
    print('Loss func:', los)
    print(model.summary())

    # input('Press ENTER to continue with training:')
    print('')
    print('Training:')
    # Train
    model.fit(xTrain, yTrain, epochs=trainingEpoch, batch_size=trainingSteps)
    # Final evaluation of the model
    print('')
    print('Test:')
    scores = model.evaluate(xTest, yTest, batch_size=testSteps)
    print(model.metrics_names)
    print(scores)
    #     print('%s: %.2f%% ' % (model.metrics_names[i], (scores[i]*100)), end='')
    print('')

    # Stats by class
    Y_test = np.argmax(yTest, axis=1)  # Convert one-hot to index
    y_pred = model.predict_classes(xTest)
    # Counter(yTrain)
    # Counter(yTest)
    print(classification_report(Y_test, y_pred))

    # fTest = filenames[-testSteps:]
    # yPred = model.predict_classes(xTest)
    # # yRat = model.predict(xTest)
    # yProba = model.predict_proba(xTest)
    #
    # # show the inputs and predicted outputs
    # for i in range(len(xTest)):
    #     print('File: %s ==> Predicted:%s   Real:%s -> OneHot:%s '
    #           % (fTest[i], yPred[i], np.argmax(yTest[i]), yTest[i]), end='')
    #     if yPred[i] == np.argmax(yTest[i]):
    #         print('+')
    #     else:
    #         print('-------')
    #     print('Proba= %s' % yProba[i])
    #     # print("Ratios=%s" % yRat[i])


# enable for file printing
# sys.stdout = open('out.txt', 'a')

f = open('conf.txt', 'r')
confList = f.read().splitlines()
f.close()
totalConfigurationCount = len(confList)
print('Total of %d configuration(s) will be run' % totalConfigurationCount)
for cIdx in range(len(confList)):
    parameters = eval(confList[cIdx])

    folderInputs = parameters['inputFolder']
    # todo - ai : fix folder, remove explicit setting
    # folderInputs = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/'
    learningRate = parameters['learningRate']
    numHidden = parameters['numHidden']
    numLayer = parameters['numLayer']
    trainingEpoch = parameters['trainingEpoch']
    optimizerOpt = parameters['optimizerOpt']
    featureMode = parameters['featureMode']
    channelMode = parameters['channelMode']
    cellType = parameters['cellType']
    stepSize = parameters['stepSize']
    printStep = parameters['printStep']

    print('============ Config: %d/%d -> lstmKeras with stepSize: %d ==============================================' %
          (cIdx+1, totalConfigurationCount, stepSize))
    print('Parameters:', parameters)
    print('==================', datetime.now().strftime('%Y.%m.%d %H:%M:%S'), '===========================', flush=True)

    # todo - ai : use this for random shuffling. use temp data for tests only
    # inputsMags, inputsFreqs, filenames, labelList = fileReader2Model(folderInputs, stepSize)

    # load from temp data
    inputsMags = loadData('C:/Users/atil/Desktop/tempDataStore/inputsMags.dat')
    inputsFreqs = loadData('C:/Users/atil/Desktop/tempDataStore/inputsFreqs.dat')
    filenames = loadData('C:/Users/atil/Desktop/tempDataStore/filenames.dat')
    labelList = loadData('C:/Users/atil/Desktop/tempDataStore/labelList.dat')

    numClasses = len(labelList)  # total number of classification classes (ie. people)

    print('')
    print('Total of ' + str(len(inputsMags)) + ' inputs loaded @ ' + folderInputs)
    print('Total of', numClasses, 'classes')

    # train with 80% files, test with 20%
    totalOfInputs = len(inputsMags)
    trainingSteps = int(totalOfInputs * 0.8)
    testSteps = totalOfInputs - trainingSteps
    print(trainingSteps, 'steps or training,', testSteps, 'steps for test')

    # create labels for inputs
    labels = list()
    for f in filenames:
        labels.append(labelList[f[:2]])

    # todo - ai : use stratified train_test_split.
    yTraining = np.asarray(labels[:trainingSteps])
    yTesting = np.asarray(labels[-testSteps:])

    xTrainMag = np.asarray(inputsMags[:trainingSteps])
    xTestMag = np.asarray(inputsMags[-testSteps:])

    print('------Model for Mags------')
    trainTestModel(xTrainMag, xTestMag, yTraining, yTesting, numClasses)

    xTrainFreq = np.asarray(inputsFreqs[:trainingSteps])
    xTestFreq = np.asarray(inputsFreqs[-testSteps:])

    print('------Model for Freqs------')
    trainTestModel(xTrainFreq, xTestFreq, yTraining, yTesting, numClasses)

    # # todo - ai : try concatanating on axis 2, remove duplicating y labels. My GPU is not enough :)
    # xTrainAll = np.concatenate((xTrainMag, xTrainFreq), axis=0)
    # xTestAll = np.concatenate((xTestMag, xTestFreq), axis=0)
    #
    # yTraining = np.concatenate((yTraining, yTraining), axis=0)
    # yTesting = np.concatenate((yTesting, yTesting), axis=0)
    #
    # print('------Model for All------')
    # trainTestModel(xTrainAll, xTestAll, yTraining, yTesting, numClasses)

