import pickle
import os
import sys
import numpy as np
from datetime import datetime
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.convolutional import Conv1D
from keras import optimizers
from keras import losses
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from math import sqrt
import gc
import scipy.io.wavfile
import pandas as pd
import kNNDTW
import utils2


def durToStr(dur):
    dur_in_secs = dur.total_seconds()
    days = divmod(dur_in_secs, 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds
    return '%d days, %d hours, %d minutes, %d seconds' % (days[0], hours[0], minutes[0], seconds[0])


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, 'rb'))


# Saves variable data to file
def saveData(data, path):
    pickle.dump(data, open(path, "wb"))


def clearGPU():
    backend.clear_session()


# prepare input for given params
def fileReader(folder, stepSz, sampRt, featureM, channelM, classificationM, shuffle=True, pad=True):
    gc.collect()
    labelDictLocal = dict()
    labelListLocal = list()
    inputFilesLocal = list()
    maxLen = 0
    r = np.random.RandomState()
    randState = r.get_state()
    imfFeatExt = 'imfFeat'
    if 8 == sampRt:
        imfFeatExt = imfFeatExt + '8khz'
    elif 48 == sampRt:
        pass
    else:
        utils2.myPrint('ERROR: Invalid sampling rate parameter:', sampRt)
        sys.exit()

    utils2.myPrint('Initial Scan.')
    for rootPath, directories, files in os.walk(folder):
        if shuffle:
            utils2.myPrint('Shuffling...')
            np.random.shuffle(files)
        utils2.myPrint('Reading:', end='')
        for flname in files:
            ext = flname.split('.')[-1]
            if (imfFeatExt == ext and featureM in ['Freqs', 'Mags', 'Phases', 'FrMg', 'MgPh', 'FrPh', 'FrMgPh',
                                                   'nFreqs', 'nMags', 'nPhases', 'FrnFr', 'MgnMg', 'PhnPh',
                                                   ]) or \
                    ('wav' == ext and featureM in ['Wav', 'Dur']) or \
                    ('specto' == ext and featureM in ['Specto']):
                # read file
                if 'Wav' == featureM:
                    _, inputFile = scipy.io.wavfile.read(rootPath + flname)
                elif 'Dur' == featureM:
                    # read duration features
                    parts = '.'.join(flname.split('.')[:-1])  # only name, without extension
                    parts = str(parts.split('_')[-1])  # 'startMs-lenMs'
                    parts = parts.split('-')
                    startMs = float(parts[0]) * 1000
                    durationMs = float(parts[1]) * 1000
                    inputFile = [startMs, durationMs]
                else:
                    inputFile = loadData(rootPath + flname)

                # read labels
                speakerId = flname[0:2]
                postureId = flname[2:4]
                if 'Speaker' == classificationM:
                    label = speakerId
                elif 'Posture5' == classificationM:
                    label = postureId
                elif 'Posture3' == classificationM:
                    if postureId in ['01', '02']:
                        label = '01'
                    elif postureId in ['03', '04']:
                        label = '02'
                    elif '05' == postureId:
                        label = '03'
                    else:
                        utils2.myPrint('ERROR: Invalid posture id:', postureId)
                        sys.exit()
                else:
                    utils2.myPrint('ERROR: Invalid classification mode:', classificationM)
                    sys.exit()
                if label not in labelDictLocal:
                    labelCount = len(labelDictLocal)
                    for l in labelDictLocal:
                        labelDictLocal[l].append(0)
                    labelDictLocal[label] = (labelCount * [0])
                    labelDictLocal[label].append(1)
                utils2.myPrint('.', end='', flush=True)

                # decimate by stepSize
                if stepSz > 1 and featureM not in ['Specto', 'Dur']:
                    inputFile = np.array(inputFile[::stepSz])
                    gc.collect()
                # update max length
                seqLen = len(inputFile)
                maxLen = max(seqLen, maxLen)

                if 'Dur' != featureM:
                    # seperate out only wanted channel(s)
                    chnSlice = None
                    if channelM in ['0', '1', '2', '3', '0Ov', '1Ov', '2Ov', '3Ov']:
                        chnSlice = int(channelM[0])  # index: [0-3] channel
                    elif 'Front' == channelM:
                        if postureId in ['01', '02']:
                            chnSlice = 1
                        elif postureId in ['03', '04']:
                            chnSlice = 2
                        elif postureId in ['05']:
                            chnSlice = 3
                        else:
                            utils2.myPrint('ERROR: Invalid posture for front microphone setting:', postureId)
                            sys.exit()
                    elif channelM in ['All', 'Split', 'SplitOv', 'AllShfUni', 'AllShfRnd']:
                        pass
                    else:
                        utils2.myPrint('ERROR: Invalid channel mode for file read:', channelM)
                        sys.exit()

                    # seperate out only wanted feature(s)
                    featSlice = None
                    if 'Freqs' == featureM:
                        featSlice = 0  # index: 0
                    elif 'Mags' == featureM:
                        featSlice = 1  # index: 1
                    elif 'Phases' == featureM:
                        featSlice = 2  # index: 2
                    elif 'nFreqs' == featureM:
                        featSlice = 3  # index: 3
                    elif 'nMags' == featureM:
                        featSlice = 4  # index: 4
                    elif 'nPhases' == featureM:
                        featSlice = 5  # index: 5
                    elif 'FrMg' == featureM:
                        featSlice = slice(0, 2)  # indexes: 0,1
                    elif 'MgPh' == featureM:
                        featSlice = slice(1, 3)  # indexes: 1,2
                    elif 'FrPh' == featureM:
                        featSlice = slice(0, 3, 2)  # indexes: 0,2
                    elif 'FrnFr' == featureM:
                        featSlice = slice(0, 4, 3)  # indexes: 0,3
                    elif 'MgnMg' == featureM:
                        featSlice = slice(1, 5, 3)  # indexes: 1,4
                    elif 'PhnPh' == featureM:
                        featSlice = slice(2, 6, 3)  # indexes: 2,5
                    elif featureM in ['Wav', 'Specto', 'FrMgPh', 'Dur']:
                        pass
                    else:
                        utils2.myPrint('ERROR: Invalid feature mode for file read:', featureM)
                        sys.exit()

                    # apply seperation
                    if (chnSlice is not None) and (featSlice is not None):
                        inputFile = inputFile[:, chnSlice, featSlice, :]
                    elif (chnSlice is not None) and (featSlice is None):
                        inputFile = inputFile[:, chnSlice, ...]
                    elif (chnSlice is None) and (featSlice is not None):
                        inputFile = inputFile[:, :, featSlice, :]
                    else:
                        pass  # No slicing needed

                    if channelM in ['Split', 'SplitOv']:
                        # make channels first dimension
                        inputFile = np.swapaxes(inputFile, 0, 1)

                        for inpFl in inputFile:
                            # flatten each channel
                            inpFl = np.reshape(inpFl, (seqLen, -1)).copy()
                            gc.collect()
                            if 'SplitOv' == channelM:
                                inpFl = np.lib.stride_tricks.as_strided(inpFl,
                                                                        (seqLen - 3, inpFl.shape[-1] * 4),
                                                                        inpFl.strides, writeable=False)
                            # append each channel as seperate items
                            inputFilesLocal.append(inpFl.copy())
                            labelListLocal.append(label)
                            del inpFl
                            gc.collect()
                    else:
                        if channelM in ['AllShfUni', 'AllShfRnd']:
                            # make channels first dimension
                            inputFile = np.swapaxes(inputFile, 0, 1)
                            if 'AllShfUni' == channelM:  # randomize each config's channel order, not each file
                                np.random.set_state(randState)  # randomize channels in unison way for each config
                            elif 'AllShfRnd' == channelM:
                                pass  # randomize channel order of each and every file differently
                            else:
                                utils2.myPrint('ERROR: Invalid channel mode for randomization:', channelM)
                                sys.exit()
                            # shuffle channels
                            np.random.shuffle(inputFile)
                            # set dimension order to default
                            inputFile = np.swapaxes(inputFile, 0, 1)

                        # flatten
                        inputFile = np.reshape(inputFile, (seqLen, -1)).copy()
                        gc.collect()
                        if channelM in ['0Ov', '1Ov', '2Ov', '3Ov']:
                            inputFile = np.lib.stride_tricks.as_strided(inputFile, (seqLen-3, inputFile.shape[-1]*4),
                                                                        inputFile.strides, writeable=False)
                        # append each item to their lists
                        inputFilesLocal.append(inputFile.copy())
                        labelListLocal.append(label)
                else:  # featureM == 'Dur'
                    inputFilesLocal.append(inputFile.copy())
                    labelListLocal.append(label)
                del inputFile
                gc.collect()

    # beacuse of overlapping, input lengths are reduced by 3.
    if channelM in ['0Ov', '1Ov', '2Ov', '3Ov', 'SplitOv'] and 'Dur' != featureM:
        maxLen -= 3
    utils2.myPrint('')  # for new line
    utils2.myPrint('Generating Labels...')
    # regenerate label list from final label dict as one-hot vector form
    for i in range(len(labelListLocal)):
        labelListLocal[i] = labelDictLocal[labelListLocal[i]]

    utils2.myPrint('%d Files with %d Label(s): %s.' % (len(inputFilesLocal), len(labelDictLocal),
                                                       list(labelDictLocal.keys())))
    if pad and 'Dur' != featureM:
        utils2.myPrint('Padding:', end='')
        for i in range(len(inputFilesLocal)):
            seqLen = len(inputFilesLocal[i])
            diff = maxLen - seqLen
            inputFilesLocal[i] = np.pad(inputFilesLocal[i], ((0, diff), (0, 0)), mode='constant', constant_values=0)
            utils2.myPrint('.', end='', flush=True)
            gc.collect()
        utils2.myPrint('')

    gc.collect()
    return inputFilesLocal, labelListLocal, labelDictLocal


# function name is explanatory enough
def trainTestkNNDTW(xTraining, xTesting, yTraining, yTesting, numCls, trainEpoch, batchSz, labelLst, losFnc, optim,
                    learnRate, featMode):
    gc.collect()
    clearGPU()
    utils2.myPrint('---DTW Classifier---')

    # convert labels from one-hot vector to decimal encoding
    yTraining = np.argmax(yTraining, axis=1)
    yTesting = np.argmax(yTesting, axis=1)

    trainShape = np.shape(xTraining)
    testShape = np.shape(xTesting)

    utils2.myPrint('Train Batch:', trainShape)
    utils2.myPrint('Test Batch:', testShape)

    # create the model
    model = kNNDTW.KnnDtw(n_neighbors=1, max_warping_window=10)
    model.fit(xTraining, yTraining)
    yPredict, proba = model.predict(xTesting)

    cls_rep = classification_report(yTesting, yPredict, target_names=[l for l in labelLst])

    conf_mat = confusion_matrix(yTesting, yPredict)

    utils2.myPrint('Labels:', labelLst)
    utils2.myPrint('Confusion Matrix:')
    utils2.myPrint(pd.DataFrame(conf_mat,
                   index=['t:{:}'.format(x) for x in labelLst],
                   columns=['{:}'.format(x) for x in labelLst]))
    utils2.myPrint('Classification Report:')
    utils2.myPrint(cls_rep)

    clearGPU()
    gc.collect()
    del xTraining
    del xTesting
    del yTraining
    del yTesting
    del labelLst
    del model
    gc.collect()
    clearGPU()


# function name is explanatory enough
def trainTestLSTM(xTraining, xTesting, yTraining, yTesting, numCls, trainEpoch, batchSz, labelLst, losFnc, optim,
                  learnRate, featMode):
    gc.collect()
    clearGPU()
    utils2.myPrint('---LSTM Classifier---')

    trainShape = np.shape(xTraining)
    testShape = np.shape(xTesting)
    utils2.myPrint('Train Batch:', trainShape)
    utils2.myPrint('Test Batch:', testShape)

    # create the model
    model = Sequential()
    if 'Specto' != featMode:  # do not convolve for spectogram, since it is not as long as other modes.
        model.add(Conv1D(8, 48, strides=48, input_shape=trainShape[1:]))
        model.add(Activation('relu'))
        model.add(Conv1D(16, 24, strides=24))
        model.add(Activation('sigmoid'))
        model.add(LSTM(24, return_sequences=True))
    else:
        model.add(LSTM(24, return_sequences=True, input_shape=trainShape[1:]))

    model.add(LSTM(12, return_sequences=False))
    model.add(Dense(numCls, activation='softmax'))
    # model.add(Activation('softmax'))

    ##
    # Optimizer selection (Paper Refs: https://keras.io/optimizers/ and
    # https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/)
    ##
    if 'Adam' == optim:
        opt = optimizers.adam(lr=learnRate)
    elif 'Sgd' == optim:
        opt = optimizers.sgd(lr=learnRate, nesterov=False)  # works well with shallow networks
    elif 'SgdNest' == optim:
        opt = optimizers.sgd(lr=learnRate, nesterov=True)  # works well with shallow networks
    elif 'Adamax' == optim:
        opt = optimizers.adamax(lr=learnRate)
    elif 'Nadam' == optim:
        opt = optimizers.nadam(lr=learnRate)
    elif 'Rms' == optim:
        opt = optimizers.rmsprop(lr=learnRate)
    else:
        utils2.myPrint('ERROR: Invalid Optimizer Parameter Value:', optim)
        sys.exit()

    ##
    # Loss function selection (Paper Refs: https://keras.io/losses/ and
    # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
    ##
    if 'SparCatCrosEnt' == losFnc:
        # sparse_categorical_crossentropy uses integers for labels instead of one-hot vectors.
        # Saves memory when numClasses is big. Other than that identical to categorical_crossentropy, use that instead.
        # Do not use this before modifying labelList structure.
        los = losses.sparse_categorical_crossentropy
    elif 'CatCrosEnt' == losFnc:
        los = losses.categorical_crossentropy
    elif 'KLDiv' == losFnc:
        los = losses.kullback_leibler_divergence
    else:
        utils2.myPrint('ERROR: Invalid Loss Function Parameter Value:', losFnc)
        sys.exit()

    model.compile(loss=los, optimizer=opt, metrics=['accuracy'])
    utils2.myPrint('Optimizer:', opt)
    utils2.myPrint('Learning Rate:', backend.eval(model.optimizer.lr))
    utils2.myPrint('Loss func:', los)
    model.summary(print_fn=utils2.myPrint)

    # input('Press ENTER to continue with training:')
    utils2.myPrint('')
    utils2.myPrint('Training:')
    # Train
    trainingResults = model.fit(xTraining, yTraining,
                                epochs=trainEpoch, batch_size=batchSz, validation_data=(xTesting, yTesting))
    # model.fit() function prints to console but we can not grab it as it is.
    # So myPrint it only to file with given info.
    for i in range(len(trainingResults.history['loss'])):
        utils2.myPrint('Epoch #%d: Loss:%.4f, Accuracy:%.4f, Validation Loss:%.4f, Validation Accuracy:%.4f'
                % (i+1, trainingResults.history['loss'][i], trainingResults.history['acc'][i],
                   trainingResults.history['val_loss'][i], trainingResults.history['val_acc'][i]), mode='file')

    utils2.myPrint(trainingResults.history, mode='code')

    # Final evaluation of the model
    utils2.myPrint('')
    utils2.myPrint('Test:')
    scores = model.evaluate(xTesting, yTesting, batch_size=testShape[0])
    utils2.myPrint('Test Loss:%.8f, Accuracy:%.4f' % (scores[0], scores[1]))

    # Stats by class
    yTesting = np.argmax(yTesting, axis=1)  # Convert one-hot to index
    yTesting = [labelLst[i] for i in yTesting]
    yPredict = model.predict_classes(xTesting)
    yPredict = [labelLst[i] for i in yPredict]
    utils2.myPrint('Labels:', labelLst)
    utils2.myPrint('Confusion Matrix:')
    # cm = confusion_matrix(yTesting, yPredict, labels=labelLst)
    # utils2.myPrint(cm)
    utils2.myPrint(pd.DataFrame(confusion_matrix(yTesting, yPredict, labels=labelLst),
                   index=['t:{:}'.format(x) for x in labelLst],
                   columns=['{:}'.format(x) for x in labelLst]))
    utils2.myPrint('Classification Report:')
    utils2.myPrint(classification_report(yTesting, yPredict, labels=labelLst))

    # Detailed stats, sample by sample prints. Uncomment with caution :). Also variable names are old. Needs refactor.
    # fTest = filenames[-testSteps:]
    # yPred = model.predict_classes(xTest)
    # # yRat = model.predict(xTest)
    # yProba = model.predict_proba(xTest)
    #
    # # show the inputs and predicted outputs
    # for i in range(len(xTest)):
    #     utils2.myPrint('File: %s ==> Predicted:%s   Real:%s -> OneHot:%s '
    #           % (fTest[i], yPred[i], np.argmax(yTest[i]), yTest[i]), end='')
    #     if yPred[i] == np.argmax(yTest[i]):
    #         utils2.myPrint('+')
    #     else:
    #         utils2.myPrint('-------')
    #     utils2.myPrint('Proba= %s' % yProba[i])
    #     # utils2.myPrint("Ratios=%s" % yRat[i])

    clearGPU()
    gc.collect()
    del xTraining
    del xTesting
    del yTraining
    del yTesting
    del labelLst
    del model
    gc.collect()
    clearGPU()


# A function to find largest prime factor
def maxPrimeFactors(n):
    # Initialize the maximum prime factor variable with the lowest one
    maxPrime = -1

    # Print the number of 2s that divide n
    while n % 2 == 0:
        maxPrime = 2
        n >>= 1  # equivalent to n /= 2

    # n must be odd at this point, thus skip the even numbers and iterate only for odd integers
    for i in range(3, int(sqrt(n)) + 1, 2):
        while n % i == 0:
            maxPrime = i
            n = n / i

    # This condition is to handle the case when n is a prime number greater than 2
    if n > 2:
        maxPrime = n

    return int(maxPrime)


def trainTestSVM(xTraining, xTesting, yTraining, yTesting, labelLst):
    gc.collect()
    utils2.myPrint('---SVM Classifier---')

    utils2.myPrint('Original Train Batch:', np.shape(xTraining))
    utils2.myPrint('Original Test Batch:', np.shape(xTesting))

    # prepare data for SVM
    yTraining = np.argmax(yTraining, axis=1)
    yTesting = np.argmax(yTesting, axis=1)
    shape = np.shape(xTraining)
    divisor = maxPrimeFactors(shape[1])
    utils2.myPrint('Divisor:', divisor)
    distribute = shape[1] // divisor
    # todo - ai : shuffle data may be needed
    xTraining = np.reshape(xTraining, (-1, shape[-1]*distribute))
    xTesting = np.reshape(xTesting, (-1, shape[-1]*distribute))

    yTrn = list()
    for y in yTraining:
        yTrn.extend([y] * divisor)
    del yTraining
    yTst = list()
    for y in yTesting:
        yTst.extend([y] * divisor)
    del yTesting

    utils2.myPrint('Mini-Batched Train Batch:', np.shape(xTraining))
    utils2.myPrint('Mini-Batched Test Batch:', np.shape(xTesting))

    model = LinearSVC(verbose=True)
    utils2.myPrint('')
    utils2.myPrint('Training...')
    model.fit(xTraining, yTrn)

    utils2.myPrint('')
    utils2.myPrint('Testing...')
    yPredict = model.predict(xTesting)

    utils2.myPrint('Test Accuracy:', model.score(xTesting, yTst))
    yTst = [labelLst[i] for i in yTst]
    yPredict = [labelLst[i] for i in yPredict]
    utils2.myPrint('Labels:', labelLst)
    utils2.myPrint('Confusion Matrix:')
    utils2.myPrint(confusion_matrix(yTst, yPredict, labels=labelLst))
    utils2.myPrint('Classification Report:')
    utils2.myPrint(classification_report(yTst, yPredict, labels=labelLst))

    del xTraining
    del xTesting
    del yTrn
    del yTst
    del labelLst
    gc.collect()


def runConfig(parameters):
    folderInputs = parameters['inputFolder']
    trainingEpoch = parameters['trainingEpoch']
    featureMode = parameters['featureMode']
    channelMode = parameters['channelMode']
    classificationMode = parameters['classificationMode']
    stepSize = parameters['stepSize']
    sampRate = parameters['sampRate']
    batchSize = parameters['batchSize']
    learningRate = parameters['learningRate']
    lossFunction = parameters['lossFunction']
    optimizer = parameters['optimizer']
    clsModel = parameters['clsModel']

    if 'DTW' == clsModel:
        padding = False
    else:
        padding = True
    # use fileReader() for random shuffling every iteration. use some temp data for tests only. read explanations below
    inputs, labels, labelDict = fileReader(folderInputs, stepSize, sampRate, featureMode, channelMode,
                                           classificationMode, pad=padding)

    # Save some randomly shuffled data, then load them each run, instead of shuffling every run.
    # Best found way for comparing performances of different network variations
    # input('Before. Press ENTER to continue:')
    # # save temp data (run fileReader() with uncommenting below, only once for saving random data)
    # saveData(inputsMags, 'C:/Users/atil/Desktop/tempDataStore/inputsMags.dat')
    # saveData(filenames, 'C:/Users/atil/Desktop/tempDataStore/filenames.dat')
    # saveData(labelDict, 'C:/Users/atil/Desktop/tempDataStore/labelDict.dat')

    # # load from temp data (uncomment below for loading, comment out fileReader() function)
    # inputsMags = loadData('C:/Users/atil/Desktop/tempDataStore/inputsMags.dat')
    # filenames = loadData('C:/Users/atil/Desktop/tempDataStore/filenames.dat')
    # labelDict = loadData('C:/Users/atil/Desktop/tempDataStore/labelDict.dat')
    # input('After. Press ENTER to continue:')

    numClasses = len(labelDict)  # total number of classification classes (ie. people)

    utils2.myPrint('')
    utils2.myPrint('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
    utils2.myPrint('Total of', numClasses, 'classes')

    # train with 80%(minus remainder of batchSize) of files, test with 20%
    totalOfInputs = len(inputs)
    trainingSteps = int(totalOfInputs * 0.8)
    if (0 == batchSize) or (batchSize > trainingSteps):
        batchSize = trainingSteps
    # trainingSteps -= (trainingSteps % batchSize)  # for better fit of train size, not necessary.
    testSteps = totalOfInputs - trainingSteps
    utils2.myPrint(trainingSteps, 'steps for training,', testSteps, 'steps for test')

    utils2.myPrint('Splitting Train and Test Data...', flush=True)
    xTrain, xTest, yTrain, yTest = train_test_split(np.asarray(inputs), np.asarray(labels),
                                                    stratify=labels, train_size=trainingSteps, test_size=testSteps)

    utils2.myPrint('------Model for %s------' % featureMode)
    # todo - ai : Param 'Both' causes probable memory error: Process finished with exit code -1073741819 (0xC0000005)
    if clsModel in ['LSTM', 'Both']:
        # Classify with Keras LSTM Model
        trainTestLSTM(xTrain, xTest, yTrain, yTest, numClasses, trainingEpoch, batchSize, list(labelDict.keys()),
                      lossFunction, optimizer, learningRate, featureMode)
    gc.collect()
    if clsModel in ['SVM', 'Both']:
        # Classify with SkLearn SVM Model
        trainTestSVM(xTrain, xTest, yTrain, yTest, list(labelDict.keys()))
    gc.collect()
    if clsModel in ['DTW', 'Both']:
        # Classify with kNN DTW model
        trainTestkNNDTW(xTrain, xTest, yTrain, yTest, numClasses, trainingEpoch, batchSize, list(labelDict.keys()),
                        lossFunction, optimizer, learningRate, featureMode)
    gc.collect()

    del xTrain
    del yTrain
    del xTest
    del yTest
    del inputs
    del labelDict
    del labels
    gc.collect()


# ===================================== MAIN STARTS HERE. FUNCTIONS ARE ABOVE =====================================

# enable for capturing console to file (does not print to console this way, but captures all stdout from other libs too)
# utils2.myPrint() function on the other hand, writes to both console and file, but does not capture stdout
# sys.stdout = open('out.txt', 'a')
utils2.myPrint('======= Running File: %s =======' % sys.argv[0])
if 2 == len(sys.argv):
    fConf = open(sys.argv[1], 'r')
    utils2.myPrint('Reading Configuration from command line argument: %s' % os.path.realpath(fConf.name))
else:
    fConf = open('conf.txt', 'r')
    utils2.myPrint('Reading Default Configuration: %s' % os.path.realpath(fConf.name))
configList = fConf.read().splitlines()
fConf.close()
totalConfigurationCount = len(configList)
utils2.myPrint('Total of %d configuration(s) will be run' % totalConfigurationCount)
for cIdx in range(totalConfigurationCount):
    gc.collect()
    parameterList = eval(configList[cIdx])
    confStartTime = datetime.now()
    utils2.myPrint('============ Config: %d/%d === Start Time: %s =======================================' %
                   (cIdx + 1, totalConfigurationCount, confStartTime.strftime('%Y.%m.%d %H:%M:%S')))
    utils2.myPrint('Parameters:', parameterList)

    runConfig(parameterList)

    confEndTime = datetime.now()
    utils2.myPrint('============ Config: %d/%d === End Time: %s =========================================' %
                   (cIdx + 1, totalConfigurationCount, confEndTime.strftime('%Y.%m.%d %H:%M:%S')))
    utils2.myPrint('============ Config: %d/%d === Duration: %s =====================' %
                   (cIdx + 1, totalConfigurationCount, durToStr(confEndTime - confStartTime)))
    utils2.myPrint('', flush=True)
    utils2.myPrint('Ending script after plotting results...')
    gc.collect()

# save results in notebook and and the script
utils2.endPrint()
