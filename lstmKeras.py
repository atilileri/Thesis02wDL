import pickle
import os
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

fOut = None


# Prints to both a file and console
def myPrint(*args, mode='both', **kwargs):
    global fOut
    if (fOut is None) and (mode in ['both', 'file']):
        outFileName = './outputs/out_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.txt'
        if not os.path.exists(os.path.dirname(outFileName)):
            os.makedirs(os.path.dirname(outFileName))
        fOut = open(outFileName, 'w')
    if mode in ['both', 'file']:
        print(*args, file=fOut, **kwargs)
    if mode in ['both', 'console']:
        print(*args, **kwargs)


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, 'rb'))


# Saves variable data to file
def saveData(data, path):
    pickle.dump(data, open(path, "wb"))


def reset_graph():
    backend.clear_session()


# prepare input for given params
def fileReader(folder, stepSz, featureM, shuffle=True, pad=True):
    gc.collect()
    labelListLocal = dict()
    inputFilesLocal = list()
    filenamesLocal = list()
    maxlen = 0
    myPrint('Initial Scan.')
    for rootPath, directories, files in os.walk(folder):
        if shuffle:
            myPrint('Shuffling...')
            np.random.shuffle(files)
        myPrint('Reading:', end='')
        for flname in files:
            if '.inp2' in flname:
                inputFile = loadData(rootPath + flname)
                inpShape = np.shape(inputFile)
                if inpShape[1:] == (4, 2, 9):
                    label = flname[:2]
                    if label not in labelListLocal:
                        labelCount = len(labelListLocal)
                        for l in labelListLocal:
                            labelListLocal[l].append(0)
                        labelListLocal[label] = (labelCount * [0])
                        labelListLocal[label].append(1)
                    myPrint('.', end='', flush=True)

                    # decimate by stepSize
                    if stepSz > 1:
                        inputFile = inputFile[::stepSz]

                    # find max len
                    seqLen = len(inputFile)
                    maxlen = max(seqLen, maxlen)

                    # seperate out only wanted feature
                    if 'Mags' == featureM:
                        inputFile = inputFile[:, :, 1, :]
                    elif 'Freqs' == featureM:
                        inputFile = inputFile[:, :, 0, :]
                    else:
                        myPrint('ERROR: Valid features for file read: "Mags" | "Freqs"')
                        return
                    # flatten
                    inputFile = np.reshape(inputFile, (seqLen, -1))

                    # append each item to their lists
                    inputFilesLocal.append(inputFile)
                    filenamesLocal.append(flname)
                else:
                    myPrint('?', end='')
    myPrint('')
    myPrint('%d Files with %d Label(s): %s.' % (len(inputFilesLocal), len(labelListLocal), list(labelListLocal.keys())))
    if pad:
        myPrint('Padding:', end='')
        for i in range(len(inputFilesLocal)):
            seqLen = len(inputFilesLocal[i])
            diff = maxlen - seqLen
            inputFilesLocal[i] = np.pad(inputFilesLocal[i], ((0, diff), (0, 0)), mode='constant', constant_values=0)
            myPrint('.', end='', flush=True)
        myPrint('')

    gc.collect()
    return inputFilesLocal, filenamesLocal, labelListLocal


# function name is explanatory enough
def trainTestLSTM(xTraining, xTesting, yTraining, yTesting, numCls, batchSz, labelLst, losFnc, optim, learnRate):
    gc.collect()
    reset_graph()
    myPrint('---LSTM Classifier---')

    trainShape = np.shape(xTraining)
    testShape = np.shape(xTesting)
    myPrint('Train Batch:', trainShape)
    myPrint('Test Batch:', testShape)

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
        myPrint('ERROR: Invalid Optimizer Parameter Value:', optim)
        return

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
        myPrint('ERROR: Invalid Loss Function Parameter Value:', losFnc)
        return

    model.compile(loss=los, optimizer=opt, metrics=['accuracy'])
    myPrint('Optimizer:', opt)
    myPrint('Learning Rate:', backend.eval(model.optimizer.lr))
    myPrint('Loss func:', los)
    model.summary(print_fn=myPrint)

    # input('Press ENTER to continue with training:')
    myPrint('')
    myPrint('Training:')
    # Train
    trainingResults = model.fit(xTraining, yTraining,
                                epochs=trainingEpoch, batch_size=batchSz, validation_data=(xTesting, yTesting))
    # model.fit() function prints to console but we can not grab it as it is.
    # So myPrint it only to file with given info.
    for i in range(len(trainingResults.history['loss'])):
        myPrint('Epoch #%d: Loss:%.4f, Accuracy:%.4f Validation Loss:%.4f, Validation Accuracy:%.4f'
                % (i+1, trainingResults.history['loss'][i], trainingResults.history['acc'][i],
                   trainingResults.history['val_loss'][i], trainingResults.history['val_acc'][i]), mode='file')

    # Final evaluation of the model
    myPrint('')
    myPrint('Test:')
    scores = model.evaluate(xTesting, yTesting, batch_size=testSteps)
    myPrint('Test Loss:%.8f, Accuracy:%.4f' % (scores[0], scores[1]))

    # Stats by class
    yTesting = np.argmax(yTesting, axis=1)  # Convert one-hot to index
    yTesting = [labelLst[i] for i in yTesting]
    yPredict = model.predict_classes(xTesting)
    yPredict = [labelLst[i] for i in yPredict]
    myPrint('Labels:', labelLst)
    myPrint('Confusion Matrix:')
    myPrint(confusion_matrix(yTesting, yPredict, labels=labelLst))
    myPrint('Classification Report:')
    myPrint(classification_report(yTesting, yPredict, labels=labelLst))

    # Detailed stats, sample by sample prints. Uncomment with caution :). Also variable names are old. Needs refactor.
    # fTest = filenames[-testSteps:]
    # yPred = model.predict_classes(xTest)
    # # yRat = model.predict(xTest)
    # yProba = model.predict_proba(xTest)
    #
    # # show the inputs and predicted outputs
    # for i in range(len(xTest)):
    #     myPrint('File: %s ==> Predicted:%s   Real:%s -> OneHot:%s '
    #           % (fTest[i], yPred[i], np.argmax(yTest[i]), yTest[i]), end='')
    #     if yPred[i] == np.argmax(yTest[i]):
    #         myPrint('+')
    #     else:
    #         myPrint('-------')
    #     myPrint('Proba= %s' % yProba[i])
    #     # myPrint("Ratios=%s" % yRat[i])
    del xTraining
    del xTesting
    del yTraining
    del yTesting
    del labelLst
    del model
    gc.collect()


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
    myPrint('---SVM Classifier---')

    myPrint('Original Train Batch:', np.shape(xTraining))
    myPrint('Original Test Batch:', np.shape(xTesting))

    # prepare data for SVM
    yTraining = np.argmax(yTraining, axis=1)
    yTesting = np.argmax(yTesting, axis=1)
    shape = np.shape(xTraining)
    divisor = maxPrimeFactors(shape[1])
    myPrint('Divisor:', divisor)
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

    myPrint('Mini-Batched Train Batch:', np.shape(xTraining))
    myPrint('Mini-Batched Test Batch:', np.shape(xTesting))

    model = LinearSVC(verbose=True)
    myPrint('')
    myPrint('Training...')
    model.fit(xTraining, yTrn)

    myPrint('')
    myPrint('Testing...')
    yPredict = model.predict(xTesting)

    myPrint('Test Accuracy:', model.score(xTesting, yTst))
    yTst = [labelLst[i] for i in yTst]
    yPredict = [labelLst[i] for i in yPredict]
    myPrint('Labels:', labelLst)
    myPrint('Confusion Matrix:')
    myPrint(confusion_matrix(yTst, yPredict, labels=labelLst))
    myPrint('Classification Report:')
    myPrint(classification_report(yTst, yPredict, labels=labelLst))

    del xTraining
    del xTesting
    del yTrn
    del yTst
    del labelLst
    gc.collect()


# ===================================== MAIN STARTS HERE. FUNCTIONS ARE ABOVE =====================================

# enable for capturing console to file (does not print to console this way, but captures all stdout from other libs too)
# myPrint() function on the other hand, writes to both console and file, but does not capture stdout
# sys.stdout = open('out.txt', 'a')

fConf = open('conf.txt', 'r')
confList = fConf.read().splitlines()
fConf.close()
totalConfigurationCount = len(confList)
myPrint('Total of %d configuration(s) will be run' % totalConfigurationCount)
for cIdx in range(len(confList)):
    gc.collect()
    parameters = eval(confList[cIdx])
    folderInputs = parameters['inputFolder']
    trainingEpoch = parameters['trainingEpoch']
    featureMode = parameters['featureMode']
    stepSize = parameters['stepSize']
    batchSize = parameters['batchSize']
    learningRate = parameters['learningRate']
    lossFunction = parameters['lossFunction']
    optimizer = parameters['optimizer']
    clsModel = parameters['clsModel']

    myPrint('============ Config: %d/%d -> lstmKeras with stepSize: %d ==============================================' %
            (cIdx+1, totalConfigurationCount, stepSize))
    myPrint('Parameters:', parameters)
    myPrint('==================', datetime.now().strftime('%Y.%m.%d %H:%M:%S'), '=========================', flush=True)

    # use this for random shuffling. use temp data for tests only. read explanations below
    inputs, filenames, labelList = fileReader(folderInputs, stepSize, featureMode)

    # Save some randomly shuffled data, then load them each run, instead of shuffling every run.
    # Best found way for comparing network performances
    # input('Before. Press ENTER to continue:')
    # # save temp data (run fileReader() with uncommenting below, only once for saving random data)
    # saveData(inputsMags, 'C:/Users/atil/Desktop/tempDataStore/inputsMags.dat')
    # saveData(inputsFreqs, 'C:/Users/atil/Desktop/tempDataStore/inputsFreqs.dat')
    # saveData(filenames, 'C:/Users/atil/Desktop/tempDataStore/filenames.dat')
    # saveData(labelList, 'C:/Users/atil/Desktop/tempDataStore/labelList.dat')

    # # load from temp data (uncomment below for loading, comment out fileReader() function)
    # inputsMags = loadData('C:/Users/atil/Desktop/tempDataStore/inputsMags.dat')
    # inputsFreqs = loadData('C:/Users/atil/Desktop/tempDataStore/inputsFreqs.dat')
    # filenames = loadData('C:/Users/atil/Desktop/tempDataStore/filenames.dat')
    # labelList = loadData('C:/Users/atil/Desktop/tempDataStore/labelList.dat')
    # input('After. Press ENTER to continue:')

    numClasses = len(labelList)  # total number of classification classes (ie. people)

    myPrint('')
    myPrint('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
    myPrint('Total of', numClasses, 'classes')

    # train with 80%(minus remainder of batchSize) of files, test with 20%
    totalOfInputs = len(inputs)
    trainingSteps = int(totalOfInputs * 0.8)
    if 0 == batchSize:
        batchSize = trainingSteps
    trainingSteps -= (trainingSteps % batchSize)  # for better fit of train size, not necessary.
    testSteps = totalOfInputs - trainingSteps
    myPrint(trainingSteps, 'steps for training,', testSteps, 'steps for test')

    # create labels for inputs
    labels = list()
    for f in filenames:
        labels.append(labelList[f[:2]])

    myPrint('Splitting Train and Test Data...')
    xTrain, xTest, yTrain, yTest = train_test_split(np.asarray(inputs), np.asarray(labels),
                                                    stratify=labels, train_size=trainingSteps, test_size=testSteps)

    myPrint('------Model for %s------' % featureMode)
    # todo - ai : Param 'Both' causes probable memory error: Process finished with exit code -1073741819 (0xC0000005)
    if clsModel in ['LSTM', 'Both']:
        # Classify with Keras LSTM Model
        trainTestLSTM(xTrain, xTest, yTrain, yTest, numClasses, batchSize, list(labelList.keys()),
                      lossFunction, optimizer, learningRate)
    gc.collect()
    if clsModel in ['SVM', 'Both']:
        # Classify with SkLearn SVM Model
        trainTestSVM(xTrain, xTest, yTrain, yTest, list(labelList.keys()))
    gc.collect()

    # # todo - ai : try concatanating on Mags and Freqs on axis 2, may be this can give better accuracy. Try!
    # # todo cont.: remove duplicating y labels. My GPU is not enough :)
    # xTrainAll = np.concatenate((xTrainMag, xTrainFreq), axis=0)
    # xTestAll = np.concatenate((xTestMag, xTestFreq), axis=0)
    #
    # yTraining = np.concatenate((yTraining, yTraining), axis=0)
    # yTesting = np.concatenate((yTesting, yTesting), axis=0)
    #
    # myPrint('------Model for All------')
    # trainTestLSTM(xTrainAll, xTestAll, yTraining, yTesting, numClasses)

    del xTrain
    del yTrain
    del xTest
    del yTest
    del inputs
    del labelList
    del filenames
    del labels
    gc.collect()

if fOut is not None:
    fOut.close()
