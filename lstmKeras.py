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
def fileReader(folder, stepSz, featureM, channelM, classificationM, shuffle=True, pad=True):
    gc.collect()
    labelDictLocal = dict()
    labelListLocal = list()
    inputFilesLocal = list()
    maxLen = 0
    myPrint('Initial Scan.')
    for rootPath, directories, files in os.walk(folder):
        if shuffle:
            myPrint('Shuffling...')
            np.random.shuffle(files)
        myPrint('Reading:', end='')
        for flname in files:
            if ('.imfFeat' in flname and featureM in ['Freqs', 'Mags', 'Phases', 'FrMg', 'MgPh', 'FrPh', 'FrMgPh']) or \
                    ('.wav' in flname and featureM in ['Wav']) or \
                    ('.specto' in flname and featureM in ['Specto']):
                # read file
                if 'Wav' == featureM:
                    _, inputFile = scipy.io.wavfile.read(rootPath + flname)
                else:
                    inputFile = loadData(rootPath + flname)

                # read labels
                if 'Speaker' == classificationM:
                    label = flname[0:2]
                elif 'Posture' == classificationM:
                    label = flname[2:4]
                else:
                    myPrint('ERROR: Invalid classification mode:', classificationM)
                    return
                labelListLocal.append(label)
                if label not in labelDictLocal:
                    labelCount = len(labelDictLocal)
                    for l in labelDictLocal:
                        labelDictLocal[l].append(0)
                    labelDictLocal[label] = (labelCount * [0])
                    labelDictLocal[label].append(1)
                myPrint('.', end='', flush=True)

                # decimate by stepSize
                if stepSz > 1 and 'Specto' != featureM:
                    inputFile = inputFile[::stepSz]

                # update max length
                seqLen = len(inputFile)
                maxLen = max(seqLen, maxLen)

                # seperate out only wanted channel(s)
                chnSlice = None
                if 0 <= channelM <= 3:
                    chnSlice = channelM  # index: [0-3] channel (without losing axis count)
                elif 4 == channelM:
                    pass
                else:
                    myPrint('ERROR: Invalid channel mode for file read:', channelM)
                    return

                # seperate out only wanted feature(s)
                featSlice = None
                if 'Freqs' == featureM:
                    featSlice = 0  # index: 0 (without losing axis count)
                elif 'Mags' == featureM:
                    featSlice = 1  # index: 1 (without losing axis count)
                elif 'Phases' == featureM:
                    featSlice = 2  # index: 2 (without losing axis count)
                elif 'FrMg' == featureM:
                    featSlice = slice(0, 2)  # indexes: 0,1
                elif 'MgPh' == featureM:
                    featSlice = slice(1, 3)  # indexes: 1,2
                elif 'FrPh' == featureM:
                    featSlice = slice(0, 3, 2)  # indexes: 0,2
                elif featureM in ['Wav', 'Specto', 'FrMgPh']:
                    pass
                else:
                    myPrint('ERROR: Invalid feature mode for file read:', featureM)
                    return

                # apply seperation
                if (chnSlice is not None) and (featSlice is not None):
                    inputFile = inputFile[:, chnSlice, featSlice, :]
                elif (chnSlice is not None) and (featSlice is None):
                    inputFile = inputFile[:, chnSlice, ...]
                elif (chnSlice is None) and (featSlice is not None):
                    inputFile = inputFile[:, :, featSlice, :]
                else:
                    pass  # No slicing needed

                # flatten
                inputFile = np.reshape(inputFile, (seqLen, -1))

                # append each item to their lists
                inputFilesLocal.append(inputFile.copy())

                del inputFile
                gc.collect()

    myPrint('')  # for new line
    myPrint('Generating Labels...')
    # regenerate label list from final label dict as one-hot vector form
    for i in range(len(labelListLocal)):
        labelListLocal[i] = labelDictLocal[labelListLocal[i]]

    myPrint('%d Files with %d Label(s): %s.' % (len(inputFilesLocal), len(labelDictLocal), list(labelDictLocal.keys())))
    if pad:
        myPrint('Padding:', end='')
        for i in range(len(inputFilesLocal)):
            seqLen = len(inputFilesLocal[i])
            diff = maxLen - seqLen
            inputFilesLocal[i] = np.pad(inputFilesLocal[i], ((0, diff), (0, 0)), mode='constant', constant_values=0)
            myPrint('.', end='', flush=True)
        myPrint('')

    gc.collect()
    return inputFilesLocal, labelListLocal, labelDictLocal


# function name is explanatory enough
def trainTestLSTM(xTraining, xTesting, yTraining, yTesting, numCls, trainEpoch, batchSz, labelLst, losFnc, optim,
                  learnRate, featMode):
    gc.collect()
    clearGPU()
    myPrint('---LSTM Classifier---')

    trainShape = np.shape(xTraining)
    testShape = np.shape(xTesting)
    myPrint('Train Batch:', trainShape)
    myPrint('Test Batch:', testShape)

    # create the model
    model = Sequential()
    if 'Specto' != featMode:  # do not convolve for spectogram
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
                                epochs=trainEpoch, batch_size=batchSz, validation_data=(xTesting, yTesting))
    # model.fit() function prints to console but we can not grab it as it is.
    # So myPrint it only to file with given info.
    for i in range(len(trainingResults.history['loss'])):
        myPrint('Epoch #%d: Loss:%.4f, Accuracy:%.4f Validation Loss:%.4f, Validation Accuracy:%.4f'
                % (i+1, trainingResults.history['loss'][i], trainingResults.history['acc'][i],
                   trainingResults.history['val_loss'][i], trainingResults.history['val_acc'][i]), mode='file')

    # Final evaluation of the model
    myPrint('')
    myPrint('Test:')
    scores = model.evaluate(xTesting, yTesting, batch_size=testShape[0])
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


def runConfig(parameters):
    folderInputs = parameters['inputFolder']
    trainingEpoch = parameters['trainingEpoch']
    featureMode = parameters['featureMode']
    channelMode = parameters['channelMode']
    classificationMode = parameters['classificationMode']
    stepSize = parameters['stepSize']
    batchSize = parameters['batchSize']
    learningRate = parameters['learningRate']
    lossFunction = parameters['lossFunction']
    optimizer = parameters['optimizer']
    clsModel = parameters['clsModel']

    # use fileReader() for random shuffling every iteration. use some temp data for tests only. read explanations below
    inputs, labels, labelDict = fileReader(folderInputs, stepSize, featureMode, channelMode, classificationMode)

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

    myPrint('')
    myPrint('Total of ' + str(len(inputs)) + ' inputs loaded @ ' + folderInputs)
    myPrint('Total of', numClasses, 'classes')

    # train with 80%(minus remainder of batchSize) of files, test with 20%
    totalOfInputs = len(inputs)
    trainingSteps = int(totalOfInputs * 0.8)
    if (0 == batchSize) or (batchSize > trainingSteps):
        batchSize = trainingSteps
    trainingSteps -= (trainingSteps % batchSize)  # for better fit of train size, not necessary.
    testSteps = totalOfInputs - trainingSteps
    myPrint(trainingSteps, 'steps for training,', testSteps, 'steps for test')

    myPrint('Splitting Train and Test Data...')
    xTrain, xTest, yTrain, yTest = train_test_split(np.asarray(inputs), np.asarray(labels),
                                                    stratify=labels, train_size=trainingSteps, test_size=testSteps)

    myPrint('------Model for %s------' % featureMode)
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
    del labelDict
    del labels
    gc.collect()


# ===================================== MAIN STARTS HERE. FUNCTIONS ARE ABOVE =====================================

# enable for capturing console to file (does not print to console this way, but captures all stdout from other libs too)
# myPrint() function on the other hand, writes to both console and file, but does not capture stdout
# sys.stdout = open('out.txt', 'a')
myPrint('======= Running File: %s =======' % sys.argv[0])
if 2 == len(sys.argv):
    fConf = open(sys.argv[1], 'r')
    myPrint('Reading Configuration from command line argument: %s' % os.path.realpath(fConf.name))
else:
    fConf = open('conf.txt', 'r')
    myPrint('Reading Default Configuration: %s' % os.path.realpath(fConf.name))
configList = fConf.read().splitlines()
fConf.close()
totalConfigurationCount = len(configList)
myPrint('Total of %d configuration(s) will be run' % totalConfigurationCount)
for cIdx in range(totalConfigurationCount):
    gc.collect()
    parameterList = eval(configList[cIdx])
    startTime = datetime.now()
    myPrint('============ Config: %d/%d === Start Time: %s =======================================' %
            (cIdx+1, totalConfigurationCount, startTime.strftime('%Y.%m.%d %H:%M:%S')))
    myPrint('Parameters:', parameterList)

    runConfig(parameterList)

    endTime = datetime.now()
    myPrint('============ Config: %d/%d === End Time: %s =========================================' %
            (cIdx+1, totalConfigurationCount, endTime.strftime('%Y.%m.%d %H:%M:%S')))
    myPrint('============ Config: %d/%d === Duration: %s =====================' %
            (cIdx + 1, totalConfigurationCount, durToStr(endTime - startTime)))
    myPrint('', flush=True)
    gc.collect()

if fOut is not None:
    fOut.close()
