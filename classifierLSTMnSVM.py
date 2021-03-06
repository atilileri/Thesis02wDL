import pickle
import os
import sys
import numpy as np
from datetime import datetime
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.layers import GRU, CuDNNGRU, CuDNNLSTM, average, Input
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers.pooling import MaxPool1D
from keras.layers.pooling import MaxPool2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
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
import joblib


def durToStr(dur):
    dur_in_secs = dur.total_seconds()
    days = divmod(dur_in_secs, 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds
    return '%d days, %d hours, %d minutes, %d seconds' % (days[0], hours[0], minutes[0], seconds[0])


# Loads data from file into variable
def loadData(path):
    try:
        return pickle.load(open(path, 'rb'))
    except (pickle.UnpicklingError, TypeError):
        utils2.myPrint('Not a pickle file, trying joblib...')
        return joblib.load(path)


# Saves variable data to file
def saveData(data, path):
    try:
        pickle.dump(data, open(path, "wb"))
    except (MemoryError, OverflowError):
        utils2.myPrint('Pickle failed to save, trying joblib...')
        joblib.dump(data, path)


def clearGPU():
    backend.clear_session()


# prepare input for given params
def fileReader(folder, stepSz, sampRt, featureM, channelM, classificationM,
               lenCutMs=0, shuffle=True, pad=True, flatten=True):
    gc.collect()
    labelDictLocal = dict()
    labelListLocal = list()
    inputFilesLocal = list()
    maxLen = 0
    r = np.random.RandomState()
    randState = r.get_state()
    imfFeatExt = 'imf'
    if 8 == sampRt:
        imfFeatExt = imfFeatExt + '08'
    elif 48 == sampRt:
        imfFeatExt = imfFeatExt + '48'
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
                    ('spct48' == ext and featureM in ['Specto']):
                # read file
                if 'Wav' == featureM:
                    if 0 < lenCutMs:
                        _, inputFile = scipy.io.wavfile.read(rootPath + flname)[:lenCutMs*sampRt]
                    else:
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
                    if 0 < lenCutMs:
                        inputFile = loadData(rootPath + flname)[:lenCutMs*sampRt]
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
                    elif 'FrMgPh' == featureM:
                        featSlice = slice(0, 3)  # indexes: 0,1,2
                    elif featureM in ['Wav', 'Specto', 'Dur']:
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
                        if flatten:
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
        msLen = maxLen / sampRt  # calculate length in milliseconds
        if stepSz > 0:
            msLen *= stepSz
        utils2.myPrint('Padding(', msLen, 'ms):', end='')
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
def trainTestkNNDTW(xTraining, xTesting, yTraining, yTesting, labelLst):
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


# ensembles a new model from multiple models
def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns


# function name is explanatory enough
def trainTestLSTM(xTraining, xTesting, yTraining, yTesting, numCls, trainEpoch, batchSz, labelLst, losFnc, optim,
                  learnRate, featMode, clsVer):
    gc.collect()
    clearGPU()
    utils2.myPrint('---LSTM Classifier---')

    trainShape = np.shape(xTraining)
    testShape = np.shape(xTesting)
    utils2.myPrint('Train Batch:', trainShape)
    utils2.myPrint('Test Batch:', testShape)

    models = list()
    for version in clsVer:
        # create the model
        model = Sequential()
        # todo - ai : possible variations
        #  try lstm decay
        #  try clipnorm and clipvalue
        #  try convlstm2d and/or concatanate two models
        #  try sgd instead of adam optimizer
        if 'Specto' != featMode:  # do not convolve for spectogram, since it is not as long as other modes.
            utils2.myPrint('Classifier Version:', version)
            if 0 == version:
                # inputs 1 # 300 epoch .3924.
                model.add(Conv1D(8, 48, strides=48, input_shape=trainShape[1:]))
                model.add(Activation('relu'))
                model.add(Conv1D(16, 24, strides=24))
                model.add(Activation('sigmoid'))
                model.add(LSTM(24, return_sequences=True))
                model.add(LSTM(12, return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 1 == version:
                model.add(Conv1D(8, 48, strides=12, activation='relu', input_shape=trainShape[1:]))
                model.add(Conv1D(16, 36, strides=6, activation='relu'))
                model.add(Conv1D(32, 24, strides=2, activation='relu'))
                model.add(Conv1D(64, 24, strides=2, activation='relu'))
                model.add(LSTM(64, return_sequences=True))
                model.add(LSTM(32, activation='relu', return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 2 == version:
                # resulted better than LSTM variant(1 == clsVer) with following configuration
                # {'inputFolder': 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/4spkr5post/', 'featureMode': 'Mags',
                # 'channelMode': '0', 'classificationMode': 'Speaker', 'trainingEpoch': 200, 'stepSize': 0,
                # 'sampRate': 48, 'batchSize': 32, 'lengthCut': 600, 'learningRate': 0.001,
                # 'lossFunction': 'CatCrosEnt', 'optimizer': 'Adam', 'clsModel': 'LSTM', 'clsVersion': 2}
                # overfitted with 1.000 accuracy(started around 30th epoch) with following configuration
                # {'inputFolder': 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/allSmall/', 'featureMode': 'Mags',
                # 'channelMode': '0', 'classificationMode': 'Speaker', 'trainingEpoch': 400, 'stepSize': 0,
                # 'sampRate': 48, 'batchSize': 32, 'lengthCut': 600, 'learningRate': 0.001,
                # 'lossFunction': 'CatCrosEnt', 'optimizer': 'Adam', 'clsModel': 'LSTM', 'clsVersion': 2}
                model.add(Conv1D(8, 48, strides=12, activation='relu', input_shape=trainShape[1:]))
                model.add(Conv1D(16, 36, strides=6, activation='relu'))
                model.add(Conv1D(32, 24, strides=2, activation='relu'))
                model.add(Conv1D(64, 24, strides=2, activation='relu'))
                model.add(GRU(64, return_sequences=True))
                model.add(GRU(32, activation='relu', return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 3 == version:  # tezde -1 burdan sonra
                model.add(Conv1D(8, 48, strides=12, activation='relu', input_shape=trainShape[1:]))
                model.add(Dropout(0.5))
                model.add(Conv1D(16, 36, strides=6, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Conv1D(32, 24, strides=2, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Conv1D(64, 24, strides=2, activation='relu'))
                model.add(LSTM(64, return_sequences=True))
                model.add(LSTM(32, activation='relu', return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 4 == version:
                model.add(Conv1D(16, 96, strides=12, activation='relu', input_shape=trainShape[1:]))
                model.add(Dropout(0.5))
                model.add(Conv1D(32, 48, strides=6, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Conv1D(64, 24, strides=2, activation='relu'))
                model.add(Dropout(0.5))
                model.add(CuDNNGRU(64, return_sequences=True))
                model.add(Dropout(0.5))
                model.add(CuDNNGRU(64, return_sequences=True))
                model.add(Dropout(0.5))
                model.add(CuDNNGRU(32, return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 5 == version:
                model.add(Conv1D(16, 96, strides=12, activation='relu', input_shape=trainShape[1:]))
                model.add(Dropout(0.5))
                model.add(Conv1D(32, 48, strides=6, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Conv1D(64, 24, strides=2, activation='relu'))
                model.add(Dropout(0.5))
                model.add(CuDNNGRU(64, return_sequences=True))
                model.add(Dropout(0.5))
                model.add(CuDNNGRU(32, return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 6 == version:
                # todo - ai : this is temp clsVer. give a static version number to successful model structures
                model.add(Conv1D(16, 96, strides=12, activation='relu', input_shape=trainShape[1:]))
                model.add(Dropout(0.2))
                model.add(Conv1D(32, 48, strides=6, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Conv1D(64, 24, strides=2, activation='relu'))
                model.add(Dropout(0.2))
                model.add(CuDNNGRU(64, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(CuDNNGRU(32, return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            elif 7 == version:
                # todo - ai : this is temp clsVer. give a static version number to successful model structures
                model.add(Conv1D(32, 96, strides=16, activation='relu', input_shape=trainShape[1:]))
                model.add(Conv1D(64, 48, strides=8, activation='relu'))
                model.add(CuDNNGRU(64, return_sequences=True))
                model.add(CuDNNGRU(32, return_sequences=False))
                model.add(Dense(numCls, activation='softmax'))
            else:
                utils2.myPrint('ERROR: Unknown Classifier Version')
                sys.exit()
        else:  # for Spectograms
            utils2.myPrint('Classifier Version: Spectogram')
            model.add(LSTM(24, activation='relu', return_sequences=True, input_shape=trainShape[1:]))
            model.add(LSTM(32, activation='relu', return_sequences=False))
            model.add(Dense(numCls, activation='softmax'))

        utils2.printModelConfig(model.get_config())

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
            # Saves memory when numCls is big. Other than that identical to categorical_crossentropy, use that instead.
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
        utils2.myPrint('Training:', flush=True)
        # prepare callbacks
        earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=45, min_delta=1e-4,
                                      restore_best_weights=True, verbose=1)
        # save best model for later use
        modelName = 'model_' + utils2.scriptStartDateTime + '_numCls-'+str(numCls) + '_loss-'+losFnc + '_opt-'+optim + \
                    '_lr-'+str(learnRate) + '_featMode-'+featMode + '_clsVer-'+str(version) + '.clsmdl'
        modelPath = './models/' + modelName
        modelSaving = ModelCheckpoint(modelPath, monitor='val_loss',
                                      mode='min', save_best_only=True, verbose=1)
        reduceLrLoss = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, cooldown=10,
                                         patience=10, min_delta=1e-4, min_lr=learnRate/32, verbose=1)
        # Train
        trainingResults = model.fit(xTraining, yTraining, epochs=trainEpoch, batch_size=batchSz,
                                    validation_data=(xTesting, yTesting),
                                    callbacks=[earlyStopping, modelSaving, reduceLrLoss])
        # model.fit() function prints to console but we can not grab it as it is.
        # So myPrint it only to file with given info.
        for i in range(len(trainingResults.history['loss'])):
            utils2.myPrint('Epoch #%d: Loss:%.4f, Accuracy:%.4f, Validation Loss:%.4f, Validation Accuracy:%.4f'
                           % (i+1, trainingResults.history['loss'][i], trainingResults.history['acc'][i],
                              trainingResults.history['val_loss'][i], trainingResults.history['val_acc'][i]), mode='file')

        utils2.myPrint(trainingResults.history, mode='code')

        utils2.myPrint('')
        # Restore best Model
        utils2.myPrint('Restoring best model...')
        model = load_model(modelPath)
        models.append(modelPath)
        # Final evaluation of the model
        utils2.myPrint('Test:')
        scores = model.evaluate(xTesting, yTesting, batch_size=testShape[0])
        utils2.myPrint('Test Loss:%.8f, Accuracy:%.4f' % (scores[0], scores[1]))

        # write results to a seperate text file (part 2/3)
        fResult = open('./Results.txt', 'a+')
        fResult.write(', ' + modelName + ', Test Loss:%.8f, Accuracy:%.4f' % (scores[0], scores[1]))
        fResult.close()

        # todo - ai : kfold cross validation can be inserted here

        # Stats by class
        yTesting1Hot = np.argmax(yTesting, axis=1)  # Convert one-hot to index
        yTesting1Hot = [labelLst[i] for i in yTesting1Hot]
        yPredict = model.predict_classes(xTesting)
        yPredict = [labelLst[i] for i in yPredict]
        utils2.myPrint('Labels:', labelLst)
        utils2.myPrint('Confusion Matrix:')
        utils2.myPrint(pd.DataFrame(confusion_matrix(yTesting1Hot, yPredict, labels=labelLst),
                                    index=['t:{:}'.format(x) for x in labelLst],
                                    columns=['{:}'.format(x) for x in labelLst]))
        utils2.myPrint('Classification Report:')
        utils2.myPrint(classification_report(yTesting1Hot, yPredict, labels=labelLst))
        clearGPU()
        del model
        gc.collect()

    if 1 < len(models):
        # Test models, ensembled
        modelId = utils2.scriptStartDateTime
        modelName = 'model_' + modelId + '_ensembled.clsmdl'
        modelPath = './models/' + modelName
        # write results to a seperate text file (part 3/3)
        for mdlIdx in range(len(models)):
            models[mdlIdx] = load_model(models[mdlIdx])
            models[mdlIdx].name = modelId + '_' + str(mdlIdx)  # change name to be unique

        model_input = Input(shape=models[0].input_shape[1:])  # c*h*w
        modelEns = ensembleModels(models, model_input)

        modelEns.compile(optimizer=optimizers.adam(lr=learnRate),
                         loss=losses.categorical_crossentropy,
                         metrics=['accuracy'])
        modelEns.summary(print_fn=utils2.myPrint)
        modelEns.save(modelPath)
        utils2.myPrint('Ensemble Test:')

        scores = modelEns.evaluate(xTesting, yTesting, batch_size=testShape[0])
        utils2.myPrint('Test Loss:%.8f, Accuracy:%.4f' % (scores[0], scores[1]))

        fResult = open('./Results.txt', 'a+')
        fResult.write(', ' + modelName + ', Test Loss:%.8f, Accuracy:%.4f' % (scores[0], scores[1]))
        fResult.close()

        # Stats by class
        yTesting1Hot = np.argmax(yTesting, axis=1)  # Convert one-hot to index
        yTesting1Hot = [labelLst[i] for i in yTesting1Hot]
        yPredict = modelEns.predict(xTesting)
        yPredict = np.argmax(yPredict, axis=1)
        yPredict = [labelLst[i] for i in yPredict]
        utils2.myPrint('Labels:', labelLst)
        utils2.myPrint('Confusion Matrix:')
        utils2.myPrint(pd.DataFrame(confusion_matrix(yTesting1Hot, yPredict, labels=labelLst),
                                    index=['t:{:}'.format(x) for x in labelLst],
                                    columns=['{:}'.format(x) for x in labelLst]))
        utils2.myPrint('Classification Report:')
        utils2.myPrint(classification_report(yTesting1Hot, yPredict, labels=labelLst))
        del modelEns

    clearGPU()
    gc.collect()
    del xTraining
    del xTesting
    del yTraining
    del yTesting
    del labelLst
    del models
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
    lengthCut = parameters['lengthCut']
    learningRate = parameters['learningRate']
    lossFunction = parameters['lossFunction']
    optimizer = parameters['optimizer']
    clsModel = parameters['clsModel']
    clsVersion = parameters['clsVersion']

    if 'DTW' == clsModel:
        padding = False
    else:
        padding = True

    saveLoadData = True
    # check if inputs prepared before
    inputsFileName = 'E:/atili/Datasets/BreathDataset/temp' \
                     'DataStorage/{}_inputs_{}_{}_{}_{}_{}_{}_{}.dat'\
        .format(os.path.basename(os.path.dirname(folderInputs)), featureMode, channelMode, classificationMode,
                stepSize, sampRate, lengthCut, padding)
    labelsFileName = 'E:/atili/Datasets/BreathDataset/temp' \
                     'DataStorage/{}_labels_{}_{}_{}_{}_{}_{}_{}.dat'\
        .format(os.path.basename(os.path.dirname(folderInputs)), featureMode, channelMode, classificationMode,
                stepSize, sampRate, lengthCut, padding)
    labelDictFileName = 'E:/atili/Datasets/BreathDataset/temp' \
                        'DataStorage/{}_labelDict_{}_{}_{}_{}_{}_{}_{}.dat'\
        .format(os.path.basename(os.path.dirname(folderInputs)), featureMode, channelMode, classificationMode,
                stepSize, sampRate, lengthCut, padding)

    if os.path.exists(inputsFileName) and os.path.exists(labelsFileName)\
            and os.path.exists(labelDictFileName) and saveLoadData:
        utils2.myPrint('Loading from Previous Data Files...')
        inputs = loadData(inputsFileName)
        utils2.myPrint('Loaded:', inputsFileName)
        labels = loadData(labelsFileName)
        utils2.myPrint('Loaded:', labelsFileName)
        labelDict = loadData(labelDictFileName)
        utils2.myPrint('Loaded:', labelDictFileName)
    else:
        # use fileReader() for random shuffling every iteration. use some temp data for tests only (using saveLoadData).
        inputs, labels, labelDict = fileReader(folderInputs, stepSize, sampRate, featureMode, channelMode,
                                               classificationMode, lengthCut, pad=padding)
        # Save some randomly shuffled data, then load them each run, instead of shuffling every run.
        # Best found way for comparing performances of different network variations
        if saveLoadData:
            utils2.myPrint('Saving Data Files for Later Use...')
            saveData(inputs, inputsFileName)
            utils2.myPrint('Saved:', inputsFileName)
            saveData(labels, labelsFileName)
            utils2.myPrint('Saved:', labelsFileName)
            saveData(labelDict, labelDictFileName)
            utils2.myPrint('Saved:', labelDictFileName)

    # write results to a seperate text file (part 1/3)
    fResult = open('./Results.txt', 'a+')
    fResult.write('\n\r ' + utils2.scriptStartDateTime + ', ')
    print(parameters, end='', file=fResult)
    fResult.close()

    utils2.myPrint('Inputs Shape:', np.shape(inputs))

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

    # todo - ai : validation can be added for once
    utils2.myPrint('Splitting Train and Test Data...', flush=True)
    xTrain, xTest, yTrain, yTest = train_test_split(np.asarray(inputs), np.asarray(labels),
                                                    stratify=labels, train_size=trainingSteps, test_size=testSteps)

    utils2.myPrint('------Model for %s------' % featureMode)
    if 'LSTM' == clsModel:
        # Classify with Keras LSTM Model
        trainTestLSTM(xTrain, xTest, yTrain, yTest, numClasses, trainingEpoch, batchSize, list(labelDict.keys()),
                      lossFunction, optimizer, learningRate, featureMode, clsVersion)
        gc.collect()
    elif 'SVM' == clsModel:
        # Classify with SkLearn SVM Model
        trainTestSVM(xTrain, xTest, yTrain, yTest, list(labelDict.keys()))
        gc.collect()
    elif 'DTW' == clsModel:
        # Classify with kNN DTW model
        trainTestkNNDTW(xTrain, xTest, yTrain, yTest, list(labelDict.keys()))
        gc.collect()
    else:
        utils2.myPrint('ERROR: Invalid Classification Model:', clsModel)
        sys.exit()

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
    utils2.myPrint('Parameters: ', end='')
    for param in parameterList:
        utils2.myPrint('%s : %s' % (param, str(parameterList[param])))

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
