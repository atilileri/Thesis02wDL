import random

# some other input paths used
# inputFolders = ['E:/atili/Datasets/BreathDataset/Processed/inputsFrom_mini_sample_set/']
# inputFolders = ['E:/atili/Datasets/BreathDataset/Processed/inputsFrom_mid_sample_set/']
# inputFolders = ['E:/atili/Datasets/BreathDataset/Processed_Small/inputsFrom_20190608_143805/']

inputFolders = ['D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_max_sample_set/']
# featureModes = ['Freqs', 'Mags', 'Phases', 'FrMg', 'MgPh', 'FrPh', 'FrMgPh', 'Wav', 'Specto',
#                 'nFreqs', 'nMags', 'nPhases', 'FrnFr', 'MgnMg', 'PhnPh']
featureModes = ['Freqs', 'Mags', 'Phases', 'nFreqs', 'nMags', 'nPhases']
# channelModes = ['0', '1', '2', '3', 'Front', 'Split', '0Ov', '1Ov', '2Ov', '3Ov', 'All', 'AllShfUni', 'AllShfRnd',
#                 'SplitOv']
channelModes = ['SplitOv']
# classificationModes = ['Speaker', 'Posture5', 'Posture3']
classificationModes = ['Speaker', 'Posture5', 'Posture3']
# sampRates = [8, 48]
sampRates = [48]
# stepSizes = [1, 4, 6, 16]
stepSizes = [4]
trainingEpochs = [300]
batchSizes = [512]
learningRates = [0.001]
# lossFunctions = ['CatCrosEnt', 'KLDiv']
lossFunctions = ['CatCrosEnt']
# optimizers = ['Adam', 'Sgd', 'SgdNest', 'Adamax', 'Nadam', 'Rms']
optimizers = ['Adam']
# models = ['LSTM', 'SVM', 'DTW']
models = ['LSTM']

# fileCreationMode = 'One'  # appends all to 'conf0.txt'
fileCreationMode = 'Each'  # creates N different 'conf[1-N].txt' files
fRunner = open('configurationRunner.bat', 'w+')

countLimit = 0
fileCounter = 0
listOfParams = list()
for inp in inputFolders:
    for sr in sampRates:
        for fm in featureModes:
            for cm in channelModes:
                for clm in classificationModes:
                    for te in trainingEpochs:
                        for ss in stepSizes:
                            for bs in batchSizes:
                                for lr in learningRates:
                                    for lf in lossFunctions:
                                        for o in optimizers:
                                            for m in models:
                                                parameters = dict()
                                                parameters['inputFolder'] = inp
                                                parameters['sampRate'] = sr
                                                parameters['featureMode'] = fm
                                                parameters['channelMode'] = cm
                                                parameters['classificationMode'] = clm
                                                parameters['trainingEpoch'] = te
                                                parameters['stepSize'] = ss
                                                parameters['batchSize'] = bs
                                                parameters['learningRate'] = lr
                                                parameters['lossFunction'] = lf
                                                parameters['optimizer'] = o
                                                parameters['clsModel'] = m
                                                listOfParams.append(parameters.copy())
if 0 < countLimit:
    fConf = None
    for i in range(countLimit):
        if 'One' == fileCreationMode and 0 == i:
            fConf = open('./confFiles/conf' + str(fileCounter) + '.txt', 'w+')
            print('python classifierLSTMnSVM.py ./confFiles/conf' + str(fileCounter) + '.txt', file=fRunner)
        elif 'Each' == fileCreationMode:
            fileCounter += 1
            if fConf is not None:
                fConf.close()
            fConf = open('./confFiles/conf' + str(fileCounter) + '.txt', 'w+')
            print('python classifierLSTMnSVM.py ./confFiles/conf' + str(fileCounter) + '.txt', file=fRunner)
        print(random.choice(listOfParams), file=fConf)
    if fConf is not None:
        fConf.close()
else:
    countLimit = len(listOfParams)
    fConf = None
    for i in range(len(listOfParams)):
        if 'One' == fileCreationMode and 0 == i:
            fConf = open('./confFiles/conf' + str(fileCounter) + '.txt', 'w+')
            print('python classifierLSTMnSVM.py ./confFiles/conf' + str(fileCounter) + '.txt', file=fRunner)
        elif 'Each' == fileCreationMode:
            fileCounter += 1
            if fConf is not None:
                fConf.close()
            fConf = open('./confFiles/conf' + str(fileCounter) + '.txt', 'w')
            print('python classifierLSTMnSVM.py ./confFiles/conf' + str(fileCounter) + '.txt', file=fRunner)
        print(listOfParams[i], file=fConf)
    if fConf is not None:
        fConf.close()
print(countLimit, ' random iteration(s) put into configuration file.')
fRunner.close()
