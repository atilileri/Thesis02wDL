import random

inputFolders = ['D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/']
# featureModes = ['FirstFreq', 'FirstMag', 'Freqs', 'Mags', 'All']
featureModes = ['All']
trainingEpochs = [10]
learningRates = [0.001, 0.002, 0.008]
numHiddens = [2, 4, 8, 16]
numLayers = [1, 2, 3]
celltypes = ['lstm', 'gru']
optimizers = ['Adam', 'Grad']

f = open('conf.txt', 'w')
countLimit = 10
listOfParams = list()
iteration = 0
for inp in inputFolders:
    for fm in featureModes:
        for te in trainingEpochs:
            for lr in learningRates:
                for nh in numHiddens:
                    for nl in numLayers:
                        for ct in celltypes:
                            for opt in optimizers:
                                parameters = dict()
                                parameters['inputFolder'] = inp
                                parameters['featureMode'] = fm
                                parameters['trainingEpoch'] = te
                                parameters['learningRate'] = lr
                                parameters['numHidden'] = nh
                                parameters['numLayer'] = nl
                                parameters['cellType'] = ct
                                parameters['optimizerOpt'] = opt
                                iteration += 1
                                listOfParams.append(parameters.copy())
for i in range(countLimit):
    print(random.choice(listOfParams), file=f)
print(countLimit, ' random iteration(s) put into configuration file.')
f.close()
