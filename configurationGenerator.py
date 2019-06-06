import random

inputFolders = ['D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/']
# featureModes = ['FirstFreq', 'FirstMag', 'Freqs', 'Mags', 'All']
featureModes = ['All']
trainingEpochs = [5]
learningRates = [0.001, 0.002]
numHiddens = [2, 4, 8]
numLayers = [2, 3]
celltypes = ['lstm', 'gru']
optimizers = ['Adam', 'Grad']
stepSizes = [10]
printSteps = [10]

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
                                for ss in stepSizes:
                                    for ps in printSteps:
                                        parameters = dict()
                                        parameters['inputFolder'] = inp
                                        parameters['featureMode'] = fm
                                        parameters['trainingEpoch'] = te
                                        parameters['learningRate'] = lr
                                        parameters['numHidden'] = nh
                                        parameters['numLayer'] = nl
                                        parameters['cellType'] = ct
                                        parameters['optimizerOpt'] = opt
                                        parameters['stepSize'] = ss
                                        parameters['printStep'] = ps
                                        iteration += 1
                                        listOfParams.append(parameters.copy())
for i in range(countLimit):
    print(random.choice(listOfParams), file=f)
print(countLimit, ' random iteration(s) put into configuration file.')
f.close()
