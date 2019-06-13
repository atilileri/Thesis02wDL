import random

# some other input paths used
# inputFolders = ['E:/atil/BreathDataset/Processed/inputsFrom_mini_sample_set/']
# inputFolders = ['E:/atil/BreathDataset/Processed/inputsFrom_mid_sample_set/']
# inputFolders = ['E:/atil/BreathDataset/Processed/inputsFrom_max_sample_set/']

inputFolders = ['D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_max_sample_set/']
# featureModes = ['Freqs', 'Mags']
featureModes = ['Mags']
trainingEpochs = [400]
stepSizes = [4]
batchSizes = [512]
learningRates = [0.001, 0.002]
# lossFunctions = ['CatCrosEnt', 'KLDiv']
lossFunctions = ['CatCrosEnt']
# optimizers = ['Adam', 'Sgd', 'SgdNest', 'Adamax', 'Nadam', 'Rms']
optimizers = ['Adam']

countLimit = 0

f = open('conf.txt', 'w')
listOfParams = list()
for inp in inputFolders:
    for fm in featureModes:
        for te in trainingEpochs:
            for ss in stepSizes:
                for bs in batchSizes:
                    for lr in learningRates:
                        for lf in lossFunctions:
                            for o in optimizers:
                                parameters = dict()
                                parameters['inputFolder'] = inp
                                parameters['featureMode'] = fm
                                parameters['trainingEpoch'] = te
                                parameters['stepSize'] = ss
                                parameters['batchSize'] = bs
                                parameters['learningRate'] = lr
                                parameters['lossFunction'] = lf
                                parameters['optimizer'] = o
                                listOfParams.append(parameters.copy())
if 0 < countLimit:
    for i in range(countLimit):
        print(random.choice(listOfParams), file=f)
else:
    countLimit = len(listOfParams)
    for par in listOfParams:
        print(par, file=f)
print(countLimit, ' random iteration(s) put into configuration file.')
f.close()
