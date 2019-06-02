inputFolders = ['D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/']
featureModes = ['FirstFreq', 'FirstMag', 'Freqs', 'Mags', 'All']
trainingEpochs = [5, 50]
learningRates = [0.001, 0.002]
numHiddens = [128, 256]
numLayers = [2, 4]
celltypes = ['lstm', 'gru']
optimizers = ['Adam', 'Grad']

f = open('conf.txt', 'w')
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
                                print(parameters, file=f)
print(iteration, 'iteration(s) put into configuration file.')
f.close()
