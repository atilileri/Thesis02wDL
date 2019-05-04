import pickle


# Loads data from file into variable
def loadData(path):
    return pickle.load(open(path, "rb"))


filepath = 'D:/atili/MMIExt/Audacity/METU Recordings/Dataset/inputsFrom_mini_sample_set/ba01_17_45.451-0.507.inp'

inputFile = loadData(filepath)
inputFile2 = loadData(filepath+'2')

print(inputFile2.shape)
