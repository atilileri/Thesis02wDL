import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(len(classes)//2, len(classes)//2))
    # plt.figure(figsize=(len(classes), len(classes)))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        fmt = '.2f'
    else:
        fmt = 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    titleToPrint = str.replace(title, ' ', '')
    plt.savefig("./plots/confMat/cm_" + titleToPrint + '_' + scriptStartTime + ".png")
    plt.savefig("./plots/confMat/cm_" + titleToPrint + '_' + scriptStartTime + ".svg")
    plt.show()


# Compute confusion matrix
scriptStartTime = datetime.now().strftime('%Y%m%d_%H%M%S')
np.set_printoptions(precision=2)
labelsPos3 = ['Sitting', 'Standing', 'Lying']
valsPos3 = np.array([[382, 5, 0], [11, 388, 0], [2, 0, 226]])
labelsPos5 = ['Sitting', 'Low-Sitting', 'Standing', 'Hands-Behind', 'Lying']
valsPos5 = np.array([[157, 28, 1, 0, 1], [22, 172, 6, 0, 0], [4, 4, 160, 31, 0], [2, 0, 31, 167, 0], [1, 0, 0, 0, 227]])
labelsSpkr = ['aa', 'ab', 'am', 'by', 'ce', 'ck', 'ds', 'eb', 'eg', 'ek',
              'eo', 'ib', 'ig', 'kk', 'mb', 'my', 'sd', 'sg', 'sk', 'yd']
valsSpkr = np.array([[27, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0],
[0, 17, 0, 0, 0, 0, 0, 5, 1, 0, 1, 1, 0, 2, 1, 1, 1, 0, 0, 1],
[0, 0, 31, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 23, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 3, 19, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 18, 4, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0],
[0, 3, 0, 0, 1, 1, 0, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[3, 0, 0, 1, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 3, 0, 0, 0, 0, 0, 0, 35, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 30, 3, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 44, 1, 0, 0, 0, 0, 3, 0, 2],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 137, 1, 0, 0, 1, 0, 0, 2],
[0, 1, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 73, 1, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 3, 0],
[0, 1, 0, 1, 0, 0, 1, 2, 2, 0, 0, 2, 1, 3, 1, 0, 68, 1, 0, 3],
[0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 44, 0, 3],
[0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 29, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 57]])

plot_confusion_matrix(valsPos3, classes=labelsPos3, normalize=True, title='3-Posture Classification')
plot_confusion_matrix(valsPos5, classes=labelsPos5, normalize=True, title='5-Posture Classification')
plot_confusion_matrix(valsSpkr, classes=labelsSpkr, normalize=True, title='Speaker Classification')
