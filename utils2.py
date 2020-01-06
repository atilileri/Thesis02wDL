import os
import numpy as np
from datetime import datetime

import nbformat as nbf
import pandas as pd

# print options
np.set_printoptions(edgeitems=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)

# global init
fOutTxt = None
scriptStartDateTime = datetime.now().strftime('%Y%m%d_%H%M%S')
results = None


# Prints to both a file and console
def myPrint(*args, mode='both', **kwargs):
    global fOutTxt
    global scriptStartDateTime
    global results

    if (fOutTxt is None) and (mode in ['both', 'file']):
        outFileName = './outputs/out_' + scriptStartDateTime + '.txt'
        if not os.path.exists(os.path.dirname(outFileName)):
            os.makedirs(os.path.dirname(outFileName))
        fOutTxt = open(outFileName, 'w+')

    if mode in ['both', 'file']:
        print(*args, file=fOutTxt, **kwargs)
    if mode in ['both', 'console']:
        print(*args, **kwargs)
    if 'code' == mode:
        results = args[0]


def endPrint():
    if fOutTxt is not None:
        fOutTxt.flush()
        fOutTxt.seek(0)
        txt = list()
        txt.append('"""\n')
        txt.extend(fOutTxt.readlines())
        txt.append('"""\n')
        txt.append('print("")\n')
        fOutTxt.close()

        outFileName = './outputs/notebooks/out_' + scriptStartDateTime + '.ipynb'
        if not os.path.exists(os.path.dirname(outFileName)):
            os.makedirs(os.path.dirname(outFileName))
        fOutNtb = open(outFileName, 'w')

        nb = nbf.v4.new_notebook()
        txt = ''.join(txt)
        cell1 = nbf.v4.new_code_cell(str(txt))
        cell2 = nbf.v4.new_code_cell('r = ' + str(results) + '''\n%matplotlib inline
from matplotlib import pyplot as plt
# ACCURACIES
plt.figure(figsize=(10, 10))
plt.title('Accuracies')
plt.xlabel('Epoch(s)')
plt.ylabel('Accuracy')
plt.plot(r['acc'], label='Train Acc')
plt.plot(r['val_acc'], label='Test Acc')
plt.grid(linestyle='dashed', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# LOSSES
plt.figure(figsize=(10, 10))
plt.title('Losses')
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(r['loss'], label='Train Loss')
plt.plot(r['val_loss'], label='Test Loss')
plt.grid(linestyle='dashed', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()''')
        if results is not None:
            nb['cells'] = [cell1, cell2]
        else:
            nb['cells'] = [cell1]
        nbf.write(nb, fOutNtb)
        fOutNtb.close()


def printModelConfig(modelConfiguration):
    myPrint('Model Layer Parameters:')
    for layer in modelConfiguration['layers']:
        if 'Conv1D' == layer['class_name']:
            myPrint('Name: %s, Filters: %s, Kernel Size: %s, Strides: %s, Activation: %s.'
                    % (layer['config']['name'],
                       str(layer['config']['filters']),
                       str(layer['config']['kernel_size']),
                       str(layer['config']['strides']),
                       layer['config']['activation']))
        elif 'Dropout' == layer['class_name']:
            myPrint('Name: %s, Rate: %s.' % (layer['config']['name'], str(layer['config']['rate'])))
        elif 'LSTM' == layer['class_name']:
            myPrint('Name: %s, Units: %s, Activation: %s.' % (layer['config']['name'],
                                                              str(layer['config']['units']),
                                                              str(layer['config']['activation'])))
        else:
            myPrint('Name: %s' % (layer['config']['name']))


