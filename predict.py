from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, LambdaMerge, Reshape, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from theano import function
import theano.tensor

print(theano.config.device)

from configparser import ConfigParser

import data_loader
from CustomDense import CustomDense
from CustomRepeatVector import CustomRepeatVector
from vggRepresentation import getRepresentation

import numpy as np;

import sys
sys.setrecursionlimit(10000)

#This is the data loader call
# right now it loads the DAQUAR dataset, but is memory inefficient.
# also a pairing of question <-> image is still missing

if (len(sys.argv) < 2):
    sys.exit("specify ini file of dataset as first argument")

datasetIniFile = sys.argv[1]
parser = ConfigParser()
parser.readfp(open(datasetIniFile))

qFolder = parser.get('dataset', 'qFolder')
qFullFile = parser.get('dataset', 'qFullFile')
qTrainFile = parser.get('dataset', 'qTrainFile')
qTestFile = parser.get('dataset', 'qTestFile')
iFolder = parser.get('dataset', 'iFolder')
iMatrix = parser.get('dataset', 'iMatrix')
print(qFolder)


[images, trainSet, testSet, testRevDict] = \
    data_loader.load_both(
                          qFolder=qFolder,
                          qFullFile=qFullFile,
                          qTrainFile=qTrainFile,
                          qTestFile=qTestFile,
                          iFolder=iFolder,
                          imLimit=-1,
                          qLimit=-1
                          )
'''
#print(trainSet.qMatrix[3, :])
#print(trainSet.iMatrix[0, :, :, :])
#print(trainSet.iMatrix[127	, :, :, :])

print("getting vgg repres")
#trainSet.vggIMatrix = getRepresentation(trainSet.iMatrix)
#trainSet.vggIMatrix = np.load(iMatrix)

testSet.vggIMatrix = np.load('testVggIMatrix.npy')
#np.save('testVggIMatrix.npy',testSet.vggIMatrix )
#sys.exit("breakpoint")


#m
numRegions = 49

#d
imageRepDimension = 512

#k = d (how else would you add  u = v_i~ + v_q ???
LSTMDimension = imageRepDimension

dictSize = trainSet.dictSize
queryLen = trainSet.qLength



model = Graph()
#LSTM

model.add_input(name='langInput', input_shape=(queryLen,), dtype=int)
model.add_node(Embedding(dictSize, LSTMDimension, input_length=queryLen), name='embed', input='langInput')
model.add_node(LSTM(output_dim=LSTMDimension, activation='sigmoid', inner_activation='hard_sigmoid'), name='lstm', input='embed')
#split output of LSTM after this dropout
model.add_node(Dropout(0.5), name='dropout', input='lstm')

#dense is w_qa from paper. custom repeat vector is repeat & transpose to facilitate matrix-vector addition
model.add_node(Dense(LSTMDimension), name='dense', input='dropout')
model.add_node(CustomRepeatVector(numRegions), name='repeatedLangOutput', input='dense')


#image input. Custom dense is w_ia from paper
model.add_input(name='imInput', input_shape=(imageRepDimension, numRegions), dtype='float')
model.add_node(CustomDense(output_dim=(LSTMDimension, numRegions)), name='imDense', input='imInput')


#combination of image and lstm in one layer of attention network
#  hA, pI, vI~ and u correspond to paper. CustomDense is w_p from paper (bias unit missing atm)
model.add_node(Activation('tanh'), inputs=['imDense', 'repeatedLangOutput'], merge_mode='sum', name='hA')
model.add_node(CustomDense(output_dim=(1, numRegions)), input='hA', name='preActivationPI')
model.add_node(Reshape(dims=(numRegions,)), input='preActivationPI', name='reshapedPAPI')
model.add_node(Activation('softmax'), input='reshapedPAPI', name='pI')
model.add_node(Reshape(dims=(numRegions, 1)), input='pI', name='reshapedPI')

model.add_node(Activation('linear'), inputs=['imInput', 'reshapedPI'], merge_mode='dot', dot_axes=([2], [1]), name='vITilde')
model.add_node(Reshape(dims=(imageRepDimension,)), input='vITilde', name='reshapedVIT')
model.add_node(Activation('linear'), name='u', inputs=['reshapedVIT', 'lstm'], merge_mode='sum')
model.add_node(Dense(dictSize), name='Wu', input='u')
model.add_node(Activation('softmax'), name='pans', input='Wu')
model.add_output(name='output', input='pans')

print("compiling full model")
model.load_weights('test_weights.h5')

model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='sgd')




Y = model.predict({'imInput': testSet.vggIMatrix, 'langInput':testSet.qMatrix})

np.save('testResult.npy', Y['output'])
'''
Youtput = np.load('testResult.npy')
for i in range(10):

    print(Youtput[i, :])
    print(testSet.questionList[i].question)
    print(testSet.questionList[i].answer)
    print(testSet.questionList[i].imageID)
    print(testRevDict[np.argmax(Youtput[i])])
    print(testRevDict[np.argmax(Youtput[i]) - 1])
    print(testRevDict[np.argmax(Youtput[i]) + 1])
    print("")
    print("")