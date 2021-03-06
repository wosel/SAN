from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, LambdaMerge, Reshape, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from theano import function
import theano.tensor

print(theano.config.device)

from configparser import ConfigParser

import dataLoaderShortAnswer
from CustomDense import CustomDense
from CustomRepeatVector import CustomRepeatVector
from vggRepresentation import getRepresentation

import numpy as np;

import sys
sys.setrecursionlimit(10000)

#This is the data loader call
# right now it loads the DAQUAR dataset, but is memory inefficient.
# also a pairing of question <-> image is still missing

if (len(sys.argv) < 4):
    sys.exit("usage: preditct.py [ini file] [trained weight file] [network output saveFile]")

datasetIniFile = sys.argv[1]
parser = ConfigParser()
parser.read_file(open(datasetIniFile))

modelWeightsFile = sys.argv[2]
outputFile = sys.argv[3]


qFolder = parser.get('dataset', 'qFolder')
qFullFile = parser.get('dataset', 'qFullFile')
qTrainFile = parser.get('dataset', 'qTrainFile')
qTestFile = parser.get('dataset', 'qTestFile')
iFolder = parser.get('dataset', 'iFolder')
iTestMatrix = parser.get('dataset', 'iTestMatrix')



[images, trainSet, testSet, testRevDict] = \
    dataLoaderShortAnswer.load_both(
                          qFolder=qFolder,
                          qFullFile=qFullFile,
                          qTrainFile=qTrainFile,
                          qTestFile=qTestFile,
                          iFolder=iFolder,
                          imLimit=-1,
                          qLimit=-1
                          )

#print(trainSet.qMatrix[3, :])
#print(trainSet.iMatrix[1, :, :, :])
#print(trainSet.iMatrix[127	, :, :, :])

print("getting vgg repres")
#trainSet.vggIMatrix = getRepresentation(trainSet.iMatrix)

testSet.vggIMatrix = np.load(iTestMatrix)

#sys.exit("breakpoint")


#m
numRegions = 196

#d
imageRepDimension = 512

#k = d (how else would you add  u = v_i~ + v_q ???
LSTMDimension = imageRepDimension

questionDictSize = trainSet.questionDictSize
answerDictSize = trainSet.answerDictSize
queryLen = trainSet.qLength


model = Graph()
#LSTM

model.add_input(name='langInput', input_shape=(queryLen,), dtype=int)
model.add_node(Embedding(questionDictSize, LSTMDimension, input_length=queryLen), name='embed', input='langInput')
model.add_node(LSTM(output_dim=LSTMDimension, activation='sigmoid', inner_activation='hard_sigmoid'), name='lstm', input='embed')
#split output of LSTM after this dropout
model.add_node(Dropout(0.5), name='lstmD', input='lstm')

#dense is w_qa from paper. custom repeat vector is repeat & transpose to facilitate matrix-vector addition
model.add_node(Dense(LSTMDimension), name='dense', input='lstmD')
model.add_node(CustomRepeatVector(numRegions), name='repeatedLangOutput', input='dense')


#image input. Custom dense is w_ia from paper

model.add_input(name='imInput', input_shape=(imageRepDimension, numRegions), dtype='float')
model.add_node(CustomDense(output_dim=(LSTMDimension, numRegions)), name='imDense', input='imInput')
model.add_node(Dropout(0.3), input='imDense', name='imDenseD')

#combination of image and lstm in one layer of attention network
#  hA, pI, vI~ and u correspond to paper. CustomDense is w_p from paper (bias unit missing atm)

model.add_node(Activation('relu'), inputs=['imDenseD', 'repeatedLangOutput'], merge_mode='sum', name='hA')
#LANG ONLY MODEL
#model.add_node(Activation('tanh'), input='repeatedLangOutput', name='hA')
model.add_node(CustomDense(output_dim=(1, numRegions), bias=True), input='hA', name='preActivationPI')
model.add_node(Dropout(0.3), input='preActivationPI', name='preActivationPID')
model.add_node(Reshape(dims=(numRegions,)), input='preActivationPID', name='reshapedPAPI')
model.add_node(Activation('softmax'), input='reshapedPAPI', name='pI')
model.add_node(Reshape(dims=(numRegions, 1)), input='pI', name='reshapedPI')

model.add_node(Activation('linear'), inputs=['imInput', 'reshapedPI'], merge_mode='dot', dot_axes=([2], [1]), name='vITilde')
model.add_node(Reshape(dims=(imageRepDimension,)), input='vITilde', name='reshapedVIT')
model.add_node(Activation('linear'), name='u', inputs=['reshapedVIT', 'lstmD'], merge_mode='sum')
#LANG ONLY MODEL
#model.add_node(Activation('linear'), name='u', input='lstm')
model.add_node(Dense(answerDictSize), name='Wu', input='u')
model.add_node(Dropout(0.3), input='Wu', name='WuD')
model.add_node(Activation('softmax'), name='pans', input='WuD')
model.add_output(name='output', input='pans')


print("compiling full model")

model.load_weights(modelWeightsFile)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.95, nesterov=True, clipnorm=3.0)
model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=sgd)

Y = model.predict({'imInput': testSet.vggIMatrix, 'langInput':testSet.qMatrix})

np.save(outputFile, Y['output'])


'''
print("fit started")

print(trainSet.vggIMatrix.shape)
print(trainSet.qMatrix.shape)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25)


hist = model.fit({'imInput': trainSet.vggIMatrix, 'langInput': trainSet.qMatrix, 'output': trainSet.aMatrix}, nb_epoch=360, show_accuracy=True, verbose=1, validation_split=0.1, callbacks=[early_stopping])
#LANG ONLY MODEL
#hist = model.fit({'langInput': trainSet.qMatrix, 'output': trainSet.aMatrix}, nb_epoch=360, show_accuracy=True, verbose=1, validation_split=0.1, callbacks=[early_stopping])

print (hist.history)
#json_string = model.to_json()
#open('model.json', 'w').write(json_string)
model.save_weights(outputFile, overwrite=True)
'''



