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

if (len(sys.argv) < 3):
    sys.exit("usage: preditct.py [ini file] [network output saveFile]")

datasetIniFile = sys.argv[1]
parser = ConfigParser()
parser.read_file(open(datasetIniFile))

modelOutputFile = sys.argv[2]


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



Y = np.load(modelOutputFile)

answers = np.sum(testSet.aMatrix, axis=0)


#for a in range(len(answers)):
#    if (a<0.1):
#        print (testRevDict[a] + ": " + str(answers([a]))

for i in range(10):

    print(trainSet.questionList[i].question)
    print(trainSet.questionList[i].answer)
    print np.argmax(trainSet.aMatrix[i, :])
    print testRevDict[np.argmax(trainSet.aMatrix[i, :])]
    print (np.argmax(Y[i, :]))
    print testRevDict[np.argmax(Y[i, :])]
    print(Y[i, np.argmax(trainSet.aMatrix[i, :])])
    print(Y[i, np.argmax(Y[i, :])])

    print ("")
    print ("")

outputA = np.zeros(Y.shape)
for i in range(Y.shape[0]):
	ansIdx = np.argmax(Y[i, :])
	for j in range(Y.shape[1]):
		if (j==ansIdx):
			outputA[i, j] = 1.
		else:
			outputA[i, j] = 0.

outputS = np.sum(outputA, axis=0)
print (outputS[45])


print (answers)

print (answers[45])






