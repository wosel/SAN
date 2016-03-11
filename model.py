from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, LambdaMerge, Reshape, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from theano import function
import theano.tensor

print(theano.config.device)


import data_loader
from CustomDense import CustomDense
from CustomRepeatVector import CustomRepeatVector
from vggRepresentation import getRepresentation

'''
    Right now it throws a very cryptic optimization error unfortunately. I suspect a problem with the Custom Dense layer
'''


#This is the data loader call
# right now it loads the DAQUAR dataset, but is memory inefficient.
# also a pairing of question <-> image is still missing

[images, trainSet, testSet] = \
    data_loader.load_both(qFolder='C:/daquar',
                          qFullFile='qa.37.raw.txt',
                          qTrainFile='qa.37.raw.train.txt',
                          qTestFile='qa.37.raw.reduced.test.txt',
                          iFolder='C:/daquar/nyu_depth_images',
                          imLimit=100,
                          qLimit=100
                          )



import sys
sys.setrecursionlimit(10000)

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
model.add_output(name='u', inputs=['reshapedVIT', 'lstm'], merge_mode='sum')

model.compile(loss={'u': 'categorical_crossentropy'}, optimizer='sgd')
