from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, LambdaMerge, Reshape, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from theano import function
import theano.tensor


import data_loader
from CustomDense import CustomDense


'''
    Right now it throws a very cryptic optimization error unfortunately. I suspect a problem with the Custom Dense layer
'''

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
'''

#VGG-16 net, starts with 224x224 and ends with the last max-pooling layer
#last max-pooling layer has 49 (7x7) image regions with 512 features each
imageModel = Sequential()
imageModel.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
imageModel.add(Convolution2D(64, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(64, 3, 3, activation='relu'))
imageModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(128, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(128, 3, 3, activation='relu'))
imageModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(256, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(256, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(256, 3, 3, activation='relu'))
imageModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
imageModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
imageModel.add(ZeroPadding2D((1, 1)))
imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
imageModel.add(MaxPooling2D((2, 2), strides=(2, 2)))
#end of VGGnet

# reshape to matrix num_featurex x num_regions (was 512x7x7)
imageModel.add(Reshape(dims=(512, 49)))

# custom layer - weight matrix for a matrix (keras has Dense for 2D only
imageModel.add(CustomDense(output_dim=(512, 512), activation='linear'))

dummyDictSize = 1000
dummyQuestionLength = 100

#language model: embedding, LSTM
languageModel = Sequential()
languageModel.add(Embedding(input_dim=dummyDictSize, output_dim=256, input_length=dummyQuestionLength))
languageModel.add(LSTM(output_dim=512, activation='sigmoid', inner_activation='hard_sigmoid'))
languageModel.add(Dropout(0.5))
# SAN specified W_QA
languageModel.add(Dense(512, activation='linear'))

#to make 'sum' possible in merge
languageModel.add(RepeatVector(512))

#end of language model



attentionModelP = Sequential()
attentionModelP.add(Merge([imageModel, languageModel], mode='sum'))
attentionModelP.add(CustomDense(output_dim=(1,512), activation='softmax'))


attentionModelV = Sequential()
attentionModelV.add(Merge([imageModel, attentionModelP], mode='dot', dot_axes=[(1,), (2,)]))

attentionModelV.compile(loss='categorical_crossentropy', optimizer='sgd')


#this needs figuring out - language model has to somehow "split" the language model
#attentionModelU = Sequential()
#attentionModelU.add(Merge([attentionModelV, languageModel], mode='sum'))

#print("compiling model")
#attentionModelU.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# model.add_input(name='input_text', input_shape=(trainSet.)
