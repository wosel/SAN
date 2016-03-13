from configparser import ConfigParser

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, LambdaMerge, Reshape, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np

import data_loader


def getModel(VGGWeightsPath):

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
    print(imageModel.layers[-1].output_shape)
    imageModel.add(ZeroPadding2D((1, 1)))
    imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
    print(imageModel.layers[-1].output_shape)
    imageModel.add(ZeroPadding2D((1, 1)))
    imageModel.add(Convolution2D(512, 3, 3, activation='relu'))
    print(imageModel.layers[-1].output_shape)
    #imageModel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #print(imageModel.layers[-1].output_shape)




    print("loading weights")


    import h5py
    f = h5py.File(VGGWeightsPath)
    for k in range(f.attrs['nb_layers']):
        if (k >= len(imageModel.layers)):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        imageModel.layers[k].set_weights(weights)
    f.close()

    print("weights loaded")
    print("compiling image model")
    # reshape to matrix num_features x num_regions (was 512x7x7)

    imageModel.add(Reshape(dims=(512, 196)))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    imageModel.compile(optimizer=sgd, loss='categorical_crossentropy')


    print("image model compiled")
    return imageModel

def getRepresentation(imageMatrix, imageModel):

    print("calculating image representation")
    imageRep = imageModel.predict(imageMatrix, verbose=True, batch_size=32)
    return imageRep


if __name__ == '__main__' :
    import sys
    if (len(sys.argv) < 2):
        sys.exit("specify ini file of dataset as first argument")
    datasetIniFile = sys.argv[1]
    parser = ConfigParser()
    parser.read_file(open(datasetIniFile))

    qFolder = parser.get('dataset', 'qFolder')
    qFullFile = parser.get('dataset', 'qFullFile')
    qTrainFile = parser.get('dataset', 'qTrainFile')
    qTestFile = parser.get('dataset', 'qTestFile')
    iFolder = parser.get('dataset', 'iFolder')
    iTrainMatrixLoc = parser.get('dataset', 'iTrainMatrix')
    iTestMatrixLoc = parser.get('dataset', 'iTestMatrix')
    VGGWeightsPath = parser.get('dataset', 'weights')

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

    imageModel = getModel(VGGWeightsPath)
    print("getting vgg representation of train set")
    iTrainMatrix = getRepresentation(trainSet.iMatrix, imageModel)
    np.save(iTrainMatrixLoc, iTrainMatrix)
    print("getting vgg representation of test set")
    iTestMatrix = getRepresentation(testSet.iMatrix, imageModel)
    np.save(iTestMatrixLoc, iTestMatrix)




