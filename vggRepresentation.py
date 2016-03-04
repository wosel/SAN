from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, LambdaMerge, Reshape, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np


def getRepresentation(imageDataset):


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

    # reshape to matrix num_features x num_regions (was 512x7x7)

    VGGWeightsPath='C:/DP/vgg/vgg16_weights.h5'

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
    print("compiling model")

    imageModel.add(Reshape(dims=(512, 49)))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    imageModel.compile(optimizer=sgd, loss='categorical_crossentropy')

    print("model compiled")
    print("calculating image representation")

    imageRep = imageModel.predict(imageDataset.dataset, verbose=True)
    imageDataset.regionRep = imageRep
    return imageRep





