# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from collections import OrderedDict
import copy


from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.layers.core import Layer
from keras.regularizers import ActivityRegularizer

import marshal
import types
import sys




class CustomRepeatVector(Layer):
    '''
        Repeat input n times.

        Dimensions of input are assumed to be (nb_samples, dim).
        Return tensor of shape (nb_samples, n, dim).
    '''
    def __init__(self, n, **kwargs):
        super(CustomRepeatVector, self).__init__(**kwargs)
        self.n = n

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], self.n)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.repeat(X, self.n).dimshuffle(0, 2, 1)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "n": self.n}
        base_config = super(CustomRepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))