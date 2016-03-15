# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from collections import OrderedDict
import copy

'''
    I do not wish to figure out how to do theano tensordot with tensorflow
    so theano only atm.
'''

from theano import tensor

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.layers.core import Layer
from keras.regularizers import ActivityRegularizer



import marshal
import types
import sys


class CustomDense(Layer):
    '''
        Fully connected layer, accepts 2Dmatrix shaped inputs, no bias
    '''
    input_ndim = 3

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        #self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        #self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=3)
        super(CustomDense, self).__init__(**kwargs)

    def build(self):
        input_dim = (self.input_shape[1], self.input_shape[2])
        self.W = self.init((self.output_dim[0], input_dim[0]))

        #self.b = K.zeros((self.output_dim,))
        self.trainable_weights = [self.W]

        self.params = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        #if self.b_regularizer:
        #    self.b_regularizer.set_param(self.b)
        #    self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim[0], self.output_dim[1])

    def get_output(self, train=False):
        X = self.get_input(train)

        output = self.activation(K.dot(X.dimshuffle(0, 2, 1), self.W.dimshuffle(1, 0)).dimshuffle(0, 2, 1))
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  #"b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  #"b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim}
        base_config = super(CustomDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

