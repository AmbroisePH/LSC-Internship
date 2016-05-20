# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:20:36 2016

@author: Ambroise
"""

'''TUTO : http://deeplearning.net/tutorial/logreg.html#logreg'''

##Imports

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

# initialize with 0 the weights W as a matrix of shape (n_in, n_out)

n_in=10
n_out=8

self.W = theano.shared(
    value=numpy.zeros(
(n_in, n_out),
dtype=theano.config.floatX
    ),
    name='W',
    borrow=True
)
# initialize the biases b as a vector of n_out 0s
self.b = theano.shared(
    value=numpy.zeros(
(n_out,),
dtype=theano.config.floatX
    ),
    name='b',
    borrow=True
)

# symbolic expression for computing the matrix of class-membership
# probabilities
# Where:
# W is a matrix where column-k represent the separation hyperplane for
# class-k
# x is a matrix where row-j  represents input training sample-j
# b is a vector where element-k represent the free parameter of
# hyperplane-k
self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

# symbolic description of how to compute prediction as class whose
# probability is maximal
self.y_pred = T.argmax(self.p_y_given_x, axis=1)

