# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:36:24 2016

@author: Ambroise
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy


import theano
import theano.tensor as T

def predict(n_examples = 10):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    params = pickle.load(open('best_model_MLP1_params1.pkl'))
    W1 = params[0]
    b1 = params[1]
    W2 = params[2]
    b2 = params[3]    
    
    
    
    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()


    test_set_x = T.tanh(T.dot(test_set_x, W1) + b1)

    p_y_given_x = T.nnet.softmax(T.dot(test_set_x, W2) + b2)
    # symbolic description of how to compute prediction as class whose
    # probability is maximal
    predicted_values = T.argmax(p_y_given_x, axis=1)
    # end-snippet-1
    
    real_values= test_set_y.eval()
    
    print(('Predicted values for the first %i examples in test set:') % (n_examples))
    print(predicted_values[:n_examples].eval())
    print(('real values for the first %i examples in test set:') % (n_examples))
    print(real_values[:n_examples])
    print('% error for all examples in test set:')
    print(T.mean(T.neq(predicted_values,real_values)).eval()*100)
   # print(T.mean(T.neq(predict_model(test_set_x),test_set_y.eval())).eval()*100)
   # print(T.mean(T.neq(predicted_values,real_values.eval())).eval()*100)

if __name__ == '__main__':
        predict(30)
    
    