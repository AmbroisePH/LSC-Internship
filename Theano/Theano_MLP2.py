# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:34:07 2016

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
    entree = pickle.load(open('best_model_MLP1.pkl'))
    LogRegressionLayer = pickle.load(open('best_model_MLP1_LogRegressionLayer.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[entree],
        outputs=LogRegressionLayer.y_pred,
        on_unused_input='warn')
    


    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    #y = theano.shared(numpy.asarray(test_set_y,dtype='float32')
    
    predicted_values = predict_model(test_set_x[:n_examples])
    real_values= test_set_y[:n_examples].eval()
    print('Predicted values for the first 10 examples in test set:' % (n_examples))
    print(predicted_values)
    print('real values for the first 20 examples in test set:')
    print(real_values)
    #print(T.neq(predicted_values,real_values))
    
if __name__ == '__main__':
        predict(30)
        