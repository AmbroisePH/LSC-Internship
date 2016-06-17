# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:33:36 2016

@author: ambroise
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import codecs as cd

from gensim import corpora, models, similarities

import numpy
from pprint import pprint #pretty-printer

import theano
import theano.tensor as T

import fnmatch

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



            
def load_data(dataset):
    
#load buckeye dictionary 
    os.chdir("/home/ambroise/Documents/LSC-Internship/data")
    if fnmatch.fnmatchcase(dataset, '*real*'):
        dictio = corpora.Dictionary.load('BuckeyeDictionary_real.dict')
    elif fnmatch.fnmatchcase(dataset, '*dictio*'):
        dictio = corpora.Dictionary.load('BuckeyeDictionary_dictio.dict')
    else:
        raise TypeError('Filename does not contain real or dictio, load data cannot find its dictionary',(dataset))

    
    dico = dictio.token2id    
    print(dico)
    n_in = len(dictio)
    
    with cd.open(dataset, 'r', encoding='utf8') as f:
        text = f.read().lower().split()
  #      text=text[0]
        
    n_examples=len(text)
    print("n_examples = ", n_examples)
    
    train_set_x = numpy.zeros((n_examples,n_in),dtype='int')
    
    
    for i in range (1,n_examples-1):
        train_set_x[i,dico[text[i-1]]] =  1
        train_set_x[i,dico[text[i+1]]] = 1
   
    print(train_set_x)
    
    #for i in range (0,n_examples):
    #    print(sum(train_set_x[i,]))
    
        
    
    train_set_y = numpy.zeros((n_examples,),dtype='int')
    for i in range (0,n_examples):
        train_set_y[i] = dico[text[i]]   
    
#    train_set_y[n_examples-1]=numpy.nonzero(train_set_x[0])[0]
#    for i in range (0,n_examples-2):
#        train_set_y[i]=numpy.nonzero(train_set_x[i+1])[0]
#            
    print(train_set_y)

    
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    arr = (train_set_x[:3999],train_set_y[:3999])
    train_set = tuple(map(tuple, arr))

    arr = (train_set_x[4000:4999],train_set_y[4000:4999])
    valid_set = tuple(map(tuple, arr))

    arr = (train_set_x[5000:],train_set_y[5000:])
    test_set = tuple(map(tuple, arr))    

    def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables
    
            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')
    
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    print(train_set_x.eval().shape)
    print(train_set_y.eval().shape)
    print(valid_set_x.eval().shape)
    print(valid_set_y.eval().shape)
    print(test_set_x.eval().shape)
    print(test_set_y.eval().shape)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
              (test_set_x, test_set_y)]
    return rval
    
    
   
def predict(dataset='s3802a_dictio.words', activation = T.tanh):
        """
        load a trained model and use it
        """
        #load trained model
        model = pickle.load(open('best_model_PhonesCBOW2.pkl'))
    
        W1=model[0]
        b1=model[1]
        W2=model[2]
        b2=model[3]
        
        #load dataset 
        
        datasets = load_data(dataset)
        test_set_x, test_set_y = datasets[2]
    
        
    #
        lin_output = T.dot(test_set_x, W1) + b1
        HL_output = activation(lin_output)
        p_y_given_x = T.nnet.softmax(T.dot(HL_output, W2) + b2)
        y_pred = T.argmax(p_y_given_x, axis=1)

        if test_set_y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('test_set_y', test_set_y.type, 'y_pred', y_pred.type)
            )        
        errors = T.mean(T.neq(y_pred, test_set_y))       
        
        print(errors.eval())
        # load the saved model
       
        
  # #
if __name__ == '__main__':
    os.chdir("/home/ambroise/Documents/LSC-Internship/results")
    predict(dataset='s3802b_dictio.words')
        
