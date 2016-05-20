# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:57:42 2016

@author: Ambroise
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

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

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#######
#Load data, build dictionnary
#######

dictionary = corpora.Dictionary(line.lower().split() for line in cd.open('corpus1.txt', encoding='utf8'))
#dictionary = str(dictionary)

n_out = len(dictionary)
n_in = n_out

print(dictionary)
print(dictionary.token2id)
dico = dictionary.token2id

####### get the inputs ready


with cd.open('corpus1.txt', 'r', encoding='utf8') as f:
    text = [f.read().lower().split()]
inp =  print(text)
text=text[0]

n_examples=len(text)

train_set_x = numpy.zeros((n_examples,n_in),dtype='int')
print(train_set_x)


for i in range (0,n_examples-1):
    train_set_x[i,dico[text[i]]] = 1
    
print(train_set_x)
#    
#print(train_set_x)    
    
#stoplist = set('for a of the and to in'.split( ))
#              
#              
#texts = [word for word in document.lower().split( ) if word not in stoplist] 
#  
#  
### remove words that appear only once
#  
#from collections import defaultdict
#frequency = defaultdict(int)  # A quoi sert cette ligne ?
#
#for text in texts:
#     for token in text:
#         frequency[token] += 1
#
#texts = [[token for token in text if frequency[token] > 1]
#          for text in texts]  
##pprint(texts)      
#
#dictionary = corpora.Dictionary(texts)