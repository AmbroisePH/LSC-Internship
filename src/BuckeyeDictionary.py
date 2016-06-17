# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:27:41 2016

@author: ambroise

GOAL : Build an universal dictionary for all Buckeye corpora

Input : Concatenaton of all buckeye corpora
Output : Universal dictionary
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'

#import six.moves.cPickle as pickle
#import gzip
import os
#import sys
#import timeit
import codecs as cd
#
from gensim import corpora, models, similarities
#
import numpy
#from pprint import pprint #pretty-printer
#
#import theano
#import theano.tensor as T

import fnmatch

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def built_dictionary(folder = '/home/ambroise/Documents/LSC-Internship/data/data_cleaned'):
        
        
        totalcorpus=[]
        
        os.chdir("/home/ambroise/Documents/LSC-Internship/data/data_cleaned")
        
        for file in os.listdir(folder):
            if fnmatch.fnmatch(file, '*dictio.words'):

                totalcorpus = totalcorpus + [line.lower().split() for line in cd.open(file, encoding='utf8')]    
        
        
        dictionary = corpora.Dictionary(totalcorpus)
        print(dictionary.token2id)
        print("Dictionary size :", len(dictionary), 'phones')
        dictionary.save('/home/ambroise/Documents/LSC-Internship/data/data_cleaned/BuckeyeDictionary_dictio.dict')
        
#        test = dictionary.load('/home/ambroise/Documents/LSC-Internship/results/BucheyeDictionary_real.dict')
#        print(test.token2id)    




if __name__ == '__main__':
    built_dictionary()
        
