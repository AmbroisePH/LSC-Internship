# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:39:24 2016

@author: ambroise

representation of the dico in the embedding space 
input : dico + saved model
output : csv file "phnem":"embed. representation 30 rows vector"

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



            

    
   
def predict(dataset_dico='BuckeyeDictionary_real.dict', activation = T.tanh):
        """
    load a trained model and use it
    """
    #load trained model
        os.chdir("/home/ambroise/Documents/LSC-Internship/data/data_cleaned")
        dictionary = corpora.Dictionary.load(dataset_dico)
        
        dico = dictionary.token2id    
        dictio=[key for key, value in dico.iteritems()]
        
        
        os.chdir("/home/ambroise/Documents/LSC-Internship/results/Phone_Emb4/Real")
        model = pickle.load(open('BestModelEmb4_0.09_3000_75.pkl'))
        
        W1=model[0]
        b1=model[1]
        W2=model[2]
        b2=model[3]
        
        #prepare dataset
        dico_set = numpy.zeros((len(dico),len(dico)),dtype='int')
        for i in range (0, len(dico)):
            
            dico_set[i,dico[dictio[i]]] =  1

        
    #
        dico_rep = T.dot(dico_set, W1) + b1
        #HL_output = activation(lin_output)
        print(dico_rep.eval())
        
        with open ('Dico_real_emb.csv','w') as f:
            lines=[]            
            for i in range (0, len(dico)):
                coord = dico_rep[i].eval().tolist()
                coord=map(str,coord)
                phone = [dictio[i]]
                line = ",".join(phone + coord)
                lines.append(line)
            lines = "\n".join(lines)
            f.write(lines)
                
        
        

        # load the saved model
       
        
  # #
if __name__ == '__main__':
    os.chdir("/home/ambroise/Documents/LSC-Internship/data/data_cleaned")
    predict()
        

