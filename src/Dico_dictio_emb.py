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
import glob

from gensim import corpora, models, similarities

import numpy
from pprint import pprint #pretty-printer

import theano
import theano.tensor as T

import fnmatch

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



            

    
   
def predict(pathname_input, pathname_output, dataset_dico, modelname, activation = T.tanh):
        """
    load a trained model and use it
    """
    #load trained model
        os.chdir(pathname_input)
        dictionary = corpora.Dictionary.load(dataset_dico)
        
        dico = dictionary.token2id    
        dictio=[key for key, value in dico.iteritems()]
        
        
        os.chdir(pathname_output)
        model = pickle.load(open(modelname))
        
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
        
        
        embedding_filename = "Embedding_" + modelname
        embedding_filename = embedding_filename.replace("pkl", "csv")
        print(embedding_filename)
        with open (embedding_filename,'w') as f:
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
    pathname_input="/home/ambroise/Documents/LSC-Internship/data/data_cleaned/Buckeye_#_real"
    pathname_output="/home/ambroise/Documents/LSC-Internship/results/reproductibility"
    models = glob.glob(pathname_output + "/*.pkl")
    print(models)
    for modelname in models:
        modelname = modelname.replace(pathname_output+"/","")
        print("####")        
        print(modelname)
        print("####")  
        predict(pathname_input=pathname_input,
                pathname_output=pathname_output,
                dataset_dico='BuckeyeDictionary_real.dict', 
                modelname = modelname)
        

