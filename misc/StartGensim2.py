# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:37:39 2016

@author: Ambroise
"""
#Import

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

from pprint import pprint #pretty-printer

## Création d'un dictionnary

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
              
## remove common words and tokenize
stoplist = set('for a of the and to in'.split( ))
              
              
texts = [[word for word in document.lower().split( ) if word not in stoplist]
          for document in documents] 
  
  
## remove words that appear only once
  
from collections import defaultdict
frequency = defaultdict(int)  # A quoi sert cette ligne ?

for text in texts:
     for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
          for text in texts]  
#pprint(texts)      

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
print(dictionary.token2id)

##LE DICO EST PRET ET SAVED

## Corpus line by line

class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split()) # Comptage des termes

corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory!

for vector in corpus_memory_friendly: # load one vector into memory at a time
   print(vector)




              
              
              