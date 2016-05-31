w# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:15:05 2016

@author: Ambroise
"""

#Import

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

from pprint import pprint #pretty-printer



from pprint import pprint #pretty-printer


documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
              
# remove common words and tokenize
stoplist = set('for a of the and to in'.split( ))
              
              
texts = [[word for word in document.lower().split( ) if word not in stoplist]
          for document in documents] 
  
pprint(texts)      



sentences = texts
# train word2vec on the two sentences
#model = models.Word2Vec(sentences, min_count=1, size=100) #default size = 100
model = models.Word2Vec(sentences, min_count=1, size=200)


fname = 'fmame'
model.save(fname)  #Persist a model to disk 
model = models.Word2Vec.load(fname) #continue training with the loaded model

print(model.similarity('trees', 'well'))
print(model.similarity('graph', 'graph'))
print(model.similarity('well', 'trees'))

print(model.doesnt_match("system system trees system ".split()))
print(model.doesnt_match("system graph machine trees  ".split()))

print(model.most_similar(positive=['human', 'system'], negative=['machine']))





