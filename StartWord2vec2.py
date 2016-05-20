# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:26:21 2016

@author: Ambroise
"""

#Import

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

from pprint import pprint #pretty-printer



from pprint import pprint #pretty-printer


#class MySentences(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
# 
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname)):
#                 yield line.split()
#
#sentences = MySentences('/some/directory') # a memory-friendly iterator
#model = gensim.models.Word2Vec(sentences)

class MySentences(object):

 
     def __iter__(self):
 
             for line in open('corpus1.txt'):
                 yield line.lower().split()

sentences = MySentences() # a memory-friendly iterator

for vec in sentences:
    print(vec)

model = models.Word2Vec(sentences, min_count=4, size=200)

fname = 'ModelCorpus1'
model.save(fname)  #Persist a model to disk 
model = models.Word2Vec.load(fname) #continue training with the loaded model

#print(model.similarity('trees', 'well'))
#print(model.similarity('graph', 'graph'))
#print(model.similarity('well', 'trees'))
#
#print(model.doesnt_match("system system trees system ".split()))
#print(model.doesnt_match("system graph machine trees  ".split()))








