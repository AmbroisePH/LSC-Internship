# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:15:12 2016

@author: ambroise

Goal: get the data ready for phonotactic embedding NN (skipgram, CBOW)

inputs: buckeye files .words

outputs: .txt list of phones. Beginning/end of the text : "<"/">"

option1: - choose btw proper dictionary phones or really prononced phones (_dictio or _real in newfilename)
option2: - "#" signs can separate the words (_# or NONE in newfilename)
       
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import codecs as cd
import numpy as np


os.chdir("/home/ambroise/Documents/Buckeye")    

for i in range (0,5000): 
    
    filename = ("s%ia.words" % (i))
    newfilename = ("s%ia_real.words" % (i))
    
    if os.path.exists(filename):   #does the file exist ?
        
        with cd.open(filename, 'r', encoding='utf8') as f:
            text = f.read().splitlines()
        
        text=text[9:]  # 9 first lines of .word buckeye files are infos
        
        
        text = [line.split(";") for line in text]
        text=np.array(text)[:,2]  # choose btw proper dictionary phones or really prononced phones ([:,1] or [:,2])
        
        phone_corpus=[]
        
        # Remove all non phones content (written in capital letter in buckeye)
        for word in text:
            if word.islower():
                "word contain a capital letter"
                phone_corpus = phone_corpus + [word]
        
        phone_corpus = [line.split() for line in phone_corpus]
        
        phone_corpus2=[]
        
        #"#" signs can separate the words 
        for line in phone_corpus:
            #line.insert(0,"#")
            phone_corpus2 = phone_corpus2 + line
            
        phone_corpus = ["<"] + phone_corpus2[0:] + [">"] #Beginning/end of the text : "<"/">"
            
        
        print(phone_corpus[:10], "   ...saved as...   ",newfilename)
        
        
        with open(newfilename, 'wb') as f:
            np.savetxt(f,phone_corpus,fmt='%s')
    
            
     