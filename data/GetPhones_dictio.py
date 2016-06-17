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
import fnmatch


    

    
folder = "/home/ambroise/Documents/raw_data/raw_data_cleaned"
#    filename = ("s0%ia.words" % (i))
#    newfilename = ("s0%ia_#_real.words" % (i))
#    
#    if os.path.exists(filename):   #does the file exist ?
for file in os.listdir(folder):
            
            os.chdir(folder)
            if fnmatch.fnmatch(file, '*.words'):
                filename = file.split(".")[0]                
                newfilename = filename + "_#_dictio.words"
                
                with cd.open(file, 'r', encoding='utf8') as f:
                    text = f.read().splitlines()
                
                text=text[9:]  # 9 first lines of .word buckeye files are infos
                
                
                text = [line.split(";") for line in text]
                text=np.array(text)[:,1]  # choose btw proper dictionary phones or really prononced phones ([:,1] or [:,2])
                
                phone_corpus=[]
                
                # Remove all non phones content (written in capital letter in buckeye)
                for word in text:
                    if word.islower():
                        "word contain no capital letter"
                        phone_corpus = phone_corpus + [word]
                
                phone_corpus = [line.split() for line in phone_corpus]
                
                phone_corpus2=[]
                
                #"#" signs can separate the words 
                for line in phone_corpus:
                    line.insert(0,"#")
                    phone_corpus2 = phone_corpus2 + line
                
                #Beginning/end of the text : "<"/">"
                phone_corpus = ["<"] + phone_corpus2[1:] + [">"] #1 instead of 0 if "#" signs separates words 
        
                        
                phone_corpus = [phone for phone in phone_corpus if "=" not in phone]    
                
                print(phone_corpus[:10], "   ...saved as...   ",newfilename)
                
                os.chdir("/home/ambroise/Documents/LSC-Internship/data/data_cleaned") 
                with open(newfilename, 'wb') as f:
                    np.savetxt(f,phone_corpus,fmt='%s')
