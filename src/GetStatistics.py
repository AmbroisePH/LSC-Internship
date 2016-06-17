# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:34:17 2016

@author: ambroise

Get statistics phones/context

Input : corpus or corpora
Output : number of combinaisons phone1-phone2-phone3
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os

import codecs as cd

import csv

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    
def load_data(datasets):
    
  
    text=[]
    for dataset in datasets:
        with cd.open(dataset, 'r', encoding='utf8') as f:
            text = text + f.read().lower().split()
      #      text=text[0]
    
    return text

def Get_Statistics(file_list):
    
    dataset=load_data(file_list)
    
    #print(dataset)
    
    dico={}
    
    for i in range(0, len(dataset)-2):
        if (dataset[i],dataset[i+2],dataset[i+1]) not in dico:
            dico[(dataset[i],dataset[i+2],dataset[i+1])] = 1
        else:
            dico[(dataset[i],dataset[i+2],dataset[i+1])] += 1
            
    print(dico)
    print(len(dico))

    
    writer = csv.writer(open('GetStat.csv', 'wb'))
    for key, value in dico.items():
        writer.writerow([(key[0],key[1]), key[2], value])
    
    
    
    

if __name__ == '__main__':
    file_list = ['s0101a_dictio.words','s0101b_dictio.words','s0102a_dictio.words','s0102a_dictio.words']   
    os.chdir("/home/ambroise/Documents/LSC-Internship/data/data_cleaned")
    Get_Statistics(file_list)
        


