# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:15:46 2016

@author: ambroise


Get statistics phones/context

Input : corpus or corpora (here whole buckeye corpus)
Output : number of combinaisons phone1-phone2-phone3
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os

import codecs as cd

import csv
import glob

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    
def load_data(pathname):
    
    os.chdir(pathname)
    datasets = glob.glob(pathname + "/*.words")
  
    text=[]
    for dataset in datasets:
        with cd.open(dataset, 'r', encoding='utf8') as f:
            text = text + f.read().lower().split()
      #      text=text[0]
    
    return text

def Get_Statistics(pathname):
    
    dataset=load_data(pathname)
    print(len(dataset))
    #dataset=['a','v','b','b','f','a','b','v','a']
   

    dico={}
    dico_combined={}
    
    for i in range(0, len(dataset)-2):
        if (dataset[i],dataset[i+2],dataset[i+1])  not in dico:
            dico[(dataset[i],dataset[i+2],dataset[i+1])] = 1
        else:
            dico[(dataset[i],dataset[i+2],dataset[i+1])] += 1

    for i in range(0, len(dataset)-2):
        
        if (dataset[i],dataset[i+2],dataset[i+1])in dico_combined:
            dico_combined[(dataset[i],dataset[i+2],dataset[i+1])] += 1
            
        elif (dataset[i+2],dataset[i],dataset[i+1]) in dico_combined:
            print(dataset[i],dataset[i+2]) 
            dico_combined[(dataset[i+2],dataset[i],dataset[i+1])] += 1
            
        else:
            dico_combined[(dataset[i],dataset[i+2],dataset[i+1])] = 1
            
    #print(dico)
    print(len(dico))
    print(len(dico_combined))

    writer = csv.writer(open('GetStat2.csv', 'wb'))
    for key, value in dico.items():
        writer.writerow([(key[0],key[1]), key[2], value])
        
    writer = csv.writer(open('GetStat2_both.csv', 'wb'))
    for key, value in dico.items():
        cont_comb = (min(key[:2]),max(key[:2]))
        writer.writerow([(key[0],key[1]),cont_comb, key[2], value])
        
    writer = csv.writer(open('GetStat2_combined.csv', 'wb'))
    for key, value in dico_combined.items():
        writer.writerow([(key[0],key[1]), key[2], value])        
        
    print('FINI !')    
    
    
    
    

if __name__ == '__main__':
    pathname = "/home/ambroise/Documents/LSC-Internship/data/data_cleaned/Buckeye_dictio"
    #file_list = ['s0101a_dictio.words','s2501b_dictio.words'] 
    os.chdir("/home/ambroise/Documents/LSC-Internship/data/data_cleaned")
    Get_Statistics(pathname)
        


