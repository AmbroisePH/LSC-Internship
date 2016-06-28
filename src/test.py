# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:40:02 2016

@author: ambroise
"""

import csv
import os




loop=0
l=[]
def func(inp1, inp2):
    global p
    p = p+inp1
    print(p)
    return inp1*inp2



p=0
for epoch in range (1,5,1):
    func(3,4)
    
    

