#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:41:00 2020

@author: ujjwalkulkarni
"""

import csv
import nltk
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#import re, collections
#from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import Counter 
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.svm import SVR
#from sklearn import ensemble
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import cohen_kappa_score
import pickle
import io 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

#loaded_model = pickle.load(open('finalized_model_2.sav', 'rb'))
filename = "SOP.csv"
data = []
for row in csv.reader(open(filename, 'r'), delimiter=','):
    data.append(row)
    
dataset=data[1][2]


#word_tokenize accepts a string as an input, not a file. 
stop_words = set(stopwords.words("english"))

#add words that aren't in the NLTK stopwords list
new_stopwords = ['I','The','My','computer','part','It']
new_stopwords_list = stop_words.union(new_stopwords)
file1 = dataset
line = file1# Use this to read file content as a stream: 
words = line.split() 

#for r in words: 
 #   if not r in new_stopwords_list: 
  #      appendFile = open('filteredtext8.txt','a') 
   #     appendFile.write(" "+r) 
    #    appendFile.close() 
  
#print(word_tokens) 
#print(filtered_sentence) 



file2 = open("filteredtext8.txt") 
data_set=file2.read()








count=0             #count of a specific word
maxcount=0          #maximum among the count of each words
l=[]    
#words=content.split()
#print(data_set)
  
# split() returns list of all the words in the string 
split_it = data_set.split() 
  
# Pass the split_it list to instance of Counter class. 
Counter = Counter(split_it) 
  
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur = Counter.most_common(10) 
  
#print(most_occur)
#print(type(most_occur))

#print ("List index-value are : ") 
#for i in range(len(most_occur)): 
 #   print (i, end = " ") 
  #  print (most_occur[i]) 

#print(data_set)          