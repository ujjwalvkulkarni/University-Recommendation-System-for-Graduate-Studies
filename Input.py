#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 7 21:40:21 2020

@author: KaranDavey
"""
import numpy as np
import csv
import time
import math
import random
from test import *
from topics import *
def main():
    #print(dataframe['essay'])
    #Inititalize the user by prompting it to enter their details
    userInfo = []
    time.sleep(1)
    print("Please enter the following data", end='\n')
    #print(sop)
    esscore= y_pred    
    time.sleep(1)
    name = input("Enter your name : ")
    time.sleep(1)
    phoneNumber = input("Enter phone number (0-9) : ")
    time.sleep(1)
    DOB = input("Enter Date of Birth (DD/MM/YYYY) : ")
    time.sleep(1)
    email = input("Enter email ID (abc@xyz.com) : ")
    time.sleep(1)
    userInfo.append(name)
    userInfo.append(phoneNumber)
    userInfo.append(DOB)
    userInfo.append(email)
    #userCollege = []

    #print("Enter 5 preferred list of colleges : ", end='\n')

    #for i in range(5):
    #    uc = input()
    #    userCollege.append(uc)
    print('\n')
    print("Welcome "+name)
    time.sleep(1)
    print("\n")
    #print("Your list : ")
    #print(userCollege)

    scoreEval()
    #print(userInfo)
    #User enter the list of colleges





def scoreEval():
    print("Welcome to URS, please enter the following details ", end='\n')
    time.sleep(1)
    quants = int(input("Enter Quantitative section score(130-170) : "))
    time.sleep(1)
    verbal = int(input("Enter Verbal section score(130-170) : "))
    time.sleep(1)

    q_score = quants-120
    v_score = verbal-120

    score = q_score+v_score
    
    esscore = y_pred
    #print(esscore)
    print('\n')
    print("Select the Language Proficiency Test taken : ",end='\n')
    time.sleep(1)
    print("1. TOEFL    2. IELTS    3. Yet to give one",end='\n')
    time.sleep(1)
    test = int(input("Enter your choice : "))
    time.sleep(1)
    if(test==1):
        a = int(input("Enter TOEFL score : "))
        time.sleep(1)
        if(a>=110):
            score+=20
        elif((a>=100) and (a<110)):
            score+=18
        elif((a>=90) and (a<100)):
            score+=16
        elif((a>=80) and (a<90)):
            score+=14
        else:
            score+=10

    
    if(test==2):
        b = float(input("Enter IELTS score : "))
        if(b>=8):
            score+=20
        elif(b>=7 and b<8):
            score+=18
        elif(b>=6 and b<7):
            score+=16
        else:
            score+=10

    if(test==3):
        score+=10

    #print(score)


    unimarks = float(input("Enter undergraduate overall percentage : "))
    time.sleep(1)
    if(unimarks>=90.0):
        score+=70
    elif(unimarks>=80.0 and unimarks<90.0):
        score+=60
    elif(unimarks>=70.0 and unimarks<80.0):
        score+=50
    elif(unimarks>=60.0 and unimarks<70.0):
        score+=40
    else:
        score+=25



    rp = int(input("Enter number of published and non-published papers : "))
    time.sleep(1)
    if(rp>=5):
        score+=10
    else:
        score+=rp
    
        

    recommender(score,esscore)
    

def recommender(score,esscore):

    filename = "Dataset.csv"
    data = []
    for row in csv.reader(open(filename, 'r'), delimiter=','):
        data.append(row)

    
    a1 = data[0][1:]
    a2 = data[1][1:]
    a3 = data[2][1:]
    b1 = data[3][1:]
    b2 = data[4][1:]
    b3 = data[5][1:]
    c = data[6][1:] 
    d = data[7][1:]
    
    print('\n')
    print("The SOP you have uploaded is as follows:")
    time.sleep(1)
    print('\n')
    #print(dataframe['essay'])
    filename = "SOP.csv"
    data = []
    for row in csv.reader(open(filename, 'r'), delimiter=','):
        data.append(row)
    
    dataset=data[1][2]
    print(dataset)
    #print('\n')
    print("Evaluating the SOP")
    time.sleep(0.5)
    print(".")
    time.sleep(0.5)
    print(".")
    time.sleep(0.5)
    print(".")
    time.sleep(0.5)
    print(".")
    time.sleep(0.5)
   # print('\n')
    print("Your SOP is based on the following topics : ")
    time.sleep(1)
    for i in range(len(most_occur)): 
        print (i+1, end = " ") 
        print (most_occur[i][0]) 
    time.sleep(1) 
    print('\n')
    print("Your SOP score is : ")
    time.sleep(1)
    print(np.rint(esscore*8.33))
    print('\n')
    print("Recommended list of universities to apply to are as follows :\n ")
    time.sleep(3)
    score=score+(esscore*8.330)
    #print(score)
    if(score>=265):
        printing(a1)
    elif(score>=260 and score<265):
        printing(a2)
    elif(score>=230 and score<260):
        printing(a3)
    elif(score>=200 and score<230):
        printing(b1)
    elif(score>=170 and score<200):
        printing(b2)
    elif(score>=140 and score<170):
        printing(b3)
    elif(score>=100 and score<140):
        printing(c)
    else:
        printing(d)
        
    #print(score)

def printing(x):
    for i in x:
        print(i)
        




main()
