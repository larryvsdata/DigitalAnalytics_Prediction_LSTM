#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:03:45 2018

@author: ermanbekaroglu
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random as rd

class behaviourPrediction:

    
    def __init__(self):
        self.dfTrain=None
        self.actionCodes={}
        self.dfTrainRefactored=None
        self.XValues=[]
        self.yValues=[]
        self.testX=[]
        self.trainCutOff=150000
        self.cutOffLength=6
        self.splitRatio=0.7
        self.validationX=None
        self.validationY=None
        self.validationPred=None
        self.dfTest=None
        self.dfTestRefactored=None
        self.model=None
        self.ValidationAccuracy=0
        self.dropProbability=0.85
        self.trainIterations=300
        self.XValuesFiltered=[]
        self.yValuesFiltered=[]
        self.eligibleTestUsers=[]
        self.testResults=[]
        self.buyingCustomers=[]
        self.purchasePattern=[]
        self.reverseActions={}
        self.patternDict={}
        self.possibleBuyersDf=None
        self.probabilities=None
        
    def readTrainData(self):
        self.dfTrain=pd.read_csv("training.tsv", sep='\t', header=None)
        self.dfTrain.columns = ['username', 'actionDate','action']
        
    def readTestData(self):
        self.dfTest=pd.read_csv("test.tsv", sep='\t', header=None)
        self.dfTest.columns = ['username', 'actionDate','action']
        
    def setActionCodes(self):
        actions=list(set(self.dfTrain["action"]))
        for ii in range(len(actions)):
            self.actionCodes[actions[ii]]=ii
            
#        print(self.actionCodes)
            
    def convertDays(self,dFrame):
        ref=pd.Timestamp(dFrame['actionDate'].iloc[0])
        dFrame['actionDate']=dFrame['actionDate'].apply(lambda x: (pd.Timestamp(x)-ref).days)
        
    def refactorTrainData(self):
        self.dfTrainRefactored=self.dfTrain.copy()
        self.dfTrainRefactored["action"]=self.dfTrain["action"].apply(lambda x: self.actionCodes[x])
        
    def readXYValues(self):
        userNames=list(set(self.dfTrainRefactored['username']))


        counter=0
                
        for ii in range(self.trainCutOff):
            
            user=userNames[ii]
            d=self.dfTrainRefactored[self.dfTrainRefactored['username']==user]
            
            if len(d)>self.cutOffLength-1:
                if self.actionCodes["Purchase"] in list(d['action']):
                    
                    indices=d.index[d['action']==self.actionCodes["Purchase"]].tolist()                    
                    if -d.index[0]+indices[0]>self.cutOffLength-1:
#                        print(-d.index[0]+indices[0])

                        startPoint=rd.randint(d.index[0],indices[0]-self.cutOffLength+1)
   
                        d=d.ix[startPoint:startPoint+self.cutOffLength-2,1:3].copy()
                        self.convertDays(d)
                        self.XValues.append(d.values)
                        self.yValues.append(1)
#                        print(d.values)
                        
                else: 

                    startPoint=rd.randint(d.index[0],d.index[-1]-self.cutOffLength+1)
                    d=d.ix[startPoint:startPoint+self.cutOffLength-2,1:3].copy()
                    self.convertDays(d)
                    self.XValues.append(d.values)
                    self.yValues.append(0)
#                    print(d.values)
                    
                    
            if counter%100==0:
                print('Train counter: ',counter)
            counter+=1
    
                    
        self.XValues=np.array(self.XValues)    
        self.yValues=np.array(self.yValues) 
        
        
    def saveTrainingValues(self):
        with open('trainX_'+str(self.trainCutOff)+'_'+str(self.cutOffLength), 'wb') as fp:
            pickle.dump(self.XValues,fp)
            
        with open('trainY'+str(self.trainCutOff)+'_'+str(self.cutOffLength), 'wb') as fp:
            pickle.dump(self.yValues,fp)
            

            
    def readTrainingFromFile(self):
        with open('trainX_'+str(self.trainCutOff)+'_'+str(self.cutOffLength), 'rb') as fp:
            self.XValues=pickle.load(fp)
            
        with open('trainY'+str(self.trainCutOff)+'_'+str(self.cutOffLength), 'rb') as fp:
            self.yValues=pickle.load(fp)
            
    def refactorAndSave(self):

        self.refactorTrainData()
        self.readXYValues()
        self.saveTrainingValues()
        
    def checkAndLaunch(self):
        myFile1='trainX_'+str(self.trainCutOff)+'_'+str(self.cutOffLength)
        myFile2='trainY'+str(self.trainCutOff)+'_'+str(self.cutOffLength)
        
        path1=Path(myFile1)
        path2=Path(myFile2)
        
        self.readTrainData()
        self.setActionCodes()
        
        if path1.is_file() and path2.is_file():
            self.readTrainingFromFile()
            
            
        else :
            self.refactorAndSave()
#            pass
            
    def splitValidaton(self):
        totalLength=self.XValues.shape[0]
        splitPoint=int(self.splitRatio*totalLength)
        self.validationX=self.XValues[splitPoint:]
        self.validationY=self.yValues[splitPoint:]
        self.XValues=self.XValues[:splitPoint:]
        self.yValues=self.yValues[:splitPoint]
        
    def getTestData(self):
        self.readTestData()
        testUsers=list(set(self.dfTest['username']))
        self.dfTestRefactored=self.dfTest.copy()
        self.dfTestRefactored["action"]=self.dfTest["action"].apply(lambda x: self.actionCodes[x])
        
        
        counter2=0
        for ii in range(len(testUsers)):
            
            user=testUsers[ii]
            d=self.dfTestRefactored[self.dfTestRefactored['username']==user]
            if len(d)>self.cutOffLength-1:
                startPoint=rd.randint(d.index[0],d.index[-1]-self.cutOffLength+1)
                d=d.ix[startPoint:startPoint+self.cutOffLength-2,1:3].copy()
                self.convertDays(d)
                self.testX.append(d.values)
                self.eligibleTestUsers.append(user)
#                print(d)
            if counter2%100==0:
                print('Test counter: ',counter2)
            counter2+=1
        
        self.testX=np.array(self.testX) 
        self.saveTest()
        
    def saveTest(self):
        with open('testX_'+str(self.cutOffLength), 'wb') as fp:
            pickle.dump(self.testX,fp)
        with open('eligibleTestUsers_'+str(self.cutOffLength), 'wb') as fp:
            pickle.dump(self.eligibleTestUsers,fp)    
        
            
    def readTest(self):
        with open('testX_'+str(self.cutOffLength), 'rb') as fp:
            self.testX=pickle.load(fp)
            
        with open('eligibleTestUsers_'+str(self.cutOffLength), 'rb') as fp:
            self.eligibleTestUsers=pickle.load(fp)
        
        
    def setUpTest(self):
        
        myFileTest='testX_'+str(self.cutOffLength)
        myFileEligible='eligibleTestUsers_'+str(self.cutOffLength)
        pathTest=Path(myFileTest)
        pathEligible=Path(myFileEligible)
        
        if pathTest.is_file() and pathEligible.is_file():
            self.readTest()
            
        else :
            self.getTestData()
    
    
        
    def buildModel(self):
        K.clear_session()

        self.model=Sequential()
        
        self.model.add(LSTM(10,input_shape=(self.cutOffLength-1,2)))
        
        self.model.add(Dense(1,activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        
    def trainModel(self):
        self.model.fit(self.XValues, self.yValues, epochs=self.trainIterations, batch_size=1,verbose=1)
        
    def testModel(self):
        self.validationPred=self.model.predict_classes(self.validationX)
#        plt.plot(self.validationY)
#        plt.plot(self.validationPred)
        self.ValidationAccuracy=accuracy_score(self.validationPred,self.validationY)
        print('Validation Accuracy : ',self.ValidationAccuracy)
        
        
    def saveModel(self):
        with open('LSTM_'+str(self.trainCutOff)+str(self.trainIterations)+str(self.dropProbability)+str(self.cutOffLength), 'wb') as fp:
            pickle.dump(self.model,fp)
            
    def readModel(self):
        with open('LSTM_'+str(self.trainCutOff)+str(self.trainIterations)+str(self.dropProbability)+str(self.cutOffLength), 'rb') as fp:
            self.model=pickle.load(fp)
            
    def launchModel(self):
        myFileModel='LSTM_'+str(self.trainCutOff)+str(self.trainIterations)+str(self.dropProbability)+str(self.cutOffLength)
        
        
        path1=Path(myFileModel)
        
        if path1.is_file() :
            self.readModel()
            
        else :
            self.trainModel()
            self.saveModel()
        
        
        
        
        
        
    def filterValues(self):
        
        for ii in range(len(self.XValues)):
            if self.yValues[ii]==0:
                if rd.random()>self.dropProbability:
                    self.XValuesFiltered.append(self.XValues[ii])
                    self.yValuesFiltered.append(self.yValues[ii])
            else:
                    self.XValuesFiltered.append(self.XValues[ii])
                    self.yValuesFiltered.append(self.yValues[ii])
#            if ii%100==0:
#                print('Filter index: ', ii)
                
        self.XValues=  np.array(self.XValuesFiltered )
        self.yValues=  np.array(self.yValuesFiltered )
        
        
    def getTestResults(self):
        self.testResults=self.model.predict_classes(self.testX)
        
    def getReverseCodes(self):
        
        for key, value in self.actionCodes.items():
            self.reverseActions[value]=key
            
    def getPayingUsersActions(self):
        
        for ii in range(len(self.testResults)):
            if self.testResults[ii][0]==1:
                self.buyingCustomers.append(self.eligibleTestUsers[ii])
                self.purchasePattern.append(list(self.testX[ii,:,1]))
                
    def intListToString(self,intList):
        
        myString=""
        for item in intList:
            myString+=str(item)
        return myString
            
    def convertPatterns(self):
        self.purchasePattern=[self.intListToString(pattern) for pattern in self.purchasePattern]
        
    def formPatternDict(self):
        
        for pattern in self.purchasePattern:
            if pattern in self.patternDict:
                self.patternDict[pattern]+=1
            else:
                self.patternDict[pattern]=1
                
                
    def reportPatterns(self, number):
        sorted_Patterns=sorted(self.patternDict.items(),key=lambda x:-x[1])
        
        print('Top ', number, ' purchasing patterns: ')
        
        for ind in range(number):
            print(self.convertPatternToText(sorted_Patterns[ind][0]))
                
        
    def convertPatternToText(self,pattern):
        pattern=list(pattern)
        
        patternString=""
        for action in pattern:
            patternString+=self.reverseActions[int(action)]+' '
            
        return patternString
    
    def getPossibleBuyers(self):
        self.probabilities=self.model.predict(self.testX)
        self.probabilities=[probability[0] for probability in list(self.probabilities)]
        temp=list(zip(self.eligibleTestUsers,self.probabilities))
        self.possibleBuyersDf=pd.DataFrame(temp,columns=['UserName','Probability'])
        self.possibleBuyersDf=self.possibleBuyersDf.sort_values(by='Probability',ascending=False)
        
        top1000=list(self.possibleBuyersDf['UserName'])[:1000]
            
        with open("Top1000PossibleBuyers.txt",'w') as f:
            for item in top1000:
                f.write("%s\n" %item)
                    
        
            
            
        
        

        
if __name__ == "__main__":
    
    BP=behaviourPrediction()
    BP.checkAndLaunch()
    BP.splitValidaton()
    BP.filterValues()
    BP.buildModel()
    BP.trainModel()
    BP.testModel()
    BP.setUpTest()
    BP.getTestResults()
    BP.getPayingUsersActions()
    BP.convertPatterns()
    BP.formPatternDict()
    BP.getReverseCodes()
    BP.reportPatterns(20)
    BP.getPossibleBuyers()
#

    
    
    
    

        
        
        
        
        
        
        