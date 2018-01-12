#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 21:22:56 2017

@author: aditya
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer, StandardScaler
import os

os.chdir('/home/aditya/Projects/AKI Alert/Code/first/')
from helper import AUUC
os.chdir('/home/aditya/Projects/AKI Alert/Code/second/')
from Models import TLearner, XLearner, ZLearner, ProgLearner
os.chdir('/home/aditya/Projects/AKI Alert/Data/')




action = 'assignment'
target = 'creatoutcome7percent'

predictorBase = 'cratio0 cratio1 mcv mchc icuatalert c0value c1value bicarbonate bun age malegender hemoglobin wbc plateletcount sodium surgical'.split(' ')
predictorBank = ['uaprotein', 'timesec', 'redcelldistribution', 'alkphos', 'cdeltapercent', 'cslope',
                  'aaorno', 'pt', 'eospercent', 'paralyticcategory', 'neutrophilabs', 'basosabs','uaspecgrav',
                  'bilitotal', 'magnesium', 'missing', 'orders', 'nsaidcategory','chloride', 'glucose', 'pressorcategory', 
                  'hematocrit', 'mch', 'lactate', 'phosphorus','chf', 'basophilpercent',
                  'lymphpercent', 'leukocyteabs', 'neutrophilpercent', 'monopercent', 'monoabs', 'eosinophilabs', 'potassium']




def makeSetIndices(df, trainPercent, valPercent):
    trainIndex = np.repeat(False, len(df))
    valIndex = np.repeat(False, len(df))
    testIndex = np.repeat(False, len(df))
    
    cutTrain = int(round(len(df)*trainPercent))
    cutVal = int(round(len(df)*(trainPercent + valPercent)))
    
    trainIndex[:cutTrain] = True
    valIndex[cutTrain:cutVal] = True
    testIndex[cutVal:] = True
    
    return trainIndex, valIndex, testIndex

def refresh(temporal = False):
    global df, trainValIndex, testIndex, predictorBase, predictorBank
    if temporal is True:
        df = df.sort_values(by = 'timesec')
    else:
        df = df.sample(frac = 1.)
        
    dfTrain = df[trainValIndex]
    dfTest = df[testIndex]
    imputer = Imputer()
    scaler = StandardScaler()
    xTrain = imputer.fit_transform(dfTrain[predictorBase + predictorBank].values)
    xTest = imputer.transform(dfTest[predictorBase + predictorBank].values)
    xTrain = pd.DataFrame(scaler.fit_transform(xTrain), columns = predictorBase + predictorBank, index = dfTrain.index.values.astype('int'))
    xTest = pd.DataFrame(scaler.transform(xTest), columns = predictorBase + predictorBank, index = dfTest.index.values.astype('int'))
    
    yTrain, yTest = dfTrain[target].values, dfTest[target].values
    aTrain, aTest = dfTrain[action].values == 1, dfTest[action].values == 1
    
    return (xTrain, xTest), (yTrain, yTest), (aTrain, aTest)


def featureSelect(modelClass, xTrain, xVal, yTrain, yVal, aTrain, aVal):
    global predictorBase, predictorBank
    predictorsCurrent = predictorBase.copy()
    searching = True
    bestPerformance = 1e8
    while searching:
        bestSubPerformance = 1e8
        for k in range(len(predictorBank)):
            predictors = predictorsCurrent + [predictorBank[k]]
            model = modelClass()
            model.fit(xTrain[predictors].values, yTrain, aTrain)
            
            uVal = model.predict(xVal[predictors].values)
            
            if modelClass is ProgLearner:
                auuc = np.mean(np.power(uVal - yVal, 2))
            else:
                auuc = AUUC(uVal, yVal, aVal, graph = False)
    
            if auuc < bestSubPerformance:
                currentFeature = predictorBank[k]
                bestSubPerformance = auuc
                
        if bestSubPerformance < bestPerformance:
            bestPerformance = bestSubPerformance
            predictorsCurrent += [currentFeature]
        else:
            searching = False
    
    return predictorsCurrent





allPredictions = pd.DataFrame(index = df.index.values.astype('int64'))

trainIndex, valIndex, testIndex = makeSetIndices(df, 0.5, 0.2)
trainValIndex = trainIndex + valIndex
trainIndex, valIndex = trainIndex[:sum(trainValIndex)], valIndex[:sum(trainValIndex)]
modelClass = XLearner

auucs = []
counter = 0
iterations = 100
while counter < iterations:
    X, Y, A = refresh(temporal = False)
    xTrainVal, xTest = X
    yTrainVal, yTest = Y
    aTrainVal, aTest = A
    #aTrainVal, aTest = np.random.choice([True, False], len(aTrainVal)), np.random.choice([True, False], len(aTest))
    
    
    
    ## cross validation loop
#    cvIters = 10
#    modelList = [modelClass() for i in range(cvIters)]
#    predictorList = [None for i in range(cvIters)]
#    uTest = np.zeros((len(yTest), cvIters))
#    for j in range(cvIters):
#        
#        ## shuffle and separate
#        shuffleIndices = np.random.permutation(range(len(yTrainVal)))
#        xTrainVal, yTrainVal, aTrainVal = xTrainVal.iloc[shuffleIndices,:], yTrainVal[shuffleIndices], aTrainVal[shuffleIndices]
#        xTrain, xVal = xTrainVal.iloc[trainIndex,:], xTrainVal.iloc[valIndex,:]
#        yTrain, yVal = yTrainVal[trainIndex], yTrainVal[valIndex]
#        aTrain, aVal = aTrainVal[trainIndex], aTrainVal[valIndex]
#        
#        ## feature select
#        predictorList[j] = featureSelect(modelClass, xTrain, xVal, yTrain, yVal, aTrain, aVal)
#        
#        ## train and predict
#        modelList[j].fit(xTrain[predictorList[j]].values, yTrain, aTrain)
#        uTest[:,j] = modelList[j].predict(xTest[predictorList[j]].values)

    
    ## for predicting everything
    if False:
        xAll = pd.concat([xTrainVal, xTest])
        A = np.concatenate((aTrainVal, aTest))
        U = np.concatenate([modelList[j].predict(xAll[predictorList[j]].values)[:,None] for j in range(cvIters)], axis = 1)
        
        temp = pd.DataFrame(data = {'uplift': np.mean(U, axis = 1), 'test': testIndex, 'randomAlert': A}, index = xAll.index)
        temp.to_csv('ZLearnerRand.csv')
    
    
    
#    auuc = AUUC(np.mean(uTest, axis = 1), yTest, aTest, graph = False)
#    auucs += [auuc]
    
    
    #uTest = pd.DataFrame(np.mean(uTest, axis = 1), index = xTest.index)
    uTest = pd.DataFrame(np.random.uniform(-10, 10, size = len(yTest)), index = xTest.index)
    allPredictions[counter] = uTest
    
    print(counter)
    counter += 1
    
    
        

print('\nMean: ' + str(np.mean(auucs)))
print('Std: '+ str(np.std(auucs)))

        
        
        
        
        
        
        
        
        
        
        