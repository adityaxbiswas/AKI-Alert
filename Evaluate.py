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

#os.chdir('/home/aditya/Projects/AKI Alert/Code/first/')
from helper import AUUC
#os.chdir('/home/aditya/Projects/AKI Alert/Code/second/')
from Models import TLearner, XLearner, ZLearner, ProgLearner
#os.chdir('/home/aditya/Projects/AKI Alert/Data/')




action = 'assignment'
#target = 'yLastPer'
target = 'yMaxPer'

predictorBase = 'cratio0 cratio1 mcv mchc icuatalert c0value c1value bicarbonate bun age malegender hemoglobin wbc plateletcount sodium surgical'.split(' ')
predictorBank = ['uaprotein', 'timesec', 'redcelldistribution', 'alkphos', 'cdeltapercent', 'cslope',
                  'aaorno', 'pt', 'eospercent', 'paralyticcategory', 'neutrophilabs', 'basosabs','uaspecgrav',
                  'bilitotal', 'magnesium', 'missing', 'orders', 'nsaidcategory','chloride', 'glucose', 'pressorcategory', 
                  'hematocrit', 'mch', 'lactate', 'phosphorus','chf', 'basophilpercent',
                  'lymphpercent', 'leukocyteabs', 'neutrophilpercent', 'monopercent', 'monoabs', 'eosinophilabs', 'potassium',
                  'loopcategory', 'acearbcategory', 'hctzcategory', 'antibioticcategory', 'narcoticcategory']
predictorsAll = predictorBase + predictorBank

predictorsBinary = ['loopcategory', 'acearbcategory', 'hctzcategory', 'antibioticcategory', 'narcoticcategory',
                    'pressorcategory', 'nsaidcategory', 'paralyticcategory', 'aaorno', 'icuatalert',
                    'malegender', 'surgical', 'chf']
predictorsFloat = list(set(predictorsAll) - set(predictorsBinary))


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
        
    dfTrain = df.loc[trainValIndex,:]
    dfTest = df.loc[testIndex,:]
    imputer = Imputer()
    scaler = StandardScaler()
    xTrain = imputer.fit_transform(dfTrain[predictorBase + predictorBank].values)
    xTest = imputer.transform(dfTest[predictorBase + predictorBank].values)
    xTrain = pd.DataFrame(scaler.fit_transform(xTrain), columns = predictorBase + predictorBank, index = dfTrain.index)
    xTest = pd.DataFrame(scaler.transform(xTest), columns = predictorBase + predictorBank, index = dfTest.index)
    
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





baseStr = 'LearnerMaxTemporal.csv'
saveNameDict = {XLearner: 'X'+baseStr, ZLearner: 'Z'+baseStr, TLearner: 'T'+baseStr, 
            ProgLearner: 'Prog'+baseStr}


randomTreatments = False
temporal = True
predictEverything = True
iterations = 1
cvIters = 100

for modelClass in [XLearner, ZLearner, TLearner, ProgLearner]:
    allPredictions = pd.DataFrame(index = df.index)
    allPredictions[target] = df[target]
    allPredictions[action] = df[action]
    # checks if any of the 3 new added variables are ever chosen in a model
    selectCheck = np.array([False, False, False])
    
    trainIndex, valIndex, testIndex = makeSetIndices(df, 0.5, 0.2)
    trainValIndex = trainIndex + valIndex
    trainIndex, valIndex = trainIndex[:sum(trainValIndex)], valIndex[:sum(trainValIndex)]
    #modelClass = ZLearner
    
    auucs = []
    counter = 0
    while counter < iterations:
        X, Y, A = refresh(temporal = temporal)
        xTrainVal, xTest = X
        yTrainVal, yTest = Y
        if randomTreatments:
            aTrainVal = np.random.choice([True, False], len(yTrainVal))
            aTest = np.random.choice([True, False], len(yTest))
        else:
            aTrainVal, aTest = A 

        
        ## cross validation loop
        modelList = [modelClass() for i in range(cvIters)]
        predictorList = [None for i in range(cvIters)]
        uTest = np.zeros((len(yTest), cvIters))
        for j in range(cvIters):
            
            ## shuffle training and validation and then separate
            shuffleIndices = np.random.permutation(range(len(yTrainVal)))
            xTrainVal, yTrainVal, aTrainVal = xTrainVal.iloc[shuffleIndices,:], yTrainVal[shuffleIndices], aTrainVal[shuffleIndices]
            xTrain, xVal = xTrainVal.iloc[trainIndex,:], xTrainVal.iloc[valIndex,:]
            yTrain, yVal = yTrainVal[trainIndex], yTrainVal[valIndex]
            aTrain, aVal = aTrainVal[trainIndex], aTrainVal[valIndex]
            
            ## full forward feature selection process
            predictorList[j] = featureSelect(modelClass, xTrain, xVal, yTrain, yVal, aTrain, aVal)
            
            ## train and predict with final features
            modelList[j].fit(xTrain[predictorList[j]].values, yTrain, aTrain)
            uTest[:,j] = modelList[j].predict(xTest[predictorList[j]].values)
            print(j)
        
        ## turn to true for predicting everything for this cut
        if predictEverything:
            xAll = pd.concat([xTrainVal, xTest], axis = 0)
            A = np.concatenate((aTrainVal, aTest), axis = 0)
            U = np.concatenate([modelList[j].predict(xAll[predictorList[j]].values)[:,None] for j in range(cvIters)], axis = 1)
            Y = np.concatenate((yTrainVal, yTest), axis = 0)
            
            results = pd.DataFrame(data = {'uplift': np.mean(U, axis = 1), 'testing': testIndex, 
                                        'alert': A, 'yMaxPer': Y}, index = xAll.index)
            results.to_csv(saveNameDict[modelClass])
        
        
        ## update the checker if any one of the 3 are used
        for i, predictor in enumerate(['loopcategory', 'acearbcategory', 'hctzcategory']):
            selectCheck[i] = np.logical_or(selectCheck[i], 
                       np.any(np.array([predictor in l for l in predictorList])))
        
        
        ## evaluate and add performance to list
        auuc = AUUC(np.mean(uTest, axis = 1), yTest, aTest, graph = True)
        auucs += [auuc]
        
        
        # below line for a fully randomly generated set of predictions
        # useful for exploring variance
        #uTest = pd.DataFrame(np.random.uniform(-10, 10, size = len(yTest)), index = xTest.index)
        
        ## record test set predictions
        uTest = pd.DataFrame(np.mean(uTest, axis = 1), index = xTest.index)
        allPredictions[counter] = uTest
        
        print(counter)
        counter += 1
        
        
            
    
    print('\nMean: ' + str(np.mean(auucs)))
    print('Std: '+ str(np.std(auucs)))

        
    allPredictions.to_csv(saveNameDict[modelClass])
        
        
        

        