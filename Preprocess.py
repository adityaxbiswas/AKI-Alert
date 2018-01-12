#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:54:43 2017

@author: aditya
"""


from scipy.stats import skewtest
import numpy as np
import seaborn as sns
import pandas as pd
import os
    
os.chdir('/home/aditya/Projects/AKI Alert/Code/first/')
from helper import loadDict, searchName, createDict
    
def isSkewed(df, vars, significance = 1e-5):
    results = [False for i in range(len(vars))]
    for i, var in enumerate(vars):
        data = df.loc[np.isfinite(df[var]),var].values
        if len(np.unique(data)) > 2:
            results[i] = skewtest(data)[1] < significance
    return results

def logTrans(df, names):
    for name in names:
        df.loc[np.isfinite(df[name]),name] =  np.log(df.loc[np.isfinite(df[name]), name].values + 1)
    return df

os.chdir('/home/aditya/Projects/AKI Alert/Data/')
dataFileName = '3 day outcome all patients.dta'


df = pd.read_stata(dataFileName)
df.index = df['uid_id'].values
df['timesec'] = pd.to_numeric(df['timesec'], errors = 'coerce')
df['assignment'] = 1*(df['assignment'] == 'Alert')
df = df[df['akinstage'] == 1]


df['creatchange'] = df['c1value'] - df['c0value']
df['cslope'] = df['creatchange']/df['creat0tocreat1time']
df['cratio'] = df['c1value']/df['c0value']
baseline = df['c0value']
df['cdeltapercent'] = df['creatchange']/baseline
df['creatoutcome7'] = df['lastcreat72']
df['creatoutcome7percent'] = (df['creatoutcome7'] - df['c1value'])/df['c1value']
df['Z'] = pd.Series(np.zeros(len(df)), index = df.index)
#df.loc[df['assignment'] == 1, 'Z'] = df.loc[df['assignment'] == 1, 'creatoutcome7percent']
#df.loc[df['assignment'] == 0, 'Z'] = -df.loc[df['assignment'] == 0, 'creatoutcome7percent']
#df.loc[df['creatoutcome7percent'] == 0, 'creatoutcome7percent'] = (df.loc[df['creatoutcome7percent'] == 0, 'c0value'] -   \
#       df.loc[df['creatoutcome7percent'] == 0, 'c1value'])/df.loc[df['creatoutcome7percent'] == 0, 'c1value']  


df['cratio0'] = df['icuatalert']*df['cratio']
df['cratio1'] = (1-df['icuatalert'])*df['cratio']


## load dictionary, flatten, and remove medications, missing vars, etc...
varDict = loadDict('multiview alert.csv', df)


# dont know why i have to do this, but for some reason the uplift package cant find the feature 'assignment' w/o it....
#df['missing'] = 6
df['missing'] = np.sum(np.isfinite(df.loc[:,'a1c':'wbc'].values), axis = 1)


df['orders'] = df[varDict['orders']].sum(axis = 1)
varDict['orders'] = ['missing', 'orders', 'timesec']
#predictors = [item for sublist in varDict.values() for item in sublist]
predictors = ['timesec', 'creatchange', 'cdeltapercent', 'cslope', 'cratio', 'c1value', 'c0value', 'age', 'aaorno', 'surgical',
          'malegender', 'mcv', 'bun', 'bilidirect', 'redcelldistribution', 'alkphos', 'hemoglobin', 'orders',
          'wbc', 'nsaidcategory', 'potassium', 'chloride', 'glucose', 'mchc', 'plateletcount', 'sodium', 'missing', 'icuatalert',
         'pressorcategory', 'uaspecgrav', 'uaprotein', 'alt', 'ast', 'pt', 'ptt', 'inr', 'hematocrit', 'mch',
         'chf', 'paralyticcategory', 'antibioticcategory', 'narcoticcategory',
         'basophilpercent', 'basosabs',
         'lymphpercent', 'leukocyteabs', 'neutrophilabs', 'neutrophilpercent', 'monopercent', 'monoabs', 'eosinophilabs',
         'eospercent', 'cratio0', 'cratio1']

## log transform heavily skewed variables
varsLogTrans = list(np.array(predictors)[isSkewed(df, predictors)])
df = logTrans(df, varsLogTrans)