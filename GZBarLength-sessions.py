#!/usr/bin/env python
#generate session length data
__author__ = 'vmaidel'
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import json
import re
import os
import pickle
import calendar
import time

def createSubjectID(metadata):
    return re.search(r'^{"([0-9]+)":',metadata).group(1)

def toSec(diffs_to_convert):
    if pd.notnull(diffs_to_convert):
        return calendar.timegm(time.strptime(str(diffs_to_convert), '%Y-%m-%d %H:%M:%S'))
    else:
        return np.nan

n=1
def calcSessionNum(diffs):
    global n
    if  pd.notnull(diffs) and diffs<=1800:
        return n
    elif pd.isnull(diffs):
        n=1
        return n
    elif pd.notnull(diffs) and diffs>1800:
        n=n+1
        return n

print "Loading all the pickles..."
features_DF=pd.read_pickle('features.p')
#assign the features to X
#fill NAs with 0 - in the future consider a different way of dealing with NAs
features_DF = features_DF.fillna(0)

# read the csv files
classification_dataDF=pd.read_csv("classification_data.csv")
subject_dataDF=pd.read_csv("subject_data.csv")

#timebeforefiledownload=datetime.datetime.now()
#
#use regex to extract just the subject id from the subject_data field 
classification_dataDF['subject_id']=classification_dataDF['subject_data'].apply(createSubjectID)
classification_dataDF[['subject_id']] = classification_dataDF[['subject_id']].astype(int)


classification_dataDF['created_at'] = pd.to_datetime(classification_dataDF['created_at'])
classification_dataDF.sort(['user_name', 'created_at'], inplace=True)

#calculate the gaps between classifications
classification_dataDF['diffs'] = classification_dataDF.groupby(['user_name'])['created_at'].transform(lambda x: x.diff()) 
#the result is returned in time since epoch, so this will convert the gaps to seconds
classification_dataDF['diffs']=classification_dataDF['diffs'].apply(toSec)
#calculate session number for a diff between classifications that lasted longer than 1800 seconds
classification_dataDF['session_num'] = classification_dataDF['diffs'].apply(calcSessionNum)

session_dataDF = classification_dataDF.groupby(['user_name','session_num'])['session_num'].agg(['count']).reset_index()
session_dataDF.rename(columns={'count' : 'session_length'},inplace=True)

classification_dataDF=pd.merge(classification_dataDF,session_dataDF,on=['user_name','session_num'])

features_DF[['subject_id']] = features_DF[['subject_id']].astype(int)
features_DF=features_DF.fillna(0)

classification_dataDF=pd.merge(classification_dataDF,features_DF,on=['subject_id'],how='left')
columnsToAgg=classification_dataDF.columns[0:119].values.tolist()

#show columns and their numbers
#for idx, val in enumerate(columnsToAgg):
#        print idx, val
#what columns to remove from the list
removeset = set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,34,35,36,37,114,116])
#remove unwanted columns from a list
columnsToAgg=[v for i, v in enumerate(columnsToAgg) if i not in removeset]
print columnsToAgg
#some cells in the PHOTOZ column contain blanks, which makes it difficult 
classification_dataDF.loc[classification_dataDF['PHOTOZ']=="",'PHOTOZ']=0
classification_dataDF[columnsToAgg]=classification_dataDF[columnsToAgg].astype(float)
#the columns to group by exclude the columns to aggregate and the columns to aggregate exclude the ones that where used in the group by
session_classification_dataDF=classification_dataDF.groupby(['user_name','session_num','session_length'])[columnsToAgg].median().reset_index()

session_classification_dataDF.iloc[0:10,:].to_csv('small_classifications_sessions.csv',sep=',')
print "\nPickling classification data...\n"
session_classification_dataDF.to_pickle('classifications.p')

