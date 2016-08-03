#!/usr/bin/env python
#performs preprocessing on the data, among others, extracts image features
__author__ = 'vmaidel'
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import json
import urllib
import re
from subprocess import Popen, PIPE
import os
import pickle
import datetime

def returnLines(longString): return iter(longString.splitlines())

def createSubjectID(metadata):
    return re.search(r'"(.*)":{',metadata).group(1)

def extractMentionedSubj(discussion_ttl):
    subj_mentioned=re.search(r'^[Ss]ubject ([0-9]+)',discussion_ttl)
    if subj_mentioned is not None:
        return subj_mentioned.group(1)

# read the csv files
#workflow_dataDF=pd.read_csv("workflow_data.csv")
#workflow_contentDF=pd.read_csv("workflow_content.csv")
classification_dataDF=pd.read_csv("classification_data.csv")
subject_dataDF=pd.read_csv("subject_data.csv")
talk_commentsDF=pd.read_json("talk_comments.json")
collections_dataDF=pd.read_csv("gz-bar-collections.csv")

#count the subjects that were mentioned in a discussion (located in discussion title), subject_id contains the mentioned subject_id
talk_commentsDF['subject_id']=talk_commentsDF['discussion_title'].apply(extractMentionedSubj)
talk_commentsDF.to_csv('talk_commentsDF.csv',sep=',',encoding='utf-8')

#group by subject_id and discussion_id to count how many times a subject is mentioned (same discussion counts only once)
talk_mentions_aggDF=talk_commentsDF.groupby(['subject_id','discussion_id'])['subject_id'].agg(['count']).reset_index()
talk_mentions_aggDF=talk_mentions_aggDF.groupby('subject_id')['subject_id'].agg(['count']).reset_index()
talk_mentions_aggDF.rename(columns={ 'count' : 'talk_mentions_counts'},inplace=True)
talk_mentions_aggDF[['subject_id']] = talk_mentions_aggDF[['subject_id']].astype(int)

#group by subject_id and collection_id to count how many time times a subject is added to a collection (same collection and subject_id counts only once)
collections_aggDF=collections_dataDF.groupby(['subject_id','collection_id'])['subject_id'].agg(['count']).reset_index()
collections_aggDF=collections_aggDF.groupby('subject_id')['subject_id'].agg(['count']).reset_index()
collections_aggDF.rename(columns={ 'count' : 'collections_counts'},inplace=True)
collections_aggDF[['subject_id']] = collections_aggDF[['subject_id']].astype(int)

#merge the talk mentions and collection additions into one data frame, with an outer join
talk_collections_DF=pd.merge(talk_mentions_aggDF,collections_aggDF,how='outer',on='subject_id')
talk_collections_DF.fillna(0,inplace=True)

#apply a json.loads function on the whole locations column and save it in a new urls column, to enable extraction of the URL where the image is stored
subject_dataDF['urls']=subject_dataDF['locations'].map(lambda x: json.loads(x))
#apply a json.loads function on the whole meta data column and save it in a new urls column, to enable extraction of metadata that comes with the subject
subject_dataDF['metadata_fields']=subject_dataDF['metadata'].map(lambda x: json.loads(x))

#use regex to extract just the subject id from the subject_data field 
classification_dataDF['subject_id']=classification_dataDF['subject_data'].apply(createSubjectID)
#count classifications of each subject
classifications_aggDF=classification_dataDF.groupby('subject_id')['subject_id'].agg(['count']).reset_index()

timebeforefiledownload=datetime.datetime.now()

#go over all the urls and save them as jpeg files in the GZBarJPEGs directory
for index, row in subject_dataDF.iterrows():
    URLpair=row['urls']
    matchObj=re.search(r'(?:[^/][\d\w\.-]+)$(?<=(?:.jpeg))',URLpair['0'])
    #create a new column with disk location of the file that is about to be saved to disk - used later for matching between the df row and the parameters extracted with imagemagick
    subject_dataDF.loc[index,'disk_loc'] = 'GZBarJPEGs/'+matchObj.group()
    print "saving URL:"+str(URLpair['0'])
    #save the file to the GZBarJPEGs folder
    #urllib.urlretrieve(URLpair['0'],'GZBarJPEGs/'+matchObj.group())
    #adding fields that correspond to the names of the metadata fields
    metadata_set=row['metadata_fields']
    if row['subject_set_id']==31:
        subject_dataDF.loc[index,'RA']=metadata_set['RA']
        subject_dataDF.loc[index,'Dec']=metadata_set['Dec']
        subject_dataDF.loc[index,'SPECZ']=metadata_set['SPECZ']
        subject_dataDF.loc[index,'PHOTOZ']=metadata_set['PHOTOZ']
        subject_dataDF.loc[index,'ZQUALITY']=metadata_set['ZQUALITY']

timeafterfiledownload=datetime.datetime.now()
timetodownload=timeafterfiledownload-timebeforefiledownload
  
file_num=0
timebeforeidentify=datetime.datetime.now()
for file in os.listdir("/Users/citisci/Documents/Python/GZBarJPEGs/"):
    if file.endswith(".jpeg"):
        file_num=file_num+1
        print "Identifying file num:"+str(file_num)+" file name:"+str(file)
        p1 = Popen(["identify","-verbose","-features","1","-moments","-unique","GZBarJPEGs/"+file],stdout=PIPE)
        #print p1.communicate()
        imOutput = p1.stdout.read()
        #print imOutput
        linesIter=returnLines(imOutput)
        numOfIndents=0
        indentDict={}
        indentDict[0]='Image:'
        prevParamName=""
        for oneLine in linesIter:
            #get the name of the parameter (everything before the :)
            matchParamName=re.search(r'^[^:]+:',oneLine)
            if matchParamName is not None: #if there was a param name at the beginning of the line
                paramName=matchParamName.group()
                #calculate the number of indents of the current line
                numOfIndents=len(oneLine) - len(oneLine.lstrip(' '))
                if numOfIndents>0:
                    indentDict[numOfIndents]=indentDict[numOfIndents-2]+paramName
                    matchLineNoParam=re.search(r':(.*)',oneLine)
                    oneLineNoParam=matchLineNoParam.group()
                    finalLine=indentDict[numOfIndents]+oneLineNoParam+"["+str(numOfIndents)+"]"
                    prevParamName=paramName
            else: #if there was no param name at the beginning of the line (meaning that I need to take the previous line param name)
                finalLine=indentDict[numOfIndents-2]+prevParamName+":"+oneLine+"["+str(numOfIndents)+"]"
            if numOfIndents>0:    
                #separate between the param name and the actual values
                matchFinalLine=re.search(r'(.*)::(.*)\[',finalLine)
                if matchFinalLine.group(2) is not "":
                    fieldName = matchFinalLine.group(1)
                    chars_to_remove=[':',')','(',',']
                    #clean the potential field names
                    fieldName=fieldName.replace(' ','').replace(':','_').replace(',','_').replace('(','_').replace(')','_')
                    #print fieldName+"|"+matchFinalLine.group(2)
                    paramValue=matchFinalLine.group(2).strip()
                    if ',' not in paramValue:
                        matchParamNumOnly=re.search(r'(^-?[0-9.e-]+)',paramValue)
                        if matchParamNumOnly is not None:
                            paramValue=matchParamNumOnly.group()
                            #find the row that matches the disk location of the current file, create a new column and populate that cell with the values. By including this line in the IF, we save only the columns that contain numeric information. Return it to be not part of the last two ifs to get all the other fields as well.
                            subject_dataDF.loc[subject_dataDF['disk_loc']=="GZBarJPEGs/"+file,fieldName]=paramValue.strip()
print "time it took to save all files:"+str(timetodownload)
timeafteridentify=datetime.datetime.now()
print "time it took to identify all the files:"+str(timeafteridentify-timebeforeidentify)
#convert the subject_id column to be int in both data frames
#classifications_aggDF[['subject_id']] = classifications_aggDF[['subject_id']].astype(int)

subject_dataDF[['subject_id']] = subject_dataDF[['subject_id']].astype(int)
talk_collections_DF[['subject_id']] = talk_collections_DF[['subject_id']].astype(int)

#features_DF=pd.merge(subject_dataDF, classifications_aggDF,on='subject_id')
#features_DF.rename(columns={ 'count' : 'classification_counts'},inplace=True)
#merge the subjects with their mention in talk or how many times they were added to a collection
features_DF=pd.merge(subject_dataDF,talk_collections_DF,how='left',on='subject_id')
features_DF.fillna(0,inplace=True)

print "Saving feature data to csv..."
features_DF.to_csv('features_DF.csv',sep=',')

print "\nPickling feature data...\n"
features_DF.to_pickle('features.p')

