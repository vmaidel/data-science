#!/usr/bin/env python
__author__ = 'vmaidel'
import pandas as pd
import numpy as np
import json


def extractRetired(subject_json):
    if subject_json.values()[0]['retired'] is None:
        return "Not yet"
    else: return subject_json.values()[0]['retired']['created_at']

def extractSubject_id(subject_json):
    return subject_json.keys()[0]

def extractFilename(subject_json):
    return subject_json.values()[0]['Filename']

def extractCatalog(subject_json):
    return subject_json.values()[0]['Catalog']

def extractConstellation(subject_json):
    return subject_json.values()[0]['Constellation']


def version448_336():
    # read the csv files
    print "Reading classifications csv file for version 448.336..."
    classifications_df=pd.read_csv("e0a292b8-a74b-4f63-81c9-63f05b8104db.csv")
    
    #limit only to the desired workflow version
    classifications_df=classifications_df.loc[classifications_df['workflow_version']==float('448.336')]
    
    #apply a json.loads function on the whole annotations column
    classifications_df['annotation_json']=classifications_df['annotations'].map(lambda x: json.loads(x))
    
    #extract the elements from the annotation json
    for index, row in classifications_df.iterrows():
        for i in row['annotation_json']:
            if type(i['value']) is dict:
                #create a column for each coordinate value and the height and the width
                classifications_df.loc[index,i['task']+"_x"]=str(i['value']['x'])
                classifications_df.loc[index,i['task']+"_y"]=str(i['value']['y'])
                classifications_df.loc[index,i['task']+"_width"]=str(i['value']['width'])
                classifications_df.loc[index,i['task']+"_height"]=str(i['value']['height'])
            elif type(i['value']) is unicode:
                #create columns with task names and assign content to the appropriate row
                classifications_df.loc[index,i['task']]=i['value']
            elif type(i['value']) is list:
                for j in i['value']:
                    #create columns with task names and assign content to the appropriate row
                    classifications_df.loc[index,i['task']]=str(j['choice'])
 
    #apply a json.loads function on the subject_data column
    classifications_df['subject_json']=classifications_df['subject_data'].map(lambda x: json.loads(x))
    
    #extract the desired elements from the subject data json
    classifications_df['subject_id']=classifications_df['subject_json'].apply(extractSubject_id)
    classifications_df['when_retired']=classifications_df['subject_json'].apply(extractRetired)
    classifications_df['filename']=classifications_df['subject_json'].apply(extractFilename)
    
    #delete the unwanted columns
    classifications_df.drop(classifications_df.columns[[9, 10, 11, 12, 30]], axis=1, inplace=True)
    
    print "The dataframe to be exported as csv - will not show all columns:"
    print classifications_df.head(20)
    print classifications_df.columns

    #save to csv
    classifications_df.to_csv('expandedSkiesCSV448_336.csv',sep=',',index = False,encoding='utf-8')

#for version 5.6
def version5_6():
    # read the csv files
    print "Reading classifications csv file for version 5.6..."
    classifications_df=pd.read_csv("e0a292b8-a74b-4f63-81c9-63f05b8104db.csv")
    
    classifications_df=classifications_df.loc[classifications_df['workflow_version']==float('5.6')]
    #apply a json.loads function on the whole annotations column
    classifications_df['annotation_json']=classifications_df['annotations'].map(lambda x: json.loads(x))
    
    #extract the elements from the annotation json
    for index, row in classifications_df.iterrows():
        for i in row['annotation_json']:
            num = 0
            for circle in i['value']:
                num = num+1
                classifications_df.loc[index,"T1_Circle"+str(num)+"_angle"]=str(circle['angle'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_tool"]=str(circle['tool'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_r"]=str(circle['r'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_details"]=str(circle['details'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_tool_label"]=str(circle['tool_label'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_y"]=str(circle['y'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_x"]=str(circle['x'])
                classifications_df.loc[index,"T1_Circle"+str(num)+"_frame"]=str(circle['frame'])


    #apply a json.loads function on the subject_data column
    classifications_df['subject_json']=classifications_df['subject_data'].map(lambda x: json.loads(x))
    
    #extract the desired elements from the subject data json
    classifications_df['subject_id']=classifications_df['subject_json'].apply(extractSubject_id)
    classifications_df['when_retired']=classifications_df['subject_json'].apply(extractRetired)
    classifications_df['filename']=classifications_df['subject_json'].apply(extractFilename)
    classifications_df['catalog']=classifications_df['subject_json'].apply(extractCatalog)
    classifications_df['constellation']=classifications_df['subject_json'].apply(extractConstellation)

    #delete the unwanted columns
    classifications_df.drop(classifications_df.columns[[9, 10, 11, 12, 61]], axis=1, inplace=True)
    
    
    print "The dataframe to be exported as csv - will not show all columns:"
    print classifications_df.head(20)
    print classifications_df.columns


    classifications_df.to_csv('expandedSkiesCSV5_6.csv',sep=',',index = False,encoding='utf-8')

def version513_315():
    # read the csv files
    print "Reading classifications csv file for version 513.315..."
    classifications_df=pd.read_csv("e0a292b8-a74b-4f63-81c9-63f05b8104db.csv")
    
    classifications_df=classifications_df.loc[classifications_df['workflow_version']==float('513.315')]
    #apply a json.loads function on the whole annotations column
    classifications_df['annotation_json']=classifications_df['annotations'].map(lambda x: json.loads(x))


    #extract the elements from the annotation json
    for index, row in classifications_df.iterrows():
        for i in row['annotation_json']:
            if i['task']=="T3":
                for j in i['value']:
                    classifications_df.loc[index,"T3_x"]=str(j['x'])
                    classifications_df.loc[index,"T3_y"]=str(j['y'])
                    classifications_df.loc[index,"T3_rx"]=str(j['rx'])
                    classifications_df.loc[index,"T3_ry"]=str(j['ry'])

                    classifications_df.loc[index,"T3_tool"]=str(j['tool'])
                    classifications_df.loc[index,"T3_angle"]=str(j['angle'])

                    classifications_df.loc[index,"T3_frame"]=str(j['frame'])
                    classifications_df.loc[index,"T3_details"]=str(j['details'])
                    classifications_df.loc[index,"T3_tool_label"]=str(j['tool_label'])
            elif (i['task']=="T34" or i['task']=="T33" or i['task']=="T9"):
                for j in i['value']:
                    #create columns with task names and assign content to the appropriate row
                    classifications_df.loc[index,i['task']]=str(j['choice'])
            elif i['task']=="T2":
                num = 0
                for star in i['value']:
                    num = num+1
                    classifications_df.loc[index,"T2_Star"+str(num)+"_x"]=str(star['x'])
                    classifications_df.loc[index,"T2_Star"+str(num)+"_y"]=str(star['y'])
                    classifications_df.loc[index,"T2_Star"+str(num)+"_tool"]=str(star['tool'])
                    classifications_df.loc[index,"T2_Star"+str(num)+"_frame"]=str(star['frame'])
                    classifications_df.loc[index,"T2_Star"+str(num)+"_details"]=str(star['details'])
                    classifications_df.loc[index,"T2_Star"+str(num)+"_tool_label"]=str(star['tool_label'])
            else:
                classifications_df.loc[index,i['task']]=str(i['value'])

                    
    #apply a json.loads function on the subject_data column
    classifications_df['subject_json']=classifications_df['subject_data'].map(lambda x: json.loads(x))
    
    #extract the desired elements from the subject data json
    classifications_df['subject_id']=classifications_df['subject_json'].apply(extractSubject_id)
    classifications_df['when_retired']=classifications_df['subject_json'].apply(extractRetired)
    classifications_df['filename']=classifications_df['subject_json'].apply(extractFilename)
    
    #delete the unwanted columns
    classifications_df.drop(classifications_df.columns[[9,10,11,12,603]], axis=1, inplace=True)
    
    print "The dataframe to be exported as csv - will not show all columns:"
    print classifications_df.head(20)
    print classifications_df.columns

    classifications_df.to_csv('expandedSkiesCSV513_315.csv',sep=',',index = False,encoding='utf-8')
  
version448_336()
version5_6()
version513_315()

