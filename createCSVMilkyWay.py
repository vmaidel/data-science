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
    return subject_json.values()[0]['Image']

def extractLocation(subject_json):
    return subject_json.values()[0]['#Location']

def extractID(subject_json):
    return subject_json.values()[0]['ID']

def extractStartedAt(metadata_json):
    return metadata_json['started_at']

def extractFinishedAt(metadata_json):
    return metadata_json['finished_at']


def expandMilkyWayData():
    # read the csv files
    print "Reading classifications csv file for The Milky Way Project..."
    #change the name of the file if needed:
    classifications_df=pd.read_csv("blink_18_feb_2016.csv")
    
    #apply a json.loads function on the whole annotations column
    classifications_df['annotation_json']=classifications_df['annotations'].map(lambda x: json.loads(x))
    
    #extract the elements from the annotation json
    for index, row in classifications_df.iterrows():
        for i in row['annotation_json']:
            if type(i['value']) is unicode:
                #create columns with task names and assign content to the appropriate row
                classifications_df.loc[index,i['task']]=i['value']
            elif type(i['value']) is list:
                mark = 0
                circle = 0
                box = 0
                #create a column for each star marking, their coordinate value and other data
                for star in i['value']:
                    if str(star['tool_label']).split(' ', 1)[0]=='Mark':
                        mark = mark+1
                        classifications_df.loc[index,"T1_Mark"+str(mark)+"_x"]=str(star['x'])
                        classifications_df.loc[index,"T1_Mark"+str(mark)+"_y"]=str(star['y'])
                        classifications_df.loc[index,"T1_Mark"+str(mark)+"_tool"]=str(star['tool'])
                        classifications_df.loc[index,"T1_Mark"+str(mark)+"_frame"]=str(star['frame'])
                        classifications_df.loc[index,"T1_Mark"+str(mark)+"_details"]=str(star['details'])
                        classifications_df.loc[index,"T1_Mark"+str(mark)+"_tool_label"]=str(star['tool_label'])
                    elif str(star['tool_label']).split(' ', 1)[0]=='Circle':
                        circle = circle+1
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_r"]=str(star['r'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_x"]=str(star['x'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_y"]=str(star['y'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_tool"]=str(star['tool'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_angle"]=str(star['angle'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_frame"]=str(star['frame'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_details"]=str(star['details'])
                        classifications_df.loc[index,"T1_Circle"+str(circle)+"_tool_label"]=str(star['tool_label'])
                    elif str(star['tool_label']).split(' ', 1)[0]=='Put':
                        box = box+1
                        classifications_df.loc[index,"T1_Box"+str(box)+"_x"]=str(star['x'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_y"]=str(star['y'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_tool"]=str(star['tool'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_frame"]=str(star['frame'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_width"]=str(star['width'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_height"]=str(star['height'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_details"]=str(star['details'])
                        classifications_df.loc[index,"T1_Box"+str(box)+"_tool_label"]=str(star['tool_label'])

    #apply a json.loads function on the subject_data column
    classifications_df['subject_json']=classifications_df['subject_data'].map(lambda x: json.loads(x))
    
    #extract the desired elements from the subject data json
    classifications_df['subject_id']=classifications_df['subject_json'].apply(extractSubject_id)
    classifications_df['when_retired']=classifications_df['subject_json'].apply(extractRetired)
    classifications_df['filename']=classifications_df['subject_json'].apply(extractFilename)
    classifications_df['ID']=classifications_df['subject_json'].apply(extractID)
    classifications_df['Location']=classifications_df['subject_json'].apply(extractLocation)

    #apply a json.loads function on the subject_data column
    classifications_df['metadata_json']=classifications_df['metadata'].map(lambda x: json.loads(x))

    classifications_df['started_at']=classifications_df['metadata_json'].apply(extractStartedAt)
    classifications_df['finished_at']=classifications_df['metadata_json'].apply(extractFinishedAt)


    #delete the unwanted columns
    classifications_df.drop(classifications_df.columns[[12,336, 342]], axis=1, inplace=True)
    
    print "The dataframe to be exported as csv - will not show all columns:"
    print classifications_df.head(20)
    print classifications_df.columns

    #save to csv
    classifications_df.to_csv('expandedMilkyWayProject.csv',sep=',',index = False,encoding='utf-8')


expandMilkyWayData()

