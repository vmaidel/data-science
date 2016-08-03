#!/usr/bin/env python
__author__ = 'vmaidel'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import datetime
import psycopg2
import re
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,f_regression, f_classif
import calendar

from sklearn.decomposition import PCA
import sklearn
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn_pandas import DataFrameMapper
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import GradientBoostingClassifier
import time
import itertools
from collections import defaultdict

def sort_list(the_list):
    the_list=the_list[1:-1].split(',')
    return sorted(the_list)

def get_num_of_upvotes(the_list):
    return len(the_list)

def combinations(the_list):
    return list(itertools.combinations(the_list,2))
def find_tuples(the_tuples,to_go_over,list_of_occurence):
    for idx, tup in enumerate(the_tuples):
        if tup in to_go_over:
            list_of_occurence[idx]+=1
    return list_of_occurence
                
def count_occurences(the_list):
    global occurence_count
    upvote_users = the_list[0]
    occurences = the_list[1]
    for idx, el in enumerate(upvote_users):
        if el not in occurence_count:
            occurence_count[el]=occurences[idx]

dict_of_occurence = dict()
comments_df = pd.read_csv("comments_with_features.csv")
comments_df['upvote_users'] = comments_df['upvote_users'].apply(sort_list)
comments_df['number_of_upvotes']=comments_df['upvote_users'].apply(get_num_of_upvotes)
comments_df['upvote_users']=comments_df['upvote_users'].apply(combinations)
#get only the comments that had more than one upvote
comments_df=comments_df[comments_df.number_of_upvotes>1]
comments_df['occurences']=""
comments_df['occurences'].astype(object)
for index1,row1 in comments_df.iterrows():
    #populate the list_of_occurence with 0s for each element in the row1['upvote_users'] cell
    list_of_occurence=[0]*len(row1['upvote_users'])
    #iterate over the dataframe over all the rows (row2) and count how many of the tuples match to the tuples in row1
    for index2,row2 in comments_df.iterrows():
        list_of_occurence=find_tuples(row1['upvote_users'],row2['upvote_users'],list_of_occurence)
        #find_tuples(row1['upvote_users'],row2['upvote_users'])
    comments_df.set_value(index1,'occurences',list_of_occurence)

occurence_count = dict()
comments_df[['upvote_users','occurences']].apply(count_occurences,axis=1)
print occurence_count
#print comments_df[['upvote_users','number_of_upvotes']]
#print comments_df.head()
counter = {}
counter['single']=0
counter['multiple']=0
for key, value in occurence_count.iteritems():
    if value>1:
        counter['multiple'] +=1
    else:
        counter['single'] +=1
print "Percentage of upvoter pairs co-occuring more than once:"
print counter['multiple']/float(counter['single']+counter['multiple'])


comments_df.to_csv("co_occurrences.csv")

#for i in range(0,max(comments_df['number_of_upvotes'])):
#    comments_df[comments_df.number_of_upvotes==i].groupby('upvote_users')['upvote_users'].count()
