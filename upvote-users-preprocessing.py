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


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def upvoteDict(all_upvotes):
   upvoters = {}
   for upvote in all_upvotes.split(","):
       user_name = re.findall(r'"([A-Za-z0-9_.]+)"=>',upvote)
       date = re.findall(r'=>"([A-Za-z0-9_.]+)"', upvote)
       if user_name and date:
           upvoters[user_name[0]]=date[0]
   return upvoters

def upvoteUsers(upvotes):
   upvoters = []
   user_names = re.finditer(r'"([A-Za-z0-9_.]+)"=>', upvotes)
   if user_names is not None:
      for m in user_names:
         upvoters.append(m.group(1))
   return upvoters

def upvoteFirstDate(upvotes):
   upvote_dates = []
   dates = re.finditer(r'=>"([A-Za-z0-9_.]+)"', upvotes)
   if dates is not None:
       for m in dates:
           upvote_dates.append(int(m.group(1)))
       if upvote_dates:
           return min(upvote_dates)
        
def countUpvoters(upvoters):
   return len(upvoters)

def identifyImage(text_body):
   if re.search(r'!\[.*\]\(.*\)',text_body):
      return 1
   else:
      return 0

def identifyLink(text_body):
   if re.search(r'[^!]\[.*\]\(.*\)',text_body):
      return 1
   else:
      return 0

def identifyQuestion(text_body):
   if re.search(r'[ ?]*[?][ ?]',text_body):
      return 1
   else:
      return 0

def identifyWordsInReply(text_body):
   if re.search(r'[Tt]hank you|[Hh]elpful|[Hh]elps|[Tt]hanks',text_body):
      return 1
   else:
      return 0

def identifyHashtag(text_body):
   if re.search(r'#\w*',text_body):
      return 1
   else:
      return 0

def identifyPolarity(text_body):
   sen  = TextBlob(text_body)
   return sen.sentiment[0]

def identifySubjectivity(text_body):
   sen  = TextBlob(text_body)
   return sen.sentiment[1]

def extractBodyLength(text_body):
    words = 0
    for wordcount in text_body.split(" "):
        words += 1
    return words

def convertUpvoteDate(upvote_date):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(upvote_date)))

def toSec(diffs_to_convert):
    if pd.notnull(diffs_to_convert):
        #return calendar.timegm(time.strptime(str(diffs_to_convert), '%Y-%m-%d %H:%M:%S'))
        return (diffs_to_convert-datetime.datetime(1970,1,1)).total_seconds()
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

def removeTrailingSpace(Code):
    return Code.strip()

def roundToX(x, base=5):
    if not np.isnan(x):
        return int(base * round(float(x)/base))

def getDaysSince(upvote_date):
    type(datetime.datetime.strptime("2015-11-21 00:00:00", "%Y-%m-%d %H:%M:%S"))
    #print datetime.datetime.strptime("2015-11-21 00:00:00", "%Y-%m-%d %H:%M:%S")
    #print upvote_date
    return (datetime.datetime.strptime(upvote_date, "%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime("2015-11-21 00:00:00", "%Y-%m-%d %H:%M:%S")).days

def preprocessingFeatureExtraction(comments_df,discussions_df):

    print "Extracting features..."
    print comments_df.shape
    comments_df['upvotes_dict']=comments_df['upvotes'].apply(upvoteDict)
    comments_df['upvote_users']=comments_df['upvotes'].apply(upvoteUsers)
    comments_df['first_upvote_date']=comments_df['upvotes'].apply(upvoteFirstDate)
    comments_df['number_of_upvoters']=comments_df['upvote_users'].apply(countUpvoters)
    comments_df['contains_image']=comments_df['body'].apply(identifyImage)
    comments_df['contains_link']=comments_df['body'].apply(identifyLink)
    comments_df['contains_question']=comments_df['body'].apply(identifyQuestion)
    comments_df['contains_hashtag']=comments_df['body'].apply(identifyHashtag)
    comments_df['polarity']=comments_df['body'].apply(identifyPolarity)
    comments_df['subjectivity']=comments_df['body'].apply(identifySubjectivity)
    comments_df['word_count']=comments_df['body'].apply(extractBodyLength)

    #save a separate file for upvote co-occurance. 
    comments_df.loc[comments_df['first_upvote_date']>=1447912800,:].to_csv("comments_with_features.csv",index=False)
    
    #create a dataframe that contains a row per each upvote, if there is more than one upvote for a comment, the row for that comment will be duplicated
    upvotes_df = pd.DataFrame(columns=['id','upvote_name','upvote_date'])
    no_dups_df = comments_df.loc[:,['id','upvotes']].drop_duplicates()
    no_dups_df['upvotes_dict']=no_dups_df['upvotes'].apply(upvoteDict)
    for index, row in no_dups_df.iterrows():
        for upvoter, date in row['upvotes_dict'].iteritems():
            upvotes_df=upvotes_df.append(pd.DataFrame([[row['id'],upvoter,date]],columns=['id','upvote_name','upvote_date']))
    upvotes_df['upvote_date']=upvotes_df['upvote_date'].apply(convertUpvoteDate)
    helpful_comments_over_time_df = upvotes_df.copy(deep=True)
    helpful_comments_over_time_df['days_since_deployment']=helpful_comments_over_time_df['upvote_date'].apply(getDaysSince)
    helpful_comments_over_time_df=helpful_comments_over_time_df.groupby('days_since_deployment')['days_since_deployment'].agg(['count']).reset_index()
    X=helpful_comments_over_time_df.loc[helpful_comments_over_time_df.days_since_deployment>=0,'days_since_deployment'].values.tolist()
    Y=helpful_comments_over_time_df.loc[helpful_comments_over_time_df.days_since_deployment>=0,'count'].values.tolist()

    plt.scatter(X, Y)

    plt.savefig('upvotes-over-time.png')
    plt.clf()
    plt.close()

    #limit the comments only to the wildcam-gorongosa board or the zooniverse board. Remember to change this accordingly!!!
    comments_df = comments_df.loc[comments_df.section=='project-593',:]

    coded_comments_df = pd.read_csv("SplitCodedComments.csv",quotechar='"')
    coded_comments_df['Code'] = coded_comments_df['Code'].apply(removeTrailingSpace)
    #take out the null ones
    coded_comments_df=coded_comments_df.loc[~coded_comments_df['Code'].isnull(),:]

    #merge with the comments that have the codes generated by SU people
    comments_df = pd.merge(comments_df, coded_comments_df[['comment_id','Code']], how='inner',left_on='id',right_on='comment_id')

    #reply_id is the id of the comment that the current comment (with id) is replying to, so we need to keep just the id, to know if it had a question or not
    #the groupby inside the merge is for the purpose of eliminating duplicate values of reply_id (there may be more than one if there is more than one reply to the same comment)
    response_to_question_df=pd.merge(comments_df.loc[:,['id','contains_question','contains_image']].groupby(['id','contains_question','contains_image']).first().reset_index(),comments_df.loc[:,['reply_id']].groupby('reply_id').first().reset_index(),left_on = 'id', right_on = 'reply_id')
    response_to_question_df = response_to_question_df.drop('id', 1)
    response_to_question_df.columns=['is_response_to_question','is_reponse_to_image','response_id']
    comments_df=pd.merge(comments_df,response_to_question_df,left_on='reply_id',right_on='response_id',how='left')

    #create a new column that contains the ids (or bodies) of the comments that replied to that particular comment.
    for index, row in comments_df.iterrows():
        comments_df.loc[index,'all_that_replied']=str(comments_df.loc[comments_df['reply_id']==row['id'],'id'].tolist())
        comments_df.loc[index,'body_all_that_replied']=str(comments_df.loc[comments_df['reply_id']==row['id'],'body'].tolist())

    #identify words like "helpful","thank you", "helps"
    comments_df['gratitude_in_reply']=comments_df['body_all_that_replied'].apply(identifyWordsInReply)
    comments_df['is_helpful'] = np.where(comments_df['number_of_upvoters']>0, 1, 0)

    #merge with the discussion table
    comments_discussion_df=pd.merge(comments_df,discussions_df.loc[:,['discussion_id','title','users_count','comments_count','last_comment_created_at']],on='discussion_id',how='left')
    comments_discussion_df.loc[:,'is_helpful']=comments_discussion_df.loc[:,'is_helpful'].astype(int)
    print comments_discussion_df.columns
    #comments_discussion_df.rename(columns = {'id':'comment_id'},inplace=True)
    comments_discussion_df.loc[:,'comment_id']=comments_discussion_df.loc[:,'comment_id'].astype(int)

    #the index will be different now, doing this only to see how many helpful and not helpful comments there are for each project
    #comments_discussion_df = comments_discussion_df.sort(['section','discussion_id','created_at'], ascending=[True,True,True])
    #use cumcount to get the order of the comment in a project/board/discussion
    comments_discussion_df['comment_order']=comments_discussion_df.groupby(['section','board_id','discussion_id']).cumcount()

    #merge the comments with the idividual upvotes per row
    comments_discussion_df = pd.merge(comments_discussion_df, upvotes_df, how='left',left_on='comment_id',right_on='id')


    print "Comments shape:"
    print comments_discussion_df.shape

    users_df = pd.read_csv("helpful-users.csv")

    users_df.rename(columns = {'user_login':'upvoter_user_login'},inplace=True)
    users_df.rename(columns = {'signup_date':'upvoter_signup_date'},inplace=True)
    users_df.rename(columns = {'classifications_count':'upvoter_classifications_count'},inplace=True)

    comments_discussion_df = pd.merge(comments_discussion_df,users_df.loc[:,['upvoter_user_login','upvoter_signup_date','upvoter_classifications_count']],left_on='upvote_name',right_on='upvoter_user_login',how='left')
    comments_discussion_df['upvoter_signup_date']=pd.to_datetime(comments_discussion_df['upvoter_signup_date'])
    comments_discussion_df['created_at']=pd.to_datetime(comments_discussion_df['created_at'])
    #,format='%Y-%b-%d %H:%M:%S.%f')
    comments_discussion_df['upvote_date']=pd.to_datetime(comments_discussion_df['upvote_date'])

    #user_seniority at the time of upvoting
    comments_discussion_df['upvoter_user_seniority']=comments_discussion_df['upvote_date']-comments_discussion_df['upvoter_signup_date']
    #convert to days
    comments_discussion_df['upvoter_user_seniority']=comments_discussion_df['upvoter_user_seniority'].astype('timedelta64[D]')

    users_df.rename(columns = {'upvoter_user_login':'commenter_user_login'},inplace=True)
    users_df.rename(columns = {'upvoter_signup_date':'commenter_signup_date'},inplace=True)
    users_df.rename(columns = {'upvoter_classifications_count':'commenter_classifications_count'},inplace=True)

    #merge with the users file again, this time to get user data for the commenters 
    comments_discussion_df = pd.merge(comments_discussion_df,users_df.loc[:,['commenter_user_login','commenter_signup_date','commenter_classifications_count']],left_on='user_login',right_on='commenter_user_login',how='left')
    
    comments_discussion_df['commenter_signup_date']=pd.to_datetime(comments_discussion_df['commenter_signup_date'])

    #seniority at the time of comment creation
    comments_discussion_df['commenter_user_seniority']=comments_discussion_df['created_at']-comments_discussion_df['commenter_signup_date']
     #convert to days
    comments_discussion_df['commenter_user_seniority']=comments_discussion_df['commenter_user_seniority'].astype('timedelta64[D]')

    #counting how many comments each user has at the time of calculation, not at the time of when the comment was written
    #user_comment_counts_df = comments_discussion_df.groupby('user_id')['user_id'].agg(['count']).reset_index()
    #user_comment_counts_df.columns = ['user_id','user_comment_count']
    #comments_discussion_df = pd.merge(comments_discussion_df, user_comment_counts_df,on='user_id',how='left')

    #round seniority to the nearest 5
    comments_discussion_df['commenter_user_seniority']= comments_discussion_df['commenter_user_seniority'].apply(roundToX)
    comments_discussion_df['upvoter_user_seniority']= comments_discussion_df['upvoter_user_seniority'].apply(roundToX)
    #get unique comments in order to be able to count how many comments each commenter got
    forCommentCount = comments_discussion_df.loc[:,['commenter_user_login','id','commenter_user_seniority']].drop_duplicates()
    comment_count_df=forCommentCount.groupby(['commenter_user_seniority'])['commenter_user_seniority'].agg(['count']).reset_index()
    #comment_count_df=pd.merge(forCommentCount.loc[:,['commenter_user_login','commenter_user_seniority']],forCommentCount.groupby(['commenter_user_login','commenter_user_seniority']).agg(['count']).reset_index(),on=['commenter_user_login'])
    #comment_count_df.sort_values(by=['commenter_user_login','commenter_user_seniority'],inplace=True)
    print comment_count_df.head(20)
    comment_count_df.to_csv("comment_count.csv")

    Y = comment_count_df['count'].values.tolist()
    X = comment_count_df['commenter_user_seniority'].values.tolist()

    plt.scatter(X, Y)

    plt.savefig('comment-counts-scatter.png')
    plt.clf()
    plt.close()
  
    forUpvotesCount = comments_discussion_df.loc[:,['upvoter_user_seniority']]
    upvotes_count_df=forUpvotesCount.groupby(['upvoter_user_seniority'])['upvoter_user_seniority'].agg(['count']).reset_index()
    #upvotes_count_df.sort_values(by=['upvoter_user_login','upvoter_user_seniority'],inplace=True)
    upvotes_count_df.to_csv("upvotes_count.csv")
    print upvotes_count_df.head(20)

    Y = upvotes_count_df['count'].values.tolist()
    X = upvotes_count_df['upvoter_user_seniority'].values.tolist()

    plt.scatter(X, Y)

    plt.savefig('upvote-counts-scatter.png')
    plt.clf()
    plt.close()
    
    comments_discussion_df.to_csv('comments_discussion_indv_upvotes.csv',sep=',',index = False)

def calculatingTalkSessions(comments_df):
    classifications_df = pd.read_csv("wildcam-gorongosa-classifications.csv")
    comments_df = comments_df.copy(deep=True)
    print comments_df['created_at'].head()
    #get only project gorongoza comments created and upvoted:
    comments_df = comments_df.loc[comments_df['project_id']==float("593.0"),:]
    users_df = pd.read_csv('helpful-users.csv')
    #merge with users to get the user_login field
    classifications_df=pd.merge(classifications_df, users_df, on='user_id',how='inner')[['classification_id','user_login','created_at']]
    classifications_df.columns = ['id','user_login','created_at']
    classifications_df['type']="classification"
    print "classifications:"
    print classifications_df.head()
    upvotes_df = pd.DataFrame(columns=['comment_id','upvote_name','upvote_date'])
    comments_df['upvotes_dict']=comments_df['upvotes'].apply(upvoteDict)

    #create a dataframe that contains a row per each upvote, if there is more than one upvote for a comment, the row for that comment will be duplicated
    for index, row in comments_df.iterrows():
        for upvoter, date in row['upvotes_dict'].iteritems():
            upvotes_df=upvotes_df.append(pd.DataFrame([[row['id'],upvoter,date]],columns=['comment_id','upvote_name','upvote_date']))

    upvotes_df['upvote_date']=upvotes_df['upvote_date'].apply(convertUpvoteDate)
    upvotes_df.columns=['id','user_login','created_at']
    upvotes_df['type']="comment_upvote"
    comments_df['type'] = "comment_create"
    user_activity_df = pd.concat([comments_df[['id','user_login','created_at','type']],upvotes_df,classifications_df],ignore_index=True)
    user_activity_df.sort_values(by=['user_login','created_at'],inplace=True)

    print "Calculating sessions..."
    #upvotes_and_comment_creation = upvotes_and_comment_creation.reset_index(drop=True)
    user_activity_df['created_at'] = pd.to_datetime(user_activity_df['created_at'])

    #calculate the gaps between activity events
    user_activity_df['diffs'] = user_activity_df.groupby(['user_login'])['created_at'].transform(lambda x: x.diff()) 
    #the result is returned in time since epoch, so this will convert the gaps to seconds
    user_activity_df['diffs']=user_activity_df['diffs'].apply(toSec)
    #calculate session number for a diff between events that lasted longer than 1800 seconds
    user_activity_df['session_num'] = user_activity_df['diffs'].apply(calcSessionNum)
    user_activity_df.to_csv("user_activity_df.csv",index = False)

    #get range of dates during which this user was in session
    #aggregations = {'created_at': {'min_date':'min','max_date':'max'}}
    #user_activity_range_df = user_activity_df[['user_login','session_num','created_at']].groupby(['user_login','session_num']).agg(aggregations).reset_index()
    #user_activity_range_df.columns=['user_login','session_num','max_date','min_date']
    #user_activity_range_df.to_csv("user_activity_range_df.csv")

def assignSessionsToUpvotes():
    user_activity_df = pd.read_csv("user_activity_df.csv")
    individual_comment_upvotes_df = pd.read_csv("comments_discussion_indv_upvotes.csv")
    upvotes_activity_df = user_activity_df.loc[user_activity_df["type"]=="comment_upvote",:].copy()
    upvotes_activity_df.loc[:,'created_at']=upvotes_activity_df['created_at'].apply(lambda x: x.split(".")[0])
    individual_comment_upvotes_df= individual_comment_upvotes_df.loc[individual_comment_upvotes_df['upvote_date'].notnull(),:]
    individual_comment_upvotes_df.loc[:,'upvote_date']=individual_comment_upvotes_df['upvote_date'].apply(lambda x: x.split(".")[0])

    upvotes_activity_df.rename(columns = {'user_login':'upvote_user_login','created_at':'upvote_created_at'},inplace=True)

    upvotes_and_sessions_df = pd.merge(upvotes_activity_df[['upvote_user_login','upvote_created_at','session_num']],individual_comment_upvotes_df, how = 'inner',left_on = ['upvote_created_at'],right_on = ['upvote_date']).drop_duplicates()

    print upvotes_and_sessions_df.head()

    #upvotes_and_sessions_df =  upvotes_and_sessions_df.drop_duplicates()

    upvotes_pivot = pd.pivot_table(upvotes_and_sessions_df[['session_num','Code']],index=["session_num"],columns=["Code"],aggfunc=len,fill_value=0).reset_index()

    #upvotes_pivot = pd.pivot_table(upvotes_and_sessions_df[['session_num','Code']],index=["session_num"],columns=["Code"],aggfunc=lambda x: len(x.unique()),fill_value=0).reset_index()

    unique_counts = upvotes_and_sessions_df.groupby(['session_num', 'Code'])['upvote_user_login'].unique().map(len)
    unstacked_unique_pivot = unique_counts.unstack(level='Code').fillna(0)
    print "unstacked:"
    print unstacked_unique_pivot.head()
    print "upvotes_pivot"
    print upvotes_pivot.head()
    upvotes_pivot.to_csv("upvotes_pivot.csv",index=False)
    unstacked_unique_pivot.to_csv("unstacked_unique_pivot.csv",index=False)

    labels = [ "{0} - {1}".format(i, i + 19) for i in range(0, 200, 20) ]
    upvotes_and_sessions_df['session_group'] = pd.cut(upvotes_and_sessions_df.session_num, range(0, 205, 20), right=False, labels=labels)
    upvotes_and_sessions_df['session_group']=upvotes_and_sessions_df.session_group.astype('category')
    upvotes_and_sessions_df.sort_values(by=['upvote_user_login','session_group'],inplace=True)
    upvotes_and_sessions_df.reset_index(inplace=True)
    transition_matrix = {}
    for index in range(0,len(upvotes_and_sessions_df)):
        if index>0:
            if (upvotes_and_sessions_df.ix[index,'upvote_user_login']==upvotes_and_sessions_df.ix[index-1,'upvote_user_login']) and (upvotes_and_sessions_df.ix[index,'session_group']>upvotes_and_sessions_df.ix[index-1,'session_group']):
                if not upvotes_and_sessions_df.ix[index-1,'Code'] in transition_matrix:
                    transition_matrix[upvotes_and_sessions_df.ix[index-1,'Code']]={}
                    transition_matrix[upvotes_and_sessions_df.ix[index-1,'Code']][upvotes_and_sessions_df.ix[index,'Code']]=1
                else:
                    if not upvotes_and_sessions_df.ix[index,'Code'] in transition_matrix[upvotes_and_sessions_df.ix[index-1,'Code']]:
                        transition_matrix[upvotes_and_sessions_df.ix[index-1,'Code']][upvotes_and_sessions_df.ix[index,'Code']]=1
                    else:
                        transition_matrix[upvotes_and_sessions_df.ix[index-1,'Code']][upvotes_and_sessions_df.ix[index,'Code']]+=1
                    
    print transition_matrix
    
    upvotes_and_sessions_df.to_csv("upvotes_and_sessions_df.csv",index=False)

    upvotes_first_session = upvotes_pivot.iloc[0:1,:]

    upvotes_pivot=upvotes_pivot.iloc[1:256,:]
    lst = range(1,52)
    upvotes_pivot['groupby'] = list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in lst))
    aggregations = {'session_num':{'tuple_min':'min','tuple_max':'max'},'Answer Question':{'sum':'sum'},'Ask Question':{'sum':'sum'},'Conversational':{'sum':'sum'},'Informative':{'sum':'sum'},'Other':{'sum':'sum'},'Practice Proxy':{'sum':'sum'},'Share Resources':{'sum':'sum'},'Share/Discuss Findings':{'sum':'sum'},'Technical Problem/Improving Zooniverse':{'sum':'sum'}}
    upvotes_session_range_pivot = upvotes_pivot.groupby(['groupby']).agg(aggregations).reset_index()
    upvotes_session_range_pivot.columns=['group','Conversational', 'Ask Question', 'Other','Technical Problem/Improving Zooniverse', 'Answer Question', 'session_max', 'session_min','Informative', 'Practice Proxy', 'Share/Discuss Findings', 'Share Resources']
    upvotes_session_range_pivot['session_range']=upvotes_session_range_pivot[['session_min','session_max']].apply(lambda row:(row['session_min'],row['session_max']),axis=1)
    upvotes_session_range_pivot.drop(['session_min','session_max'],inplace=True,axis=1)
    print upvotes_session_range_pivot.columns
    upvotes_first_session['group']=0
    upvotes_first_session['session_range']=1
    upvotes_session_range_pivot=pd.concat([upvotes_first_session[['group', 'session_range','Conversational','Ask Question','Other','Technical Problem/Improving Zooniverse', 'Answer Question','Informative', 'Practice Proxy', 'Share/Discuss Findings','Share Resources']],upvotes_session_range_pivot],axis=0, ignore_index=True)
    upvotes_session_range_pivot[['group', 'session_range','Conversational','Ask Question','Other','Technical Problem/Improving Zooniverse', 'Answer Question','Informative', 'Practice Proxy', 'Share/Discuss Findings','Share Resources']].to_csv("upvotes_session_range_pivot.csv",index=False)


def zooniverseWideBoardPivot():
    individual_comment_upvotes_df = pd.read_csv("comments_discussion_indv_upvotes.csv")
    #cut the seniority into bins
    bins = [0, 100, 500, 1000, 3000]
    group_names = ['Newcomer', 'Advanced', 'Oldtimer','Veteran']
    categories = pd.cut(individual_comment_upvotes_df['upvoter_user_seniority'], bins, labels=group_names)
    individual_comment_upvotes_df['categorical_seniority'] = pd.cut(individual_comment_upvotes_df['upvoter_user_seniority'], bins, labels=group_names)
    print individual_comment_upvotes_df[['upvoter_user_seniority','categorical_seniority']]
    zooniverse_pivot = pd.pivot_table(individual_comment_upvotes_df[['categorical_seniority','Code']],index=["categorical_seniority"],columns=["Code"],aggfunc=len,fill_value=0).reset_index()
    zooniverse_pivot.to_csv("zooniverse_pivot.csv")

reload(sys)  
sys.setdefaultencoding('utf8')

try:
    #connect to the database 
    conn = psycopg2.connect("dbname='talk_helpful' user='citisci' host='localhost' password='x6axiKm!'")
    print "Connected successfully!"
    boards_sql = 'SELECT * FROM boards'
    boards_df = pd.read_sql(boards_sql,conn)
    discussions_sql = 'SELECT * FROM discussions'
    discussions_df = pd.read_sql(discussions_sql,conn)
    comments_sql = 'SELECT * FROM comments'
    comments_df =  pd.read_sql(comments_sql,conn)
except psycopg2.Error as e:
    print e.pgerror
        
#features extraction from the text and comment fields
#preprocessingFeatureExtraction(comments_df,discussions_df)
#calculatingTalkSessions(comments_df)
assignSessionsToUpvotes()
#zooniverseWideBoardPivot()


    
 
