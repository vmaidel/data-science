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
from datetime import datetime

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

def upvoteUsers(upvotes):
   upvoters = []
   user_names = re.finditer(r'"([A-Za-z0-9_.]+)"=>', upvotes)
   if user_names is not None:
      for m in user_names:
         upvoters.append(m.group(1))
   return upvoters

def upvoteLastDate(upvotes):
   upvote_dates = []
   dates = re.finditer(r'=>"([A-Za-z0-9_.]+)"', upvotes)
   if dates is not None:
       for m in dates:
           upvote_dates.append(int(m.group(1)))
       if upvote_dates:
           return max(upvote_dates)
        
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

def removeTrailingSpace(Code):
    return Code.strip()

def preprocessingFeatureExtraction():
    try:
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

    comments_df.to_csv("comments.csv")

    #comments coded by a human coder
    coded_comments_df = pd.read_csv("SplitCodedComments.csv")

    coded_comments_df['Code'] = coded_comments_df['Code'].apply(removeTrailingSpace)
    #take out the null ones
    coded_comments_df=coded_comments_df.loc[(~coded_comments_df['Code'].isnull()),:]
    #collapsing categories
    coded_comments_df.loc[coded_comments_df['Code']=='Practice Proxy','Code']='Share/Discuss Findings'
    coded_comments_df.loc[coded_comments_df['Code']=='Informative','Code']='Share Resources'
    coded_comments_df.loc[coded_comments_df['Code']=='Conversational','Code']='Other'

    #merge with the comments that have the codes generated by SU people
    comments_df = pd.merge(comments_df, coded_comments_df, how='inner',left_on='id',right_on='comment_id')

    print "Extracting features..."
    print comments_df.shape
    comments_df['upvote_users']=comments_df['upvotes'].apply(upvoteUsers)
    comments_df['last_upvote_date']=comments_df['upvotes'].apply(upvoteLastDate)
    comments_df['number_of_upvoters']=comments_df['upvote_users'].apply(countUpvoters)
    comments_df['contains_image']=comments_df['split_body'].apply(identifyImage)
    comments_df['contains_link']=comments_df['split_body'].apply(identifyLink)
    comments_df['contains_question']=comments_df['split_body'].apply(identifyQuestion)
    comments_df['contains_hashtag']=comments_df['split_body'].apply(identifyHashtag)
    comments_df['polarity']=comments_df['split_body'].apply(identifyPolarity)
    comments_df['subjectivity']=comments_df['split_body'].apply(identifySubjectivity)
    comments_df['word_count']=comments_df['split_body'].apply(extractBodyLength)

    #reply_id is the id of the comment that the current comment (with id) is replying to, so we need to keep just the id, to know if it had a question or not
    #the groupby inside the merge is for the purpose of eliminating duplicate values of reply_id (there may be more than one if there is more than one reply to the same comment)
    response_to_question_df=pd.merge(comments_df.loc[:,['id','contains_question','contains_image']],comments_df.loc[:,['reply_id']].groupby('reply_id').first().reset_index(),left_on = 'id', right_on = 'reply_id')
    response_to_question_df = response_to_question_df.drop('id', 1)
    response_to_question_df.columns=['is_response_to_question','is_reponse_to_image','response_id']
    comments_df=pd.merge(comments_df,response_to_question_df,left_on='reply_id',right_on='response_id',how='left')

    #create a new column that contains the ids (or bodies) of the comments that replied to that particular comment.
    for index, row in comments_df.iterrows():
        comments_df.loc[index,'all_that_replied']=str(comments_df.loc[comments_df['reply_id']==row['id'],'id'].tolist())
        comments_df.loc[index,'split_body_all_that_replied']=str(comments_df.loc[comments_df['reply_id']==row['id'],'body'].tolist())

    #identify words like "helpful","thank you", "helps"
    comments_df['gratitude_in_reply']=comments_df['split_body_all_that_replied'].apply(identifyWordsInReply)
    comments_df['is_helpful'] = np.where(comments_df['number_of_upvoters']>0, 1, 0)

    #merge with the discussion table
    comments_discussion_df=pd.merge(comments_df,discussions_df.loc[:,['id','title','users_count','comments_count','last_comment_created_at']],left_on='discussion_id', right_on='id',how='left')
    comments_discussion_df.loc[:,'is_helpful']=comments_discussion_df.loc[:,'is_helpful'].astype(int)

    #the index will be different now, doing this only to see how many helpful and not helpful comments there are for each project
    #comments_discussion_df = comments_discussion_df.sort(['section','discussion_id','created_at'], ascending=[True,True,True])
    #use cumcount to get the order of the comment in a project/board/discussion
    comments_discussion_df['comment_order']=comments_discussion_df.groupby(['section','board_id','discussion_id']).cumcount()

    print "Comments shape:"
    print comments_discussion_df.shape

    users = pd.read_csv("helpful-users.csv")

    comments_discussion_df = pd.merge(comments_discussion_df,users.loc[:,['user_id','signup_date','classifications_count']], on='user_id',how='left')
    comments_discussion_df['signup_date']=pd.to_datetime(comments_discussion_df['signup_date'])
    comments_discussion_df['created_at']=pd.to_datetime(comments_discussion_df['created_at'])
    comments_discussion_df['user_seniority']=comments_discussion_df['created_at']-comments_discussion_df['signup_date']
    #convert to days
    comments_discussion_df['user_seniority']=comments_discussion_df['user_seniority'].astype('timedelta64[D]')

    #counting how many comments each user has at the time of calculation, not at the time of when the comment was written
    user_comment_counts_df = comments_discussion_df.groupby('user_id')['user_id'].agg(['count']).reset_index()
    user_comment_counts_df.columns = ['user_id','user_comment_count']
    comments_discussion_df = pd.merge(comments_discussion_df, user_comment_counts_df,on='user_id',how='left')
    dummy_comments_df = pd.get_dummies(comments_discussion_df.loc[:,['id_x','Code']],prefix='dummy')
    print dummy_comments_df.head()
    comments_discussion_df = pd.merge(comments_discussion_df,dummy_comments_df,on='id_x')

    comments_discussion_df.to_csv('comments_type_discussion.csv',sep=',',index = False)
    
    X = comments_discussion_df.loc[:,['contains_image','contains_link','contains_question','contains_hashtag','is_response_to_question','is_response_to_image','gratitude_in_reply','users_count','comments_count','polarity','subjectivity','comment_order','word_count','classifications_count','user_seniority','user_comment_count','body','id_x']]
    X.fillna({'is_response_to_question':0},inplace=True)
    X.fillna({'is_response_to_image':0},inplace=True)
    X.fillna({'user_seniority':0},inplace=True)
    X.fillna({'classifications_count':0},inplace=True)
    X.fillna({'user_comment_count':0},inplace=True)
    
    X.loc[:,['users_count','comments_count']]=X.loc[:,['users_count','comments_count']].astype(float)
    X.loc[:,'contains_image']= X.loc[:,'contains_image'].astype(int)
    X.loc[:,'gratitude_in_reply']=X.loc[:,'gratitude_in_reply'].astype(int)
    X.loc[:,'is_response_to_question']=X.loc[:,'is_response_to_question'].astype(int)
    X.loc[:,'is_response_to_image']=X.loc[:,'is_response_to_image'].astype(int)
    X.loc[:,'contains_link']= X.loc[:,'contains_link'].astype(int)
    X.loc[:,'contains_question']=X.loc[:,'contains_question'].astype(int)
    X.loc[:,'contains_hashtag']=X.loc[:,'contains_hashtag'].astype(int)
    X.loc[:,'comment_order']=X.loc[:,'comment_order'].astype(int)
    X.loc[:,'word_count']=X.loc[:,'word_count'].astype(int)
    X.loc[:,['polarity','subjectivity']]=X.loc[:,['polarity','subjectivity']].astype(float)
    X.loc[:,'user_seniority']=X.loc[:,'user_seniority'].astype(int)
    X.loc[:,'classifications_count']=X.loc[:,'classifications_count'].astype(int)
    
    print "Done..."
    
    print "Pickling comments_discussion_df, X and y_actual..."
    comments_discussion_df.to_pickle('comments_type_discussion.p')
    X.to_pickle('X_type.p')

#applying Gradient Boosting Classifier
def helpfulModelingPipelineGBC(code_to_predict):
   #load the pickles
   print "Loading pickle..."
   X=pd.read_pickle('X_type.p')
   comments_discussion_df=pd.read_pickle('comments_type_discussion.p')

   #assign the target (session length) to y and convert to int
   y_actual = pd.DataFrame(comments_discussion_df.iloc[:,code_to_predict], columns=[comments_discussion_df.columns[code_to_predict]]).astype(int)

   print y_actual.head()

   print "X head without the body and the comment_id:"
   print X.iloc[:,0:len(X.columns)-2].head()

   X_train, X_test, y_actual_train, y_actual_test = train_test_split(X, y_actual.iloc[:,0].values, test_size=0.15, random_state=0)

   print type(y_actual_train)
   selection = SelectKBest(f_classif,k=15)

   X_features = selection.fit_transform(X_train.iloc[:,0:len(X.columns)-2], y_actual_train)

   gbc = GradientBoostingClassifier(n_estimators=200)

   print np.unique(X_train.iloc[:,5:6])

   #Create a pipeline of feature selection and gradient boosting classifier
   pipeline = Pipeline([('feature_selection',selection),('gbc',gbc)])

   param_grid = dict(feature_selection__k=[11,12,14,16],
                     gbc__n_estimators = [100,150,200],
                     gbc__max_depth = [60,70,80],
                     gbc__min_samples_split = [2,3,4],
                     gbc__min_samples_leaf = [2,3,4])

   grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='recall',cv=15,verbose=10,n_jobs=15)
   grid_search.fit(X_train.iloc[:,0:len(X_train.columns)-2], y_actual_train)
   print(grid_search.best_estimator_)
   print "Just the selected columns:"+str(X.iloc[:,0:len(X.columns)-2].columns[pipeline.named_steps['feature_selection'].get_support()])
   y_actual.to_pickle('y_actual_type.p')
   pickle.dump(grid_search.best_estimator_, open( str(code_to_predict)+"_gbc_best_estimator_type.p", "wb" ) )

#random forest classifier   
def helpfulModelingPipelineRFC(code_to_predict):
   print "Loading pickles..."
   #comments_discussion_df=pd.read_pickle('comments_discussion.p')
   X=pd.read_pickle('X_type.p')

   comments_discussion_df= pd.read_pickle('comments_type_discussion.p')

   #assign the target (session length) to y and convert to int
   y_actual = pd.DataFrame(comments_discussion_df.iloc[:,code_to_predict], columns=[comments_discussion_df.columns[code_to_predict]]).astype(int)

   print y_actual.head()

   X_train, X_test, y_actual_train, y_actual_test = train_test_split(X, y_actual.iloc[:,0].values, test_size=0.15, random_state=0)

   #pca = PCA(n_components=1)
   
   #use only SelectKBest to select features
   selection = SelectKBest(f_classif,k=15)

   X_features = selection.fit(X_train.iloc[:,0:len(X.columns)-2], y_actual_train).transform(X_train.iloc[:,0:len(X_train.columns)-2])

   rfc = RandomForestClassifier(criterion='entropy')

   # Do grid search over k, n_components and C:
   pipeline = Pipeline([('feature_selection', selection), ('rfc', rfc)])

   param_grid = dict(feature_selection__k=[7,10,11,13,15],
                     rfc__n_estimators=[800,900,1000,1100],
                     rfc__max_depth = [40,100,150],
                     rfc__min_samples_split = [1,2,3],
                     rfc__min_samples_leaf = [1,2,3])

   grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='recall', cv=15 ,verbose=10,n_jobs=15)
   grid_search.fit(X_train.iloc[:,0:len(X_train.columns)-2], y_actual_train)

   print(grid_search.best_estimator_)
   #print "All columns:"+str(X.columns)
   #print "Just the selected columns:"+str(X.columns[pipeline.named_steps['selection'].get_support()])
   y_actual.to_pickle('y_actual_type.p')
   pickle.dump(grid_search.best_estimator_, open( str(code_to_predict)+"_rfc_best_estimator_type.p", "wb" ) )


def helpfulPrediction(y_actual,X,grid_search_best,model_name):
   #use grid_search.best_estimator_ (the best parameters found) to predict
   X_train, X_test, y_actual_train, y_actual_test = train_test_split(X, y_actual.iloc[:,0], test_size=0.15, random_state=0)
   y_true, y_pred = y_actual_test, grid_search_best.predict(X_test.iloc[:,0:len(X.columns)-2])
   outfile = open(model_name+'_confusion_matrix.txt','w')
   print >> outfile, "Trying to predict: "+str(y_actual.columns[0])
   print >> outfile, y_actual.stack().value_counts()
   print >> outfile, grid_search_best
   print >> outfile, confusion_matrix(y_true, y_pred)
   print >> outfile, "Training precision score: "+str(precision_score(y_actual_train, grid_search_best.predict(X_train.iloc[:,0:len(X.columns)-2])))
   print >> outfile, "Testing precision score: "+str(precision_score(y_true,y_pred))
   outfile.close()

   pd.DataFrame({'comment_id': X_test['id_x'].values,'users_count':X_test['users_count'].values,'comments_count':X_test['comments_count'].values,'contains_image':X_test['contains_image'].values,'thanking_in_reply':X_test['gratitude_in_reply'].values,'is_response_to_question':X_test['is_response_to_question'].values,'is_response_to_image':X_test['is_response_to_image'].values,'contains_link':X_test['contains_link'].values,'contains_question':X_test['contains_question'].values,'contains_hashtag':X_test['contains_hashtag'].values,'comment_order':X_test['comment_order'].values,'word_count':X_test['word_count'].values,'polarity':X_test['polarity'].values,'subjectivity':X_test['subjectivity'].values,'body': X_test['body'].values,'y_true':y_true,'y_pred':y_pred}).to_csv(model_name+"_pred_true.csv", index=False)
   
reload(sys)  
sys.setdefaultencoding('utf8')

#features extraction from the text and comment fields
#preprocessingFeatureExtraction()

X=pd.read_pickle('X_type.p')

#instantiate the model and perform the learning on all the dummy columns
for i in range(51,57):
    helpfulModelingPipelineGBC(i)
    #helpfulModelingPipelineRFC(i)
    print "predict column number:"+str(i)
    y_actual=pd.read_pickle('y_actual_type.p')
    helpfulPrediction(y_actual,X,pickle.load( open( str(i)+"_gbc_best_estimator_type.p", "rb" ) ),str(i)+"_GBC")
    #helpfulPrediction(y_actual,X,pickle.load( open( str(i)+"_rfc_best_estimator_type.p", "rb" ) ),str(i)+"_RFC")

#y_actual=pd.read_pickle('y_actual_type.p')
#helpfulPrediction(y_actual,X,pickle.load( open("52_rfc_best_estimator.p", "rb" ) ),"52_RFC")

#apply the model on the testing set
#helpfulPrediction(y_actual,X,pickle.load( open( "svm_best_estimator.p", "rb" ) ),"SVM")
#predict based on the Logistic Regression model
#helpfulPrediction(y_actual,X,pickle.load( open( "lr_best_estimator.p", "rb" ) ),"LR")
#predict based on the Random Forest Classification model 
#helpfulPrediction(y_actual,X,pickle.load( open( "rfc_best_estimator.p", "rb" ) ),"RFC")
#predict based on the Gradient Boosting Classification model 
#helpfulPrediction(y_actual,X,pickle.load( open( "gbc_best_estimator_type.p", "rb" ) ),"GBC")



