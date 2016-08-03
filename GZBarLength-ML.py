#!/usr/bin/env python
#trying to predict session length based on image features
__author__ = 'vmaidel'
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, f_regression
from sklearn.decomposition import PCA
import sklearn
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import linear_model
from sklearn import decomposition
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


#SUBJECT level analysis - trying to predict whether a subject get attention, based on the image features
def subject_level_classification(subject_levelDF):
   #if the subject had either a mention or was added to a collection, its "attention" will be marked as 1, otherwise 0
   #subject_levelDF['attention'] = np.where(subject_levelDF['talk_mentions_counts']+subject_levelDF['collections_counts']>0, 1, 0)
   subject_levelDF['attention'] = np.where(subject_levelDF['collections_counts']>0, 1, 0)

   print subject_levelDF.groupby(['attention']).size()

   #assign to X only the features that can be converted to float 
   X = subject_levelDF.iloc[:,11:104]
   #turn blanks into 0 in the PHOTOZ field, because otherwise it won't convert to float
   X.loc[X['PHOTOZ']=="",'PHOTOZ']=0
   X.drop('Image_Iterations', axis=1, inplace=True)
   X.drop('Image_Depth', axis=1, inplace=True)
   X.drop('Image_Channeldepth_red', axis=1, inplace=True)
   X.drop('Image_Channeldepth_green', axis=1, inplace=True)
   X.drop('Image_Channeldepth_blue', axis=1, inplace=True)
   X.drop('Image_Properties_signature', axis=1, inplace=True)
   #X.drop('Image_Resolution',axis=1,inplace=True)
   X.astype(float)
   #assign the target (attention) to y and convert to int
   y_actual = subject_levelDF.iloc[:,109:110].astype(int)

   print X.shape
   
   X_scaled = preprocessing.scale(X)

   #split the data set for train and test sets
   X_train, X_test, y_actual_train, y_actual_test = train_test_split(X_scaled, y_actual, test_size=0.15, random_state=0)

   pca = PCA(n_components=1)

   selection = SelectKBest(k=1)

   combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

   X_features = pca.fit(X_train).transform(X_train)

   SVM = SVC(C=1, kernel='rbf')

   RFC = RandomForestClassifier(criterion='entropy')

   # Do grid search over k, n_components and SVM or Random Forest Classifier parameters:
   pipeline_SVM = Pipeline([('features', combined_features),('SVM',SVM)])
   pipeline_RFC = Pipeline([('features', combined_features),('RFC',RFC)])

   tuned_params_SVM = dict(features__pca__n_components=[5,30,40,50,60,70,80,87],
                           features__univ_select__k=[5,30,40,50,60,70,80,87],
                           SVM__C = (10.**np.arange(-3,3)).tolist(),SVM__gamma = (10.**np.arange(-3,3)).tolist())
   tuned_params_RFC = dict(features__pca__n_components=[5,30,40,50,60,70,80,87],
                           features__univ_select__k=[5,30,40,50,60,70,80,87],
                           RFC__n_estimators=[5,10,20,50],
                           RFC__max_depth = [50,150,250],
                           RFC__min_samples_split = [1,2,3,4],
                           RFC__min_samples_leaf = [1,2,3,4])
   
   grid_search = GridSearchCV(pipeline_RFC, param_grid=tuned_params_RFC,scoring='precision',cv=3,verbose=10)
   grid_search.fit(X_train, y_actual_train['attention'].values)
   print(grid_search.best_estimator_)
   y_true, y_pred = y_actual_test['attention'].values,grid_search.best_estimator_.predict(X_test)
   print confusion_matrix(y_true, y_pred)
   print "accuracy score:"+accuracy_score(y_true,y_pred)
   print "Done:"+str(datetime.datetime.now())

#predict session level, but do not aggregate all into one session
def subject_level_prediction(subject_levelDF):
      X = subject_levelDF.iloc[:,26:115]
      y_actual=subject_levelDF.iloc[:,15:16].astype(int)
      print y_actual
      #X.loc[X['PHOTOZ']=="",'PHOTOZ']=0
      X.drop('Image_Iterations', axis=1, inplace=True)
      X.drop('Image_Depth', axis=1, inplace=True)
      X.drop('Image_Channeldepth_red', axis=1, inplace=True)
      X.drop('Image_Channeldepth_green', axis=1, inplace=True)
      X.drop('Image_Channeldepth_blue', axis=1, inplace=True)
      X.astype(float)
      #scaling the data for feature selection
      X_scaled = preprocessing.scale(X)

      X_scaled_train, X_scaled_test, y_actual_train, y_actual_test = train_test_split(X_scaled, y_actual, test_size=0.3, random_state=0)

      pca_selection = PCA(n_components=2)

      selectKbest = SelectKBest(k=1)

      X_features = selectKbest.fit(X_scaled_train,y_actual_train).transform(X_scaled_train)

      svr = SVR(C=1,kernel='rbf')

      # Do grid search over k and SVR parameters in a pipeline:
      pipeline = Pipeline([('selectKbest', selectKbest),('svr',svr)])

      tuned_params = dict(selectKbest__k=[5,15,30,40,50,80,87],
                       svr__C = [1,10,50,100,500],
                       svr__gamma = [1e+2,1e+1,1e+0,1e-1,1e-2],
                       svr__epsilon=[1,0.1,0.01,0.001])

      grid_search = GridSearchCV(pipeline, param_grid=tuned_params,scoring='mean_squared_error',cv=3,verbose=10)
      grid_search.fit(X_scaled_train, y_actual_train['session_length'].values)
      print(grid_search.best_estimator_)
      y_true, y_pred = y_actual_test['session_length'].values, grid_search.best_estimator_.predict(X_scaled_test)
      print "Mean squared error:"+str(mean_squared_error(y_true,y_pred))
      pd.DataFrame(y_true, y_pred).to_csv("SVR_no_agg_pred_true.csv")

#CLASSIFICATION LEVEL PREDICTIONS - trying to predict session length based on the features
#session is defined by more than 30 minute gap between classifications, session length is defined by the number of classifications in a session
def classification_level_prediction(classifications_DF):
   X = classifications_DF.iloc[:,3:89]
   #assign the target (session length) to y and convert to int
   y_actual = classifications_DF.iloc[:,2:3]

   #scaling the data for feature selection
   X_scaled = preprocessing.scale(X)

   #feature selection:
   #featureSelector = SelectKBest(score_func=f_regression,k=15)
   #featureSelector.fit(X_norm,y_actual['session_length'].values)
   #create a list of selected features (columns)
   #selected_features = [X.columns[zero_based_index] for zero_based_index in list(featureSelector.get_support(indices=True))]
   
   #print "Those are the 15 selected features:"+str(selected_features)
   
   #create a data frame with only the selected features
   #X_selected = X[selected_features]

   #X_selected_norm=preprocessing.normalize(X_selected,norm='l2')
   #do not perform any feature selection

   #split the data set for train and test sets
   X_scaled_train, X_scaled_test, y_actual_train, y_actual_test = train_test_split(X_scaled, y_actual, test_size=0.3, random_state=0)
   # Set the parameters by cross-validation - for CLASSIFICATION LEVEL predictions
   tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e+1,1e+0,1e-1,1e-2,1e-3],
                     'C': [50, 100, 250, 500, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 500, 1000]}]
   print "y_actual_train shape:"+str(y_actual_train.shape)
   print "y_actual_test shape:"+str(y_actual_test.shape)

   scores = ['mean_squared_error','median_absolute_error']

   for score in scores:
      #using SVR to predict session length
      print(str(datetime.datetime.now())+": Predicting session length - Tuning hyper-parameters for %s" % score)

      # must be called with SVR instance as first argument
      clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=5, scoring=score)
      clf.fit(X_scaled_train, y_actual_train['session_length'].values)

      print("Best parameters set found...")
      print(clf.best_estimator_)
      print("Grid scores:")
      for params, mean_score, scores in clf.grid_scores_:
         print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
      print("Detailed classification report:")
      print("The model is trained on the full training set.")
      print("The scores are computed on the full testing set.")
      y_true, y_pred = y_actual_test['session_length'].values, clf.predict(X_scaled_test)
      print pd.DataFrame(X_scaled_test,y_true, y_pred).to_csv(str(score)+"pred_true.csv")
      print "Mean squared error:"+str(mean_squared_error(y_true,y_pred))
      print "Median absolute error:"+str(median_absolute_error(y_true,y_pred))
      print "Done:"+str(datetime.datetime.now())

#pipelining feature selection with gridsearch and SVR
def classification_level_SVR_pipeline(classifications_DF):
   X = classifications_DF.iloc[:,3:89]
   #assign the target (session length) to y and convert to int
   y_actual = classifications_DF.iloc[:,2:3].astype(float)

   #scaling the data for feature selection
   X_scaled = preprocessing.scale(X)

   X_scaled_train, X_scaled_test, y_actual_train, y_actual_test = train_test_split(X_scaled, y_actual, test_size=0.3, random_state=0)

   pca_selection = PCA(n_components=2)

   # Maybe some original features where good, too?
   selectKbest = SelectKBest(k=1)

   # Build estimator from PCA and Univariate selection:
   #combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

   # Use combined features to transform dataset:
   #X_features = combined_features.fit(X_scaled_train, y_actual_train['session_length'].values).transform(X_scaled_train)
   #X_features = pca_selection.fit(X_scaled_train).transform(X_scaled_train)
   X_features = selectKbest.fit(X_scaled_train,y_actual_train).transform(X_scaled_train)

   svr = SVR(C=1,kernel='rbf')

   # Do grid search over k and SVR parameters:
   pipeline = Pipeline([('selectKbest', selectKbest),('svr',svr)])

   tuned_params = dict(selectKbest__k=[5,15,20,30,40,50,60,70,80,86],
                       svr__C = [1,10,50,100,500],
                       svr__gamma = [1e+1,1e+0,1e-1,1e-2],
                       svr__epsilon=[1,0.1,0.01,0.001])

   grid_search = GridSearchCV(pipeline, param_grid=tuned_params,scoring='mean_squared_error',cv=3,verbose=10)
   grid_search.fit(X_scaled_train, y_actual_train['session_length'].values)
   print grid_search.best_estimator_
   y_true, y_pred = y_actual_test['session_length'].values, grid_search.best_estimator_.predict(X_scaled_test)
   print "Mean squared error:"+str(mean_squared_error(y_true,y_pred))
   pd.DataFrame(y_true, y_pred).to_csv("SVR_pred_true.csv")
   print grid_search.best_params_
   pickle.dump(grid_search.best_params_, open( "best_params.p", "wb" ) )

def classification_level_RandForest_pipeline(classifications_DF):
   X = classifications_DF.iloc[:,3:89]
   #assign the target (session length) to y and convert to int
   y_actual = classifications_DF.iloc[:,2:3].astype(float)

   #scaling the data for feature selection
   X_scaled = preprocessing.scale(X)

   X_scaled_train, X_scaled_test, y_actual_train, y_actual_test = train_test_split(X_scaled, y_actual, test_size=0.3, random_state=0)

 # Maybe some original features where good, too?
   selectKbest = SelectKBest(k=1,score_func=f_regression)

   # Build estimator from PCA and Univariate selection:
   X_features = selectKbest.fit(X_scaled_train,y_actual_train).transform(X_scaled_train)
   
   randomForestReg = RandomForestRegressor(n_estimators=1, criterion='mse')

   # Do grid search over k, n_components and SVR parameters:
   pipeline = Pipeline([('selectKbest', selectKbest),('randomForestReg',randomForestReg)])

   tuned_params = dict(selectKbest__k=[5,10,20,30,40,50,80],
                       randomForestReg__n_estimators=[1,2,4,8,16,32,64],
                       randomForestReg__min_samples_split=[2,3,5,10,20])

   grid_search = GridSearchCV(pipeline, param_grid=tuned_params,scoring='mean_squared_error',cv=3,verbose=10)
   grid_search.fit(X_scaled_train, y_actual_train['session_length'].values)
   print(grid_search.best_estimator_)
   y_true, y_pred = y_actual_test['session_length'].values,grid_search.best_estimator_.predict(X_scaled_test)
   print "Mean squared error:"+str(mean_squared_error(y_true,y_pred))
   pd.DataFrame(y_true, y_pred).to_csv("randomForestReg_pred_true.csv")

def classification_level_SGDReg_pipeline(classifications_DF):
   X = classifications_DF.iloc[:,3:89]
   #assign the target (session length) to y and convert to int
   y_actual = classifications_DF.iloc[:,2:3].astype(float)

   #scaling the data for feature selection
   X_scaled = preprocessing.scale(X)

   X_scaled_train, X_scaled_test, y_actual_train, y_actual_test = train_test_split(X_scaled, y_actual, test_size=0.5, random_state=0)

   pca_selection = PCA(n_components=2)

   X_features = pca_selection.fit(X_scaled_train['session_length'].values).transform(X_scaled_train)

   SGDReg = SGDRegressor(alpha=0.0001)

   # Do grid search over k, n_components and SVR parameters:
   pipeline = Pipeline([('pca', pca_selection),('SGDReg',SGDReg)])

   tuned_params = dict(pca__n_components=[5,30,40,50],
                     SGDReg__alpha=[0.1,0.01,0.001,0.0001,0.00001],
                     SGDReg__l1_ratio=[.05, .15, .5, .7, .9, .95, .99, 1],
                     SGDReg__penalty=['l2','l1','elasticnet'])

   grid_search = GridSearchCV(pipeline, param_grid=tuned_params,scoring='mean_squared_error',cv=3,verbose=10)
   grid_search.fit(X_scaled_train, y_actual_train['session_length'].values)
   print(grid_search.best_estimator_)
   y_true, y_pred = y_actual_test['session_length'].values,grid_search.best_estimator_.predict(X_scaled_test)
   print "Mean squared error:"+str(mean_squared_error(y_true,y_pred))
   pd.DataFrame(y_true, y_pred).to_csv("SGDReg_pred_true.csv")


def use_best_params(best_params,classifications_DF):
   print "Best Params:"+str(best_params)
   K=best_params['selectKbest__k']
   X = classifications_DF.iloc[:,3:89]
   #assign the target (session length) to y and convert to int
   y_actual = classifications_DF.iloc[:,2:3].astype(float)

   #scaling the data for feature selection
   X_scaled = preprocessing.scale(X)

   selectKbest = SelectKBest(k=K)
   X_features = selectKbest.fit_transform(X_scaled,y_actual['session_length'].values)
   scores = selectKbest.scores_
   print "All scores:"+str(scores)
   #60 best scores
   inx=np.where(scores>0.15)[0]
   print inx
   X.iloc[0:5,:].to_csv('small_features.csv',sep=',')
   X.iloc[0:5,inx].to_csv('small_best_features.csv',sep=',')
   print "Selected columns:"
   print X.iloc[0:5,inx].columns
   #print type(selectKbest.scores_)

   
print "Loading all the pickles..."
features_DF=pd.read_pickle('features.p')
classifications_DF=pd.read_pickle('classifications.p')
classifications_DF=classifications_DF.fillna(0)
best_params = pickle.load( open( "best_params.p", "rb" ) )
#fill NAs with 0 - in the future consider a different way of dealing with NAs
features_DF=features_DF.fillna(0)
timebeforerun=datetime.datetime.now()
#subject_level_prediction(classifications_DF)
subject_level_classification(features_DF)
#classification_level_prediction(classifications_DF)
#classification_level_SVR_pipeline(classifications_DF) #this was the best performing
#classification_level_SGDREG_pipeline(classifications_DF)
#classification_level_RandForest_pipeline(classifications_DF)
#use_best_params(best_params,classifications_DF)
timeafterun=datetime.datetime.now()
print "Time took to run:"+str(timeafterun-timebeforerun)

