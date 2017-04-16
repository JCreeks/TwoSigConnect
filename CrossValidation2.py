
# coding: utf-8

# In[1]:




# In[7]:

from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn import linear_model
#import xgboost as xgb
#from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import log_loss, mean_squared_error as mse, r2_score 
from sklearn.metrics.scorer import make_scorer


# In[18]:

def CVScore(model, X_train, y_train, n_splits=5, seed=2017, my_score=mse):
    cv_scores = []
    if not len(np.array(X_train).shape)==0:
        X_train=np.array(X_train)
        y_train=np.array(y_train)
    kf=StratifiedKFold(y_train, n_folds=2, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf:
        X_CVtrain = X_train[train_idx]
        y_CVtrain = y_train[train_idx]
        X_CVholdout = X_train[test_idx]
        y_CVholdout = y_train[test_idx]
        model.fit(X_CVtrain, y_CVtrain)
        if my_score==log_loss:
            pred=model.predict_proba(X_CVholdout)
        else:
            pred = model.predict(X_CVholdout)[:]
        cv_scores.append(my_score(y_CVholdout, pred))
    return np.mean(cv_scores)


# In[ ]:



