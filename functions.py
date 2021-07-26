import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb
from tqdm import tqdm
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())

random_seed=42
m=10 #number of months
num_feat=[1, 2, 3, 5, 10, 15, 25, 50, 75] #
random.seed(random_seed)
np.random.seed(random_seed)

def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def check(i,ids):
    return i in ids

def train_model(X_train, y_train, X_val, y_val, random_state):
    
    np.random.seed(random_state)
    model = xgb.XGBClassifier(n_estimators=3000,
                              seed=random_state,
                              random_state=random_state,
                              n_jobs=1)

    model.fit(X_train, y_train,
              verbose=False,
              early_stopping_rounds=30,
              eval_metric='logloss',
              eval_set=[(X_val, y_val)])
    
    return model

def simulation(b, data, eps=10**-20):
    
    #Outputs
    shift_est=[]
    shift_err=[]
    
    #Splitting data
    for i in range(m):
        data[i][b]={}
        data[i][b]['X_train'], data[i][b]['X_test'], data[i][b]['y_train'], data[i][b]['y_test'] = train_test_split(data[i]['X'], data[i]['y'], test_size=.1, stratify=data[i]['y'], random_state=b)
    
    #For each number of features we make
    for k in num_feat:
        
        if b==0:
            print(k)
        
        ##### estimating w(x)=p(x)/q(x) #####
        wx=[]

        for i in range(m):
            Xw = data[0][b]['X_train'].iloc[:,:k].append(data[i][b]['X_train'].iloc[:,:k])
            yw = pd.DataFrame(np.hstack((np.zeros(data[0][b]['X_train'].shape[0]), np.ones(data[0][b]['X_train'].shape[0])))).squeeze()

            Xw_train, Xw_val, yw_train, yw_val = train_test_split(Xw, yw, test_size=0.2, stratify=yw, random_state=b)

            model_w = train_model(Xw_train, yw_train, Xw_val, yw_val, b)

            p=model_w.predict_proba(data[i][b]['X_test'].iloc[:,:k])[:,1]
            wx.append((p+eps)/(1-p+eps))

        ##### estimating w(x,y)=p(x,y)/q(x,y) #####
        wxy=[]

        for i in range(m):
            Xw = pd.concat((data[0][b]['X_train'].iloc[:,:k].append(data[i][b]['X_train'].iloc[:,:k]), data[0][b]['y_train'].append(data[i][b]['y_train'])), axis=1)
            yw = pd.DataFrame(np.hstack((np.zeros(data[0][b]['X_train'].shape[0]), np.ones(data[0][b]['X_train'].shape[0])))).squeeze()

            Xw_train, Xw_val, yw_train, yw_val = train_test_split(Xw, yw, test_size=0.2, stratify=yw, random_state=b)

            model_w = train_model(Xw_train, yw_train, Xw_val, yw_val, b)

            p=model_w.predict_proba(pd.concat((data[i][b]['X_test'].iloc[:,:k], data[i][b]['y_test']), axis=1))[:,1]
            wxy.append((p+eps)/(1-p+eps))

        ##### estimating shifts #####
        cov_est=[np.mean(np.log(np.array(w))) for w in wx]
        total_est=[np.mean(np.log(np.array(w))) for w in wxy]
        conc_est=[np.mean(np.log(np.array(w))-np.log(np.array(ww))) for (w, ww) in zip(wxy, wx)]

        cov_err=[np.std(np.log(np.array(w)))/np.sqrt(len(w)) for w in wx]
        total_err=[np.std(np.log(np.array(w)))/np.sqrt(len(w)) for w in wxy]
        conc_err=[np.std(np.log(np.array(w))-np.log(np.array(ww)))/np.sqrt(len(w)) for (w, ww) in zip(wxy, wx)]

        ##### storing #####
        shift_est.append([total_est, conc_est, cov_est])
        shift_err.append([total_err, conc_err, cov_err])
        
    return [shift_est, shift_err]

def simulation2(b, data):
    num_feat=[1, 2, 3, 5, 10, 15, 25, 50, 75]
    
    #Outputs
    p_est=[]
    p_err=[]
    
    #Splitting data
    for i in range(m):
        data[i][b]={}
        data[i][b]['X_train'], data[i][b]['X_test'], data[i][b]['y_train'], data[i][b]['y_test'] = train_test_split(data[i]['X'], data[i]['y'], test_size=.1, stratify=data[i]['y'], random_state=b)
        data[i][b]['X_train'], data[i][b]['X_val'], data[i][b]['y_train'], data[i][b]['y_val'] = train_test_split(data[i][b]['X_train'], data[i][b]['y_train'], test_size=.1, stratify=data[i][b]['y_train'], random_state=b)
        
    #For each number of features we make
    for k in num_feat:
        
        if b==0:
            print(k)
        
        ##### estimating delta prob #####
        models=[]
        p_fact=[]
        p_cfact=[]
            
        for i in range(m):
            X = data[i][b]['X_train'].iloc[:,:k]
            y = data[i][b]['y_train'].squeeze()
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=b)
            models.append(train_model(X_train, y_train, X_val, y_val, b))
            
        for i in range(m):
            p_fact.append(models[i].predict_proba(data[i][b]['X_test'].iloc[:,:k])[:,1])
            p_cfact.append(models[0].predict_proba(data[i][b]['X_test'].iloc[:,:k])[:,1])

        ##### storing #####
        
        p_fact_est = [np.mean(d) for d in p_fact]
        p_fact_err = [np.std(np.array(d))/np.sqrt(len(d)) for d in p_fact]
        p_cfact_est = [np.mean(d) for d in p_cfact]
        p_cfact_err = [np.std(np.array(d))/np.sqrt(len(d)) for d in p_cfact]
        
        p_est.append([p_fact_est, p_cfact_est])
        p_err.append([p_fact_err, p_cfact_err])

    return [p_est, p_err]