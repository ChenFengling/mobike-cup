#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:42:04 2017
lgb
@author: leo
"""
from util import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import feather as ft
import gc
import time

import os
#%%

#==============================================================================
# V1:09-23 01(2017-09-25 11:51:02): 采用23，24的数据，加入diff特征，轮次1500， 分数0.312, scale_pos_weight = 10, 分数0.312
# V1 (2017-09-25 11:58:18): 采用2324数据，加入diff，轮次1500,scale_pos_weight=1, is_unbalance = true
# V1 (09-25 20:06:05): 修正end的bug 
# 09-23 20:44:47 采用2324号两天的训练，轮次1500，加入time range特征，分数仍为0.304
# 9/24 23:59:17 采用21-24号的数据来训练，轮次1500，加入了time range的两个特征，分数???
# V3---09-24\ 13\:23\:23 采用21-24的数据来训练，轮次3000，加入time range的两个特征，分数??
# V4---09-24 20:26:08: 采用21-24的数据，加入了gs4, gs5, gs5_u等github组合特征，轮次2500，去掉time range的特征0.294
# V5: 采用23-24的数据，在V4上述特征基础加入方向的组合特征，轮次2500
# V6 --- 09-25 00:47: 采用23-24数据，在V5上去掉github特征，只加方向，轮次1500
# V7:（09-25\ 14\:51\:03）:采用23-24数据，在V5上去掉github特征，只加方向，轮次1500，scale_pos_weight=10（之前为30），得分0.313
# V8 (09-25 18:13:23): 采用23-24数据，在V7上，加time_range与共同起点数的特征，其他不变
# V9 (09-25\ 23\:37\:08): 在V8上加入6位地址特征,以前github特征 0.317
#==============================================================================


flag = False
from sklearn.model_selection import train_test_split

D = 2 ** 17

def hash_element(el):
    h = hash(el) % D
    if h < 0: 
        h = h + D
    return h

import string
def get_diff(s,t):
    abc = [str(i) for i in range(10)]+list(string.ascii_lowercase)
    dic = {i:c for c,i in enumerate(abc)}
    s,t = sorted([s,t])
    c = 0
    diff = 0
    for i,j in zip(s,t):
        if i!=j:
            diff += len(abc)**(len(s)-c-1)*abs(dic[i]-dic[j])
        c = c + 1
    return diff




#%%
def getDeltaT(t1, t2):
    return 12 - abs(abs(t1 - t2) - 12)

def getDeltaDeg(d1, d2):
    return 180 - abs(abs(d1 - d2) - 180)

#==============================================================================
# #%%添加特征
#==============================================================================



predictors_004 = ['userid', 'biketype','hour','dist', 'deg', 'date','start','end', 
                  'user|count','user|dist|mean','user|dist|max','user|dist|min', 
                 'user|eloc|count', 'user|eloc|dist|mean','user|eloc|dist|max',
                 'user|eloc|dist|min','user|eloc|time|mean','user|eloc|deg|mean',
                 'user|sloc|count','user|sloc|dist|mean','user|sloc|dist|max',
                 'user|sloc|dist|min','user|sloc|deg|mean','user|path|count',
                 'user|path|dist|mean',
                 'user|path|time|mean','user|path|deg|mean','user|RevPath|count',
                 'sloc|count','sloc|dist|mean','sloc|dist|max','sloc|dist|min',
                 'sloc|deg|mean','eloc|count','eloc|dist|mean','eloc|dist|max',
                 'eloc|dist|min','eloc|time|mean','eloc|deg|mean','path_count',
                 'path_avgTime','path|dist|mean',
                 'path|deg|mean','RevPath|count']

predictors_005 = ['userid', 'biketype','hour','dist', 'deg', 'date','start','end', 
                  'user|count', 'user|dist|mean','user|dist|max','user|dist|min', 'user|dist|std',
                 
                 'user|eloc|count', 'user|eloc|time|mean', 'user|eloc|dist|mean','user|eloc|dist|max',
                 'user|eloc|dist|min', 'user|eloc|dist|std', 'user|eloc|deg|mean',
                 
                 'user|sloc|count', 'user|sloc|time|mean', 'user|sloc|dist|mean','user|sloc|dist|max',
                 'user|sloc|dist|min', 'user|sloc|dist|std', 'user|sloc|deg|mean',
                 
                 
                 'user|eloc_as_sloc|count', 'user|eloc_as_sloc|time|mean', 'user|eloc_as_sloc|dist|mean',
                 'user|eloc_as_sloc|dist|max', 'user|eloc_as_sloc|dist|min', 'user|eloc_as_sloc|deg|mean', 
                 
                 'user|path|count', 'user|path|time|mean',
                 
                 'user|RevPath|count', 'user|RevPath|time|mean',
                 
                 'sloc|count','sloc|dist|mean','sloc|dist|max','sloc|dist|min', 'sloc|dist|std',
                 'sloc|deg|mean','sloc|time|mean', 
                 
                 'eloc|count','eloc|dist|mean','eloc|dist|max', 'eloc|dist|min', 'eloc|dist|std',
                 'eloc|time|mean','eloc|deg|mean',
                 
                 'eloc_as_sloc|count','eloc_as_sloc|dist|mean','eloc_as_sloc|dist|max', 
                 'eloc_as_sloc|dist|min', 'eloc_as_sloc|time|mean','eloc_as_sloc|deg|mean',
                 
                 'path_count', 'path_avgTime', 
                 
                 'RevPath|count', 'RevPath|time|mean',
                 
                 'delta_t1', 'delta_t2', 'user_hist']

predictors_006 = ['userid', 'biketype','hour','dist', 'deg', 'date','start','end', 
                  'user|count', 'user|dist|mean','user|dist|max','user|dist|min', 'user|dist|std',
                 
                 'user|eloc|count', 'user|eloc|time|mean', 'user|eloc|dist|mean','user|eloc|dist|max',
                 'user|eloc|dist|min', 'user|eloc|dist|std', 'user|eloc|deg|mean',
                 
                 'user|sloc|count', 'user|sloc|time|mean', 'user|sloc|dist|mean','user|sloc|dist|max',
                 'user|sloc|dist|min', 'user|sloc|dist|std', 'user|sloc|deg|mean',
                 
                 
                 'user|eloc_as_sloc|count', 'user|eloc_as_sloc|time|mean', 'user|eloc_as_sloc|dist|mean',
                 'user|eloc_as_sloc|dist|max', 'user|eloc_as_sloc|dist|min', 'user|eloc_as_sloc|deg|mean', 
                 
                 'user|path|count', 'user|path|time|mean',
                 
                 'user|RevPath|count', 'user|RevPath|time|mean',
                 
                 'sloc|count','sloc|dist|mean','sloc|dist|max','sloc|dist|min', 'sloc|dist|std',
                 'sloc|deg|mean','sloc|time|mean', 
                 
                 'eloc|count','eloc|dist|mean','eloc|dist|max', 'eloc|dist|min', 'eloc|dist|std',
                 'eloc|time|mean','eloc|deg|mean',
                 
                 'eloc_as_sloc|count','eloc_as_sloc|dist|mean','eloc_as_sloc|dist|max', 
                 'eloc_as_sloc|dist|min', 'eloc_as_sloc|time|mean','eloc_as_sloc|deg|mean',
                 
                 'path_count', 'path_avgTime', 
                 
                 'RevPath|count', 'RevPath|time|mean',
                 
                  'user_hist', 'delta_t1', 'delta_t2', 'sloc|user|count','eloc|user|count',
                 'eloc|start|count', 'sloc|end|count', 'user|eloc|start|count', 'user|sloc|end|count',
                 'start|end|user|count', 'diff']#, 'time_range|eloc|count' ,'user|time_range|count']

timerange_features = ['time_range|eloc|count' ,'user|time_range|count']

githup_fea = ['gs4', 'gs5', 'gs6', 'gs6_user', 'gs5_user', 'us', 'ue',
                 'user|dist|max|over', 'user|eloc|dist|max|over', 'user|sloc|dist|max|over',
                 'eloc|dist|max|over', 'sloc|dist|max|over']

#githup_fea = ['us', 'ue']
dir_features = ['sloc|dir|count', 'sloc|dir|dist|mean', 'eloc|dir|count', 'eloc|dir|dist|mean',
                'user|sloc|dir|count', 'user|sloc|dir|dist|mean', 'user|eloc|dir|count','user|eloc|dir|dist|mean']

same_loc_fea = ['sameendcount', 'start1count', 'start2count', 'end1count', 'end2count', 'samestartcount']

loc6_fea = ['start6|count', 'end6|count', 'user|start6|count', 'user|end6|count', 
            'user|start6-end|count', 'user|start6-end6|count', 'start6-end|count',
            'start6-end6|count']

predictors_006.extend(githup_fea)
predictors_006.extend(dir_features)
predictors_006.extend(timerange_features)
#predictors_006.extend(same_loc_fea)
predictors_006.extend(loc6_fea)

v_get_diff = np.vectorize(get_diff)

def getSlocDirProfile(now, hist):
    result = hist.groupby(['geohashed_start_loc','Direction'], as_index = False).agg({'orderid':'count',
                         'dist':'mean'})
    result.rename(columns={'orderid':'sloc|dir|count',
                           'dist' : 'sloc|dir|dist|mean'
                           },inplace=True)
    now = pd.merge(now, result, on = ['geohashed_start_loc','Direction'], how='left')
    now['sloc|dir|count'].fillna(0, inplace = True)
    return now

def getElocDirProfile(now, hist):
    result = hist.groupby(['geohashed_end_loc','Direction'], as_index = False).agg({'orderid':'count',
                         'dist':'mean'})
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid':'eloc|dir|count',
                           'dist' : 'eloc|dir|dist|mean'
                           },inplace=True)
    now = pd.merge(now, result, on = ['candidate_end_loc','Direction'], how='left')
    now['eloc|dir|count'].fillna(0, inplace = True)
    return now

def getUserSlocDirProfile(now, hist):
    result = hist.groupby(['userid', 'geohashed_start_loc','Direction'], as_index = False).agg({'orderid':'count',
                         'dist':'mean'})
    result.rename(columns={'orderid':'user|sloc|dir|count',
                           'dist' : 'user|sloc|dir|dist|mean'
                           },inplace=True)
    now = pd.merge(now, result, on = ['userid', 'geohashed_start_loc','Direction'], how='left')
    now['user|sloc|dir|count'].fillna(0, inplace = True)
    return now

def getUserElocDirProfile(now, hist):
    result = hist.groupby(['userid','geohashed_end_loc','Direction'], as_index = False).agg({'orderid':'count',
                         'dist':'mean'})
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid':'user|eloc|dir|count',
                           'dist' : 'user|eloc|dir|dist|mean'
                           },inplace=True)
    now = pd.merge(now, result, on = ['userid','candidate_end_loc','Direction'], how='left')
    now['user|eloc|dir|count'].fillna(0, inplace = True)
    return now


#1001
def getUserTimerangeProfile(now, hist):
    result = hist.groupby(['userid', 'time','date'], as_index = False).agg({'orderid':'count'})
    #result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid' : 'user|time_range|count'
                           },inplace=True)
    now = pd.merge(now, result, on=['userid', 'time', 'date'], how='left')
    now['user|time_range|count'].fillna(0,inplace=True)
    #convertUint8(now, ['user|time_range|count'])
    return now

def getElocTimerangeProfile(now,hist):
    result = hist.groupby([ 'time','geohashed_end_loc', 'date'], as_index = False).agg({'orderid':'count'})
    #result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid' : 'time_range|eloc|count',
                           },inplace=True)
    now = pd.merge(now, result, on=['time','candidate_end_loc', 'date'], how='left')
    now['time_range|eloc|count'].fillna(0,inplace=True)
    return now

def getSloc6Profile(now, hist):
    result = hist.groupby(['gs6'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'orderid' : 'start6|count'},inplace=True)
    now = pd.merge(now, result, on = 'gs6', how = 'left')
    now['start6|count'].fillna(0,inplace=True)
    return now

def getEloc6Profile(now, hist):
    result = hist.groupby(['ge6'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'orderid' : 'end6|count'},inplace=True)
    now = pd.merge(now, result, on = 'ge6', how = 'left')
    now['end6|count'].fillna(0,inplace=True)
    return now

def getUserSloc6Profile(now, hist):
    result = hist.groupby(['gs6','userid'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'orderid' : 'user|start6|count'},inplace=True)
    now = pd.merge(now, result, on = ['gs6','userid'], how = 'left')
    now['user|start6|count'].fillna(0,inplace=True)
    return now

def getUserEloc6Profile(now, hist):
    result = hist.groupby(['ge6','userid'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'orderid' : 'user|end6|count'},inplace=True)
    now = pd.merge(now, result, on = ['ge6','userid'], how = 'left')
    now['user|end6|count'].fillna(0,inplace=True)
    return now

def getUserStart6EndProfile(now, hist):
    result = hist.groupby(['userid', 'gs6', 'geohashed_end_loc'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid' : 'user|start6-end|count'},inplace=True)
    now = pd.merge(now, result, on = ['userid', 'gs6', 'candidate_end_loc'], how = 'left')
    now['user|start6-end|count'].fillna(0,inplace=True)
    return now

def getUserStart6End6Profile(now, hist):
    result = hist.groupby(['userid', 'gs6', 'ge6'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'orderid' : 'user|start6-end6|count'},inplace=True)
    now = pd.merge(now, result, on = ['userid', 'gs6', 'ge6'], how = 'left')
    now['user|start6-end6|count'].fillna(0,inplace=True)
    return now

def getStart6EndProfile(now, hist):
    result = hist.groupby(['gs6', 'geohashed_end_loc'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid' : 'start6-end|count'},inplace=True)
    now = pd.merge(now, result, on = ['gs6', 'candidate_end_loc'], how = 'left')
    now['start6-end|count'].fillna(0,inplace=True)
    return now

def getStart6End6Profile(now, hist):
    result = hist.groupby(['gs6', 'ge6'], as_index = False).agg({'orderid':'count'})
    result.rename(columns={'orderid' : 'start6-end6|count'},inplace=True)
    now = pd.merge(now, result, on = ['gs6', 'ge6'], how = 'left')
    now['start6-end6|count'].fillna(0,inplace=True)
    return now





def gen_features(now, hist):
    
    
    now = getSlocDirProfile(now, hist)
    now = getElocDirProfile(now, hist)
    now = getUserSlocDirProfile(now, hist)
    now = getUserElocDirProfile(now, hist)
    
    now = getUserTimerangeProfile(now, hist)
    now = getElocTimerangeProfile(now, hist)
    
    now['ge6'] = now['candidate_end_loc'].apply(lambda x: x[:6])
    now['gs6'] = now['geohashed_start_loc'].apply(lambda x: x[:6])
    hist['ge6'] = hist['geohashed_end_loc'].apply(lambda x: x[:6])
    hist['gs6'] = hist['geohashed_start_loc'].apply(lambda x: x[:6])
    now = getSloc6Profile(now, hist)
    now = getEloc6Profile(now, hist)
    now = getUserSloc6Profile(now, hist)
    now = getUserEloc6Profile(now, hist)
    now = getUserStart6EndProfile(now, hist)
    now = getUserStart6End6Profile(now, hist)
    now = getStart6EndProfile(now, hist)
    now = getStart6End6Profile(now, hist)
    del now['ge6']
    del now['gs6']
    return now

train_ds, _ = loadDataset()
train_ds['date'] = (train_ds['date'] == 'weekend').astype('uint8')

#将频次，算成每天的平均次数

def process(ds, day):
    ds['us'] = (ds['geohashed_start_loc']+'_'+ds['userid'].astype(str)).apply(hash_element).astype('uint32')
    ds['ue'] = (ds['candidate_end_loc']+'_'+ds['userid'].astype(str)).apply(hash_element).astype('uint32') 
    ds['end'] = ds['candidate_end_loc'].apply(hash_element).astype('uint32')
    ds['user|count'] = ds['user|count'] / day
    ds['user|eloc|count'] = ds['user|eloc|count'] / day
    ds['user|sloc|count'] = ds['user|sloc|count'] / day
    ds['user|path|count'] = ds['user|path|count'] / day
    ds['user|RevPath|count'] = ds['user|RevPath|count'] / day
    ds['sloc|count'] = ds['sloc|count'] / day
    ds['eloc|count'] = ds['eloc|count'] / day
    ds['path_count'] = ds['path_count'] / day
    ds['RevPath|count'] = ds['RevPath|count'] / day
    ds['sloc|user|count'] = ds['sloc|user|count'] / day
    ds['eloc|user|count'] = ds['eloc|user|count'] / day
    ds['eloc|start|count'] = ds['eloc|start|count'] / day
    ds['sloc|end|count'] = ds['sloc|end|count'] / day
    ds['user|eloc|start|count'] = ds['user|eloc|start|count'] / day
    ds['user|sloc|end|count'] = ds['user|sloc|end|count'] / day
    ds['start|end|user|count'] = ds['start|end|user|count'] / day
    
    if 'user|time_range|count' in ds.columns:
        ds['time_range|eloc|count'] = ds['time_range|eloc|count'] / day
        ds['user|time_range|count'] = ds['user|time_range|count'] / day
        
    
    ds['diff'] = v_get_diff(ds['geohashed_start_loc'], ds['candidate_end_loc'])
    ds['user|dist|mean'][ds['user|dist|mean'].notnull()] = abs(ds['dist'][ds['user|dist|mean'].notnull()] * 1.0 /  ds['user|dist|mean'][ds['user|dist|mean'].notnull()])
    ds['user|dist|max'][ds['user|dist|max'].notnull()] = abs(ds['dist'][ds['user|dist|max'].notnull()] *1.0 /  ds['user|dist|max'][ds['user|dist|max'].notnull()])
    ds['user|dist|min'][ds['user|dist|min'].notnull()] = abs(ds['dist'][ds['user|dist|min'].notnull()] -  ds['user|dist|min'][ds['user|dist|min'].notnull()])
    ds['user|dist|std'][ds['user|dist|std'].notnull()] = (ds['user|dist|mean'][ds['user|dist|std'].notnull()]).astype('float32') / (1e-10 + ds['user|dist|std'][ds['user|dist|std'].notnull()])
    ds['user|dist|max|over'] = 0
    ds['user|dist|max|over'][ds['user|dist|max'].notnull()] = (ds['user|dist|max'][ds['user|dist|max'].notnull()] > 1).astype(int)    
    ds['user|eloc|dist|mean'][ds['user|eloc|dist|mean'].notnull()] = abs(ds['dist'][ds['user|eloc|dist|mean'].notnull()] *1.0 /  ds['user|eloc|dist|mean'][ds['user|eloc|dist|mean'].notnull()])
    ds['user|eloc|dist|max'][ds['user|eloc|dist|max'].notnull()] = abs(ds['dist'][ds['user|eloc|dist|max'].notnull()] * 1.0 /  ds['user|eloc|dist|max'][ds['user|eloc|dist|max'].notnull()])
    ds['user|eloc|dist|min'][ds['user|eloc|dist|min'].notnull()] = abs(ds['dist'][ds['user|eloc|dist|min'].notnull()] -  ds['user|eloc|dist|min'][ds['user|eloc|dist|min'].notnull()])
    ds['user|eloc|dist|std'][ds['user|eloc|dist|std'].notnull()] = (ds['user|eloc|dist|mean'][ds['user|eloc|dist|std'].notnull()]).astype('float32') / (1e-10 + ds['user|eloc|dist|std'][ds['user|eloc|dist|std'].notnull()])   
    ds['user|eloc|deg|mean'][ds['user|eloc|deg|mean'].notnull()] = getDeltaDeg(ds['deg'][ds['user|eloc|deg|mean'].notnull()], ds['user|eloc|deg|mean'][ds['user|eloc|deg|mean'].notnull()])
    ds['user|eloc|time|mean'][ds['user|eloc|time|mean'].notnull()] = getDeltaT(ds['hour'][ds['user|eloc|time|mean'].notnull()], ds['user|eloc|time|mean'][ds['user|eloc|time|mean'].notnull()] / 10.0)
    ds['user|eloc|dist|max|over'] = 0
    ds['user|eloc|dist|max|over'][ds['user|eloc|dist|max'].notnull()] = (ds['user|eloc|dist|max'][ds['user|eloc|dist|max'].notnull()] > 1).astype(int)
    ds['user|sloc|dist|mean'][ds['user|sloc|dist|mean'].notnull()] = abs(ds['dist'][ds['user|sloc|dist|mean'].notnull()] * 1.0 /  ds['user|sloc|dist|mean'][ds['user|sloc|dist|mean'].notnull()])
    ds['user|sloc|dist|max'][ds['user|sloc|dist|max'].notnull()] = abs(ds['dist'][ds['user|sloc|dist|max'].notnull()] * 1.0 /  ds['user|sloc|dist|max'][ds['user|sloc|dist|max'].notnull()])
    ds['user|sloc|dist|min'][ds['user|sloc|dist|min'].notnull()] = abs(ds['dist'][ds['user|sloc|dist|min'].notnull()] -  ds['user|sloc|dist|min'][ds['user|sloc|dist|min'].notnull()])
    ds['user|sloc|dist|std'][ds['user|sloc|dist|std'].notnull()] = (ds['user|sloc|dist|mean'][ds['user|sloc|dist|std'].notnull()]).astype('float32') / (1e-10 + ds['user|sloc|dist|std'][ds['user|sloc|dist|std'].notnull()] )  
    ds['user|sloc|deg|mean'][ds['user|sloc|deg|mean'].notnull()] = getDeltaDeg(ds['deg'][ds['user|sloc|deg|mean'].notnull()], ds['user|sloc|deg|mean'][ds['user|sloc|deg|mean'].notnull()])
    ds['user|sloc|time|mean'][ds['user|sloc|time|mean'].notnull()] = getDeltaT(ds['hour'][ds['user|sloc|time|mean'].notnull()], ds['user|sloc|time|mean'][ds['user|sloc|time|mean'].notnull()] / 10.0)
    ds['user|sloc|dist|max|over'] = 0
    ds['user|sloc|dist|max|over'][ds['user|sloc|dist|max'].notnull()] = (ds['user|sloc|dist|max'][ds['user|sloc|dist|max'].notnull()] > 1).astype(int)
    ds['eloc|dist|mean'][ds['eloc|dist|mean'].notnull()] = abs(ds['dist'][ds['eloc|dist|mean'].notnull()] * 1.0 /  ds['eloc|dist|mean'][ds['eloc|dist|mean'].notnull()])    
    ds['eloc|dist|max'][ds['eloc|dist|max'].notnull()] = abs(ds['dist'][ds['eloc|dist|max'].notnull()] * 1.0 /  ds['eloc|dist|max'][ds['eloc|dist|max'].notnull()])
    ds['eloc|dist|min'][ds['eloc|dist|min'].notnull()] = abs(ds['dist'][ds['eloc|dist|min'].notnull()] -  ds['eloc|dist|min'][ds['eloc|dist|min'].notnull()])
    ds['eloc|dist|std'][ds['eloc|dist|std'].notnull()] = (ds['eloc|dist|mean'][ds['eloc|dist|std'].notnull()]).astype('float32') / (1e-10 + ds['eloc|dist|std'][ds['eloc|dist|std'].notnull()])
    ds['eloc|dist|max|over'] = 0
    ds['eloc|dist|max|over'][ds['eloc|dist|max'].notnull()] = (ds['eloc|dist|max'][ds['eloc|dist|max'].notnull()] > 1).astype(int)
    ds['eloc|deg|mean'][ds['eloc|deg|mean'].notnull()] = getDeltaDeg(ds['deg'][ds['eloc|deg|mean'].notnull()], ds['eloc|deg|mean'][ds['eloc|deg|mean'].notnull()])
    ds['eloc|time|mean'][ds['eloc|time|mean'].notnull()] = getDeltaT(ds['hour'][ds['eloc|time|mean'].notnull()], ds['eloc|time|mean'][ds['eloc|time|mean'].notnull()] / 10.0)
    ds['sloc|dist|mean'][ds['sloc|dist|mean'].notnull()] = abs(ds['dist'][ds['sloc|dist|mean'].notnull()] * 1.0 /  ds['sloc|dist|mean'][ds['sloc|dist|mean'].notnull()])
    ds['sloc|dist|max'][ds['sloc|dist|max'].notnull()] = abs(ds['dist'][ds['sloc|dist|max'].notnull()] * 1.0 /  ds['sloc|dist|max'][ds['sloc|dist|max'].notnull()])
    ds['sloc|dist|min'][ds['sloc|dist|min'].notnull()] = abs(ds['dist'][ds['sloc|dist|min'].notnull()] -  ds['sloc|dist|min'][ds['sloc|dist|min'].notnull()])
    ds['sloc|dist|std'][ds['sloc|dist|std'].notnull()] = (ds['sloc|dist|mean'][ds['sloc|dist|std'].notnull()]).astype('float32') / (1e-10 + ds['sloc|dist|std'][ds['sloc|dist|std'].notnull()])
    ds['sloc|dist|max|over'] = 0
    ds['sloc|dist|max|over'][ds['sloc|dist|max'].notnull()] = (ds['sloc|dist|max'][ds['sloc|dist|max'].notnull()] > 1).astype(int)
    ds['sloc|deg|mean'][ds['sloc|deg|mean'].notnull()] = getDeltaDeg(ds['deg'][ds['sloc|deg|mean'].notnull()], ds['sloc|deg|mean'][ds['sloc|deg|mean'].notnull()])
    ds['sloc|time|mean'][ds['sloc|time|mean'].notnull()] = getDeltaT(ds['hour'][ds['sloc|time|mean'].notnull()], ds['sloc|time|mean'][ds['sloc|time|mean'].notnull()] / 10.0)
    
    if dir_features[1] in ds.columns:
        for fea in dir_features:
            if 'count' in fea:
                ds[fea] = ds[fea] / day
            if 'dist|mean' in fea:
                ds[fea][ds[fea].notnull()] = ds['dist'][ds[fea].notnull()] * 1.0 / ds[fea][ds[fea].notnull()]     
    
    ds['gs4'] = ds['geohashed_start_loc'].apply(lambda x: x[:4])
    ds['gs5'] = ds['geohashed_start_loc'].apply(lambda x: x[:5])
    ds['gs6'] = ds['geohashed_start_loc'].apply(lambda x: x[:6])
    ds['gs6_user'] = (ds['gs6']+'_'+ds['userid'].astype(str)).apply(hash_element).astype('uint32')
    ds['gs5_user'] = (ds['gs5']+'_'+ds['userid'].astype(str)).apply(hash_element).astype('uint32')
    ds['gs4'] = ds['gs4'].apply(hash_element).astype('uint32')
    ds['gs5'] = ds['gs5'].apply(hash_element).astype('uint32')
    ds['gs6'] = ds['gs6'].apply(hash_element).astype('uint32')
    
#%%
params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['sub_feature'] = 0.9
params['lambda_l1'] = 10
params['lambda_l2'] = 50

params['num_leaves'] = 300
params['max_depth'] = 10
params['bagging_fraction'] = 0.8
params['bagging_freq'] = 5
#params['query'] = 0
params['scale_pos_weight'] = 10
params['min_data'] = 500
params['min_hessian'] = 1
params['early_stopping_round'] = 30
#params['is_unbalance']='true'

params['bin_construct_sample_cnt'] = 100000

#%%

    
def train(filenames, kfold):
    model_path = 'model/lgb_model_fold%d'%kfold

    t0 = pd.DataFrame()
    for f in filenames:
        t1 = ft.read_dataframe(f)
        print 't1.shape = ', t1.shape
        print f
        print t1.columns
        if f == 'tmp/CV_train_2324_top20_addfea.feather':
            hist = train_ds[train_ds.day<23]
        if f == 'tmp/CV_train_2122_top20_addfea.feather':
            hist = train_ds[(train_ds.day<21) | (train_ds.day >22)]
        t1 = gen_features(t1, hist)
        t0 = pd.concat([t0, t1])
        
    process(t0, 12.0)
    print 't0.shape = ', t0.shape

    
    order = pd.Series(t0.orderid.unique())
    train_order, valid_order = train_test_split(order, test_size=0.2, random_state=42)
    t1 = t0[t0.orderid.isin(train_order)]
    t2 = t0[t0.orderid.isin(valid_order)]
    
    x_train = t1[predictors_006]
    y_train = t1['label']
    
    print 'feature count: %d'%x_train.shape[1]
    
    x_valid = t2[predictors_006]
    y_valid = t2['label']
    
    if os.path.exists(model_path) & flag:
        clf = lgb.Booster(model_file = model_path)
    else:
        print 'begin train, kfold = %d' % kfold
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        watchlist = [d_train, d_valid]
        
        clf = lgb.train(params, d_train, 1500, watchlist)
        print ' 模型保存....\n'
        clf.save_model('model/lgb_model_fold%d'%kfold)
    
    print('Feature names:', clf.feature_name())

# feature importances
    fea_useful = pd.Series(clf.feature_importance(), index = clf.feature_name()).sort_values(ascending = False)
    fea_useful = fea_useful / fea_useful.sum()

    print 'lgb fea shape = ', fea_useful.shape
    print fea_useful

    p_mean = clf.predict(t2[predictors_006])
    
    limit_score = t2['label'].sum() / float(t2.orderid.unique().shape[0])
    hit_score = getValidScore(t2, t2['label'], p_mean) / float(t2.orderid.unique().shape[0])
    print 'Train理论上限值: %.3f, 模型得分: %.3f'%(limit_score, hit_score) #0.567,   0.405
    
    
    print ' ...\n'
    return clf

#fiels = ['tmp/CV_train_2122_top20_addfea.feather','tmp/CV_train_2324_top20_addfea.feather']
fiels = ['tmp/CV_train_2324_top20_addfea2.feather']
#fiels = ['tmp/CV_train_2122_top20_addfea.feather']

#model = train(fiels, 2124)

model = train(fiels, 2324)




#%% predict test
#model_path = 'model/lgb_model_fold%d'%2324
3model = lgb.Booster(model_file = model_path)

gc.collect()
sub_fea = ['orderid', 'candidate_end_loc', 'day']

def predict(ts):
    ts = gen_features(ts, train_ds)
    process(ts, 14.0)
    printHeader('Model Predict')
    y_test_pred = model.predict(ts[predictors_006])
    #result = generateSubmission(ts, y_test_pred)
    return y_test_pred

t1 = ft.read_dataframe('tmp/test_sample_feature2528_addfea.feather')

p1 = predict(t1[t1.day == 25])
p2 = predict(t1[t1.day == 26])
p3 = predict(t1[t1.day == 27])
p4 = predict(t1[t1.day == 28])


del t1
gc.collect()

t2 = ft.read_dataframe('tmp/test_sample_feature2932_addfea.feather')

p5 = predict(t2[t2.day <= 30])
p6 = predict(t2[t2.day > 30])

p = np.concatenate((p1,p2,p3,p4,p5,p6))

str_T = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
np.save('pred/p_%s.npy'%str_T, p)

#t = pd.concat([r1,r2,r3,r4,r5,r6])
del t2
gc.collect()

t = pd.read_csv('tmp/test_sample_ts.csv')
#result = pd.concat([r1,r2, r3, r4, r5, r6])
result = generateSubmission(t, p)

_,test = loadDataset()
  

result = pd.merge(test[['orderid']],result,on='orderid',how='left')
result.fillna('0000000',inplace=True)
result.to_csv('sub/lgb_%s.csv'%str_T, index=False,header=False)

print 'done!'




