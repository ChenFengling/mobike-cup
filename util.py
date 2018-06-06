#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:35:00 2017

@author: leo
""" 

import pandas as pd
import numpy as np
#import seaborn as sns
import os
import geohash
import math
from pandas_ply import install_ply, X, sym_call
import hashlib 
install_ply(pd)


def convertUint16(df, cols):
    for col in cols:
        df[col] = df[col].astype('uint16')
        df[col][df[col] == 0] = np.nan

def convertFloat32(df, cols):
    for col in cols:
        df[col] = df[col].astype('float32')
        df[col][df[col] == 0] = np.nan

def convertUint8(df, cols):
    for col in cols:
        df[col] = df[col].astype('uint8')
        df[col][df[col] == 0] = np.nan


def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

weekday = [10,11,12,15,16,17,18,19,22,23,24,25,26,27,31,1]
weekend = [13,14,20,21]
holiday = [28,29,30] 

def leak_sub(sub):
    leak= pd.read_csv('leak.csv', names = ['orderid', 'endloc_leak'])
    compare = pd.merge(sub, leak, on='orderid', how='left')
    compare['p3'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())] = compare['p2'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())]
    compare['p2'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())] = compare['p1'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())]
    compare['p1'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())] = compare['endloc_leak'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())]
    compare = compare[['orderid', 'p1', 'p2', 'p3']]
    return compare 


 
def printHeader(title):
    l = ''
    for i in range(10):
        l = l + '#'
    l = l + title
    for i in range(10):
        l = l + '#'
    print l

def processDatetime(ds):
    ds['hour'] = ds.starttime.map(lambda x: int(x[11:13]) + float(x[14:16]) / 60)
    ds['time'] = ''
    ds.loc[:,'time'][(ds.hour>=6) & (ds.hour <9)] = 'Morning Peak'
    ds.loc[:,'time'][(ds.hour>=9) & (ds.hour <11.5)] = 'Working Time'
    ds.loc[:,'time'][(ds.hour>=11.5) & (ds.hour <13)] = 'Noon'
    ds.loc[:,'time'][(ds.hour>=13) & (ds.hour <17)] = 'Working Time'
    ds.loc[:,'time'][(ds.hour>=17) & (ds.hour <19)] = 'Evening Peak'
    ds.loc[:,'time'][(ds.hour>=19) & (ds.hour <23)] = 'Night Time'
    ds.loc[:,'time'][(ds.hour>=23) | (ds.hour <6)] = 'Sleeping Time'
    ds['day'] = ds.starttime.map(lambda x: int(x[8:10]))
    ds['date'] = 'weekday'
    #ds['date'][ds.day.isin(weekday)] = 'weekday'
    ds['date'][ds.day.isin(weekend)] = 'weekend'
    ds['date'][ds.day.isin(holiday)] = 'weekend'
    
def decodeLocation(ds):
    decode = np.vectorize(geohash.decode)
    tmp = decode(ds["geohashed_start_loc"])
    ds["latitude_start_loc"] = tmp[0] * math.pi / 180
    ds["longitude_start_loc"] = tmp[1] * math.pi / 180
    if "geohashed_end_loc" in ds.columns:
        tmp = decode(ds["geohashed_end_loc"])
        ds["latitude_end_loc"] = tmp[0]* math.pi / 180
        ds["longitude_end_loc"] = tmp[1]* math.pi / 180

#just in train dataset
def calcDirection(ds):
    if 'longitude_candidate_end_loc' in ds.columns:
        ds['lngdelta'] = ds['longitude_candidate_end_loc'] - ds['longitude_start_loc'] 
        ds['latdelta'] = ds['latitude_candidate_end_loc'] - ds['latitude_start_loc'] 
    else:
        ds['lngdelta'] = ds['longitude_end_loc'] - ds['longitude_start_loc'] 
        ds['latdelta'] = ds['latitude_end_loc'] - ds['latitude_start_loc'] 
        
    cos_v = np.vectorize(math.cos)
    abs_v = np.vectorize(abs)
    atan2_v = np.vectorize(math.atan2)
    
    ds['deg'] = atan2_v(abs_v(cos_v(ds['latitude_start_loc']) * ds['lngdelta']), 
                 abs_v(ds['latdelta'])) * 180.0 / math.pi
                 
    ds['deg'][(ds.latdelta>=0) & (ds.lngdelta<0)] = 360 - ds['deg'][(ds.latdelta>=0) & (ds.lngdelta<0)]
    ds['deg'][(ds.latdelta<0) & (ds.lngdelta<=0)] = 180 + ds['deg'][(ds.latdelta<0) & (ds.lngdelta<=0)]
    ds['deg'][(ds.latdelta<0) & (ds.lngdelta>0)] = 180 - ds['deg'][(ds.latdelta<0) & (ds.lngdelta>0)]
    
    ds['Direction'] = '1'
    ds['Direction'][(ds['deg'] >= 22.5) & (ds['deg'] < 67.5)] = 'NE'
    ds['Direction'][(ds['deg'] >= 67.5) & (ds['deg'] < 112.5)] = 'E'
    ds['Direction'][(ds['deg'] >= 112.5) & (ds['deg'] < 157.5)] = 'SE'
    ds['Direction'][(ds['deg'] >= 157.5) & (ds['deg'] < 202.5)] = 'S'
    ds['Direction'][(ds['deg'] >= 202.5) & (ds['deg'] < 247.5)] = 'SW'
    ds['Direction'][(ds['deg'] >= 247.5) & (ds['deg'] < 292.5)] = 'W'
    ds['Direction'][(ds['deg'] >= 292.5) & (ds['deg'] < 337.5)] = 'NW'
    ds['Direction'][(ds['deg'] >= 337.5) | (ds['deg'] < 22.5)] = 'N'
    
    R = 6378137
    if 'longitude_candidate_end_loc' in ds.columns:
        ds['dist']= abs_v(R*(ds['latitude_candidate_end_loc']-ds['latitude_start_loc']))+abs_v(R*cos_v(ds['latitude_start_loc'])*ds['lngdelta'])

    else:
        ds['dist']= abs_v(R*(ds['latitude_end_loc']-ds['latitude_start_loc']))+abs_v(R*cos_v(ds['latitude_start_loc'])*ds['lngdelta'])
    
def encodeDist(ds):
    ds['dist_code'] = 'normal' 
    ds['dist_code'][ds.dist < 140] = 'shortshort'
    ds['dist_code'][(ds.dist < 800)  & (ds.dist >= 140)] = 'short'
    ds['dist_code'][(ds.dist < 2400)  & (ds.dist >= 1200)] = 'long'
    ds['dist_code'][ds.dist >= 2400] = 'longlong'

def loadDataset():
    if os.path.exists('./train.h5'):
        train_file = './train.h5'
        train_dataset = pd.read_hdf(train_file)
    else:
        train_file = './train.csv'
        train_dataset = pd.read_csv(train_file)
        decodeLocation(train_dataset)
        calcDirection(train_dataset)
        processDatetime(train_dataset)
        train_dataset.to_hdf('train.h5','tabel')

    
    if os.path.exists('./test.h5'):    
        test_file = './test.h5'
        test_dataset = pd.read_hdf(test_file)
    else:
        test_file = './test.csv'
        test_dataset = pd.read_csv(test_file)
        decodeLocation(test_dataset)
        processDatetime(test_dataset)
        test_dataset.to_hdf('test.h5','tabel')

    return train_dataset, test_dataset

def getOldUserDS(train, test):
    train_user1 = train[train.userid.isin(test.userid)]
    test_user1 = test[test.userid.isin(train.userid)]
    return train_user1, test_user1

def getNewUserDS(train, test): 
    train_user2 = train[~(train.userid.isin(test.userid))]
    test_user2 = test[~(test.userid.isin(train.userid))]
    return train_user2, test_user2 

def getTinyDS(train, test):
    train_rows = np.random.choice(train.index.values, 100000)
    sampled_train = train.ix[train_rows]
    test_rows = np.random.choice(test.index.values, 1000)
    samples_test = test.ix[test_rows]
    return sampled_train, samples_test
    
def get_user_loc(train, test):
    col_start = ['orderid','userid', 'geohashed_start_loc']
    col_end = ['orderid','userid', 'geohashed_end_loc']
    user_start = pd.concat([train[col_start], test[col_start]])
    user_end = train[col_end]
    
    tmp1 = user_start.groupby(['userid', 'geohashed_start_loc']).ply_select(start_num = X.orderid.count()).reset_index()
    tmp2 = user_end.groupby(['userid', 'geohashed_end_loc']).ply_select(end_num = X.orderid.count()).reset_index()
    
    tmp3 = tmp1.rename(columns={'geohashed_start_loc':'candidate_end_loc'})
    del tmp3['start_num']
    tmp4 = tmp2.rename(columns={'geohashed_end_loc':'candidate_end_loc'})
    del tmp4['end_num']
    
    user_loc = pd.concat([tmp3, tmp4])
    user_loc.drop_duplicates(inplace=True)
     
    decode = np.vectorize(geohash.decode)
    tmp = decode(user_loc["candidate_end_loc"])
    user_loc["latitude_candidate_end_loc"] = tmp[0]* math.pi / 180
    user_loc["longitude_candidate_end_loc"] = tmp[1]* math.pi / 180
 
    return user_loc

def get_user_end_loc(train,test):
    user_eloc = train[['userid','geohashed_end_loc', 'longitude_end_loc', 'latitude_end_loc']].drop_duplicates()
    result = pd.merge(test[['orderid','userid']],user_eloc, on='userid',how='left')
    result = result[['orderid', 'geohashed_end_loc', 'longitude_end_loc', 'latitude_end_loc']]
    return result

def get_user_start_loc(train, test):
    user_sloc = train[['userid','geohashed_start_loc', 'longitude_start_loc', 'latitude_start_loc']].drop_duplicates()
    result = pd.merge(test[['orderid', 'userid']], user_sloc, on='userid', how='left')
    result.rename(columns={'geohashed_start_loc' : 'geohashed_end_loc', 
                           'longitude_start_loc' : 'longitude_end_loc', 
                           'latitude_start_loc': 'latitude_end_loc'},inplace=True)
    result = result[['orderid', 'geohashed_end_loc', 'longitude_end_loc', 'latitude_end_loc']]
    return result

def get_start_top3end(train, test):
    sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc', 'longitude_end_loc', 'latitude_end_loc'],as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
    sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
    sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(3)
    result = pd.merge(test[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='left')
    result = result[['orderid', 'geohashed_end_loc', 'longitude_end_loc', 'latitude_end_loc']]
    return result

def user_end_pairs(train):
    user_end_pairs = train.groupby(['userid', 
                                      'geohashed_end_loc', 
                                      'longitude_end_loc', 
                                      'latitude_end_loc'])['orderid'].count().reset_index(name = 'count')
    
    user_end_pairs = user_end_pairs.rename(columns={'geohashed_end_loc' : 'candidate_end_loc',
                                                    'longitude_end_loc' : 'longitude_candidate_end_loc',
                                                    'latitude_end_loc' : 'latitude_candidate_end_loc'})
    del user_end_pairs['count']
    return user_end_pairs

def bike_end_pair(train):
    bike_end_pair = train.groupby(['bikeid', 
                                      'geohashed_end_loc', 
                                      'longitude_end_loc', 
                                      'latitude_end_loc'])['orderid'].count().reset_index(name = 'count')
    
    bike_end_pair = bike_end_pair.rename(columns={'geohashed_end_loc' : 'candidate_end_loc',
                                                    'longitude_end_loc' : 'longitude_candidate_end_loc',
                                                    'latitude_end_loc' : 'latitude_candidate_end_loc'})
    #del bike_end_pair['count']
    return bike_end_pair

def factorizeCat(train, test, category_features):
    ds = pd.concat([train, test])
    l = {}
    for feature in category_features:
        if feature == 'candidate_end_loc':
            list1 = train['geohashed_end_loc'].tolist()
            list2 = ds['geohashed_start_loc'].tolist()
            _,l[feature] = pd.factorize(list1+list2)
        elif feature == 'Direction':
            _,l[feature] = pd.factorize(['S', 'SE', 'SW', 'N', 'E', 'NW', 'W', 'NE'])
        elif feature == 'dist_code':
            _,l[feature] = pd.factorize(['shortshort', 'short','normal','long', 'longlong'])
        else:
            _,l[feature] = pd.factorize(ds[feature])
    return l

def buildTrainSet_by_orderEnd(ds, order_end):
    ds_merge = pd.merge(ds, order_end, on = 'orderid', how='left')
    ds_merge = ds_merge[~(ds_merge['geohashed_start_loc'] == ds_merge['candidate_end_loc'])]
    ds_merge = ds_merge[ds_merge.candidate_end_loc.notnull()]
    calcDirection(ds_merge)
    encodeDist(ds_merge)
    return ds_merge

def buildTrainSet_by_UserEnd(ds, user_end_pairs):
    ds_merge = pd.merge(ds, user_end_pairs, on = ['userid'], how = 'left')
#    if('geohashed_end_loc' in ds.columns):
#        del ds_merge['Direction_x']
#        del ds_merge['dist_x']
#        ds_merge = ds_merge.rename(columns = {'dist_y':'dist','Direction_y':'Direction'})
#        
    #ds_merge = ds_merge[~(ds_merge['geohashed_start_loc'] == ds_merge['candidate_end_loc'])]

    #ds_merge = ds_merge[~(ds_merge.candidate_end_loc.isnull())]
    
#    if('geohashed_end_loc' in ds.columns):
#        ds_merge['label'] = ds_merge['geohashed_end_loc'] == ds_merge['candidate_end_loc']
#        ds_merge['label'] = ds_merge['label'].map(int)
#    else:
#        ds_merge['label'] = 0
    calcDirection(ds_merge)
    #ds_merge = ds_merge[ds_merge.dist < 3000]
    encodeDist(ds_merge)
    return ds_merge
    
def getPartSumbbission(test, filename, newfile):
    part = pd.read_csv(filename, names = ['orderid', 'p1', 'p2', 'p3'])
    all_sub = pd.merge(test, part, on ='orderid', how = 'left')
    all_sub = all_sub.fillna('0000000')
    all_sub = all_sub[['orderid','p1', 'p2', 'p3']]
    all_sub.to_csv(newfile, header = None, index = False)
 
def getValidScore(x_valid, y_valid, yv_pred):
    y_pred = yv_pred
    x_valid['label_pred'] = y_pred
    top3 = x_valid.sort_values(['orderid','label_pred'],ascending=[True, False]).groupby(['orderid']).head(3)  
    top3['group_sort']=top3['label_pred'].groupby(top3['orderid']).rank(ascending=0,method='first') 
    sub = top3.pivot_table(index='orderid', columns='group_sort', values='candidate_end_loc',aggfunc=lambda x: ' '.join(x)).reset_index()
    sub = sub.rename(columns ={1.0:'p1', 2.0:'p2', 3.0:'p3'})
    valid_compare = pd.merge(x_valid[y_valid == 1], sub, on ='orderid', how= 'left')
    score = valid_compare[valid_compare['geohashed_end_loc'] == valid_compare['p1']].shape[0] + \
            valid_compare[valid_compare['geohashed_end_loc'] == valid_compare['p2']].shape[0] / 2.0 + \
            valid_compare[valid_compare['geohashed_end_loc'] == valid_compare['p3']].shape[0] /3.0

    score = float(score)
    return score

def generateSubmission(test, y_pred, filename = 'submission.csv'):
    test['label_pred'] = y_pred
    top3 = test.sort_values(['orderid','label_pred'],ascending=[True, False]).groupby(['orderid']).head(3)  
    top3['group_sort']=top3['label_pred'].groupby(top3['orderid']).rank(ascending=0,method='first') 
    sub = top3.pivot_table(index='orderid', columns='group_sort', values='candidate_end_loc',aggfunc=lambda x: ' '.join(x)).reset_index()
    sub = sub.rename(columns ={1.0:'p1', 2.0:'p2', 3.0:'p3'})
    return sub
    #sub.to_csv(filename, header = None, index = False)


def repair_by_leak(subfile, leakfle):
    sub = pd.read_csv(subfile, names = ['orderid', 'p1', 'p2', 'p3'])    
    leak = pd.read_csv(leakfle)
    compare = pd.merge(sub, leak, on='orderid', how='left')
    
    compare['p3'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())] = compare['p2'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())]
    compare['p2'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())] = compare['p1'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())]
    compare['p1'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())] = compare['endloc_leak'][(compare.p1 != compare.endloc_leak) & (compare.endloc_leak.notnull())]
    return  compare[['orderid','p1', 'p2', 'p3']]
    

def resortCol(data):
        cols = list(data)
        cols.insert(0, cols.pop(cols.index('label')))
        data = data.ix[:, cols]
        return data
    
def toBayesianFile(d, dict_of_features_factorized, filename):
        idx = 1
        l = {}
        for col in d.columns:
            idx = idx + 1
            print '开始转换', col
            col_list = dict_of_features_factorized[col].tolist()
            l[col] = {dict_of_features_factorized[col][i] : i for i in range(0, len(col_list))}
            d.loc[:,col] = d[col].map(l[col])
        d.to_csv(filename, header = None, index = False)