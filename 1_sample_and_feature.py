#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:15:50 2017

@author: leo
"""
#==============================================================================
# 1.构建样本集
# 2.生成xgb可用的数据特征
#==============================================================================
from util import *
import pandas as pd
import feather as ft
import sys
import argparse
import os
import gc
#%%
parser = argparse.ArgumentParser( description = 'Convert CSV file to Vowpal Wabbit format.' )
parser.add_argument( "out_file", help = "path to hist file" )
parser.add_argument( "-s", "--day_start", type = int, default = 23,
	help = "day for train start)")

parser.add_argument( "-e", "--day_end", type = int, default = 24,
	help = "day for train start)")



args = parser.parse_args()


#%%
D = 2 ** 17

def hash_element(el):
    h = hash(el) % D
    if h < 0: 
        h = h + D
    return h


#%% select sample

#==============================================================================
# S2: 获取起点对应的前3的终点
# 
#==============================================================================




def get_start_top10end(train):
    sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc', 'longitude_end_loc', 'latitude_end_loc'],as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
    sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
    sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(20)
    sloc_eloc_count = sloc_eloc_count.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                                                     'longitude_end_loc' : 'longitude_candidate_end_loc',
                                                     'latitude_end_loc' : 'latitude_candidate_end_loc'})
    del sloc_eloc_count['sloc_eloc_count']
    return sloc_eloc_count


#==============================================================================
# S1: 获取用户对应的起点与终点
#==============================================================================

def get_user_loc(train, test):
    col_start = ['orderid','userid', 'geohashed_start_loc']
    col_end = ['orderid','userid', 'geohashed_end_loc']
    user_start = pd.concat([train[col_start], test[col_start]])
    user_end = train[col_end]
    
    tmp1 = user_start[['userid', 'geohashed_start_loc']].drop_duplicates()
    tmp2 = user_end[['userid', 'geohashed_end_loc']].drop_duplicates()
    
    #tmp1 = user_start.groupby(['userid', 'geohashed_start_loc']).ply_select(start_num = X.orderid.count()).reset_index()
    #tmp2 = user_end.groupby(['userid', 'geohashed_end_loc']).ply_select(end_num = X.orderid.count()).reset_index()
    
    tmp3 = tmp1.rename(columns={'geohashed_start_loc':'candidate_end_loc'})
    #del tmp3['start_num']
    tmp4 = tmp2.rename(columns={'geohashed_end_loc':'candidate_end_loc'})
    #del tmp4['end_num']
    
    user_loc = pd.concat([tmp3, tmp4])
    user_loc.drop_duplicates(inplace=True)
    
    decode = np.vectorize(geohash.decode)
    tmp = decode(user_loc["candidate_end_loc"])
    user_loc["latitude_candidate_end_loc"] = tmp[0]* math.pi / 180
    user_loc["longitude_candidate_end_loc"] = tmp[1]* math.pi / 180
 
    return user_loc


    
#==============================================================================
# bike 按时间排序进行leak
#==============================================================================


def get_leak_bike(ds, d_time):
    sorted_ds = ds.sort_values(['bikeid','starttime'])
    sorted_ds['start'] = pd.to_datetime(sorted_ds['starttime']).reset_index(drop=True).values
    sorted_ds['endtime'] = 0
    sorted_ds['endloc_leak'] = ''
    sorted_ds['nextbike'] = -1
    
    rows_len = ds.shape[0]
    sorted_ds['endtime'].iloc[:(rows_len-1)] = sorted_ds['start'].iloc[1:].values 
    sorted_ds['endloc_leak'].iloc[:(rows_len-1)] = sorted_ds['geohashed_start_loc'].iloc[1:].values
    sorted_ds['nextbike'].iloc[:(rows_len-1)] = sorted_ds['bikeid'].iloc[1:].values
    sorted_ds['delta_t'] = sorted_ds['endtime'] - sorted_ds['start']
    sorted_ds.delta_t = sorted_ds.delta_t.astype('timedelta64[s]')
    if 'dist' in sorted_ds.columns:
        sorted_ds['v'] = sorted_ds['dist'] / sorted_ds['delta_t'] * 3600
    
    sorted_ds['valid'] = sorted_ds['delta_t'].apply(lambda x: (x < d_time))
    
    leak = sorted_ds[sorted_ds.valid & (sorted_ds.bikeid == sorted_ds.nextbike)]
    return leak[['orderid', 'bikeid', 'endloc_leak', 'delta_t']]

#==============================================================================
# user 按时间排序进行leak
#==============================================================================

def get_leak_user(ds, d_time):
    sorted_ds = ds.sort_values(['userid','starttime'])
    sorted_ds['start'] = pd.to_datetime(sorted_ds['starttime']).reset_index(drop=True).values
    sorted_ds['endtime'] = 0
    sorted_ds['endloc_leak'] = ''
    sorted_ds['nextuser'] = -1
    
    rows_len = ds.shape[0]
    sorted_ds['endtime'].iloc[:(rows_len-1)] = sorted_ds['start'].iloc[1:].values 
    sorted_ds['endloc_leak'].iloc[:(rows_len-1)] = sorted_ds['geohashed_start_loc'].iloc[1:].values
    sorted_ds['nextuser'].iloc[:(rows_len-1)] = sorted_ds['userid'].iloc[1:].values
    sorted_ds['delta_t'] = sorted_ds['endtime'] - sorted_ds['start']
    sorted_ds.delta_t = sorted_ds.delta_t.astype('timedelta64[s]')
    if 'dist' in sorted_ds.columns:
        sorted_ds['v'] = sorted_ds['dist'] / sorted_ds['delta_t'] * 3600
    
    sorted_ds['valid'] = sorted_ds['delta_t'].apply(lambda x: (x < d_time))
    
    leak = sorted_ds[sorted_ds.valid & (sorted_ds.userid == sorted_ds.nextuser)]
    return leak[['orderid', 'userid', 'endloc_leak', 'delta_t']]



def make_sample(train):
    leak_bike = get_leak_bike(train, 3600*24*10)
    leak_bike.rename(columns={'endloc_leak':'candidate_end_loc',
                               'delta_t' : 'delta_t1'},inplace=True)
    
    decode = np.vectorize(geohash.decode)
    tmp = decode(leak_bike["candidate_end_loc"])
    leak_bike["latitude_candidate_end_loc"] = tmp[0]* math.pi / 180
    leak_bike["longitude_candidate_end_loc"] = tmp[1]* math.pi / 180
    
    
    #leak_bike['isLeak'] = 1
    train0 = pd.merge(leak_bike[['orderid', 'candidate_end_loc', 'latitude_candidate_end_loc', 'longitude_candidate_end_loc']] , 
                      train, on = 'orderid', how = 'left')
    
    train1 = pd.merge(train, user_loc, on = 'userid', how ='left')
    train1 = train1[train1.candidate_end_loc != train1.geohashed_start_loc]
    #del train1['user_hist']
    
    train2 = pd.merge(train, stop3e, on ='geohashed_start_loc', how ='left')
    
    result = pd.concat([train1, train2, train0]).drop_duplicates()
    result = result[result.geohashed_start_loc != result.candidate_end_loc]
    
    
    #add bike leak diff time
    result = pd.merge(result, leak_bike[['orderid', 'candidate_end_loc', 'delta_t1']], on = ['orderid', 'candidate_end_loc'], how ='left')
    
    #add user leak diff time
    leak_user = get_leak_user(train, 3600*24*10)
    leak_user.rename(columns={'endloc_leak':'candidate_end_loc',
                               'delta_t' : 'delta_t2'},inplace=True)
    
    result = pd.merge(result, leak_user[['orderid', 'candidate_end_loc', 'delta_t2']], on = ['orderid', 'candidate_end_loc'], how ='left')


    #add candidate loc from user_loc
    user_loc['user_hist'] = 1
    result = pd.merge(result, user_loc, on = ['userid', 'candidate_end_loc', 'latitude_candidate_end_loc', 'longitude_candidate_end_loc'], how = 'left')
    

    calcDirection(result)
    result['date'] = (result['date'] == 'weekend').astype('uint8')
    result['start'] = result.geohashed_start_loc.apply(hash_element).astype('uint32')
    result['end'] = result.geohashed_end_loc.apply(hash_element).astype('uint32')
        
    
    result['label'] = (result['geohashed_end_loc']==result['candidate_end_loc']).astype('uint8')
    
    num_orderid = float(result.orderid.unique().shape[0])
    print 'sample中的orderid数: %d'%(num_orderid)
    
    print 'num_orderid: %d, 理论极限：%.4f'%(num_orderid, result['label'].sum() / num_orderid)
    return result




def log_numeric(x):
    if x == np.nan:
        return 0
    else:
        return np.round(np.log(1+x))
    




#%%
def getDeltaT(t1, t2):
    return 12 - abs(abs(t1 - t2) - 12)

def getDeltaDeg(d1, d2):
    return 180 - abs(abs(d1 - d2) - 180)

ti = np.array(np.arange(1,25,.5)).reshape(1,48)
def getAvgT(x):
    x = x.values.reshape(len(x),1)
    result = np.sum(getDeltaT(ti, x), axis = 0)
    return np.argmin(result) * 5

di = np.array(range(1,361,1)).reshape(1,360)
def getAvgD(x):
    x = x.values.reshape(len(x),1)
    result = np.sum(getDeltaDeg(di, x), axis = 0)
    return np.argmin(result) 

# col = user|path|time|mean
def getStdT(ds, by, col):
    ds['tmp'] = getDeltaT(ds['hour'] , n0[col]/10.0)
    grp = ds.groupby(by, as_index=False)['tmp'].map(np.mean)
    grp.dropna(inplace = True)
    del ds['tmp']
    ds = ds.merge(grp, on = by, how ='left')
    



#%%    
#%user feature
'''
用户使用次数
用户平均距离
用户最大距离
用户最小距离
'''
def getUserProfile(now, hist):
    result = hist.groupby('userid', as_index = False).agg({'orderid':'count','dist':[np.mean, 'max', 'min', 'std']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'orderid|count':'user|count',
                           'dist|mean':'user|dist|mean',
                           'dist|max': 'user|dist|max',
                           'dist|min': 'user|dist|min',
                           'dist|std': 'user|dist|std'},inplace=True)
    now = pd.merge(now, result, on='userid', how='left')
    now.fillna(0,inplace=True)
    convertUint16(now, ['user|dist|mean', 'user|dist|max', 'user|dist|min', 'user|dist|std'])
    convertUint8(now, ['user|count'])
    return now



#% user eloc feature
'''
用户到这个地点的平均时间
用户到这个地点的平均方向（deg）
用户到这个地点的平均距离

'''

def getUserElocProfile(now, hist):
    result = hist.groupby(['userid', 'geohashed_end_loc'], as_index = False).agg({'orderid':'count','hour':[getAvgT], 'deg':[getAvgD], 'dist':[np.mean,'max', 'min', 'std']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid|count' : 'user|eloc|count',
                           'dist|mean':'user|eloc|dist|mean',
                           'dist|max': 'user|eloc|dist|max',
                           'dist|min': 'user|eloc|dist|min',
                           'dist|std': 'user|eloc|dist|std',
                           'hour|getAvgT': 'user|eloc|time|mean',
                           'deg|getAvgD': 'user|eloc|deg|mean'},inplace=True)
    now = pd.merge(now, result, on=['userid', 'candidate_end_loc'], how='left')
    now.fillna(0,inplace=True)
    convertUint16(now, ['user|eloc|dist|mean', 'user|eloc|dist|max', 
                        'user|eloc|dist|min', 'user|eloc|dist|std','user|eloc|deg|mean'])
    convertUint8(now, ['user|eloc|time|mean','user|eloc|count'])
    return now
    

#%user sloc feature
def getUserSlocProfile(now, hist):
    result = hist.groupby(['userid', 'geohashed_start_loc'], as_index = False).agg({'orderid' : 'count', 'hour':[getAvgT],'deg':[getAvgD], 'dist':[np.mean,'max', 'min', 'std']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'orderid|count': 'user|sloc|count',
                           'dist|mean':'user|sloc|dist|mean',
                           'dist|max': 'user|sloc|dist|max',
                           'dist|min': 'user|sloc|dist|min',
                           'dist|std': 'user|sloc|dist|std',
                           'hour|getAvgT': 'user|sloc|time|mean',
                           'deg|getAvgD': 'user|sloc|deg|mean'},inplace=True)
    now = pd.merge(now, result, on=['userid', 'geohashed_start_loc'], how='left')
    
    result.rename(columns={'geohashed_start_loc' : 'candidate_end_loc',
                            'user|sloc|count': 'user|eloc_as_sloc|count',
                           'user|sloc|dist|mean':'user|eloc_as_sloc|dist|mean',
                           'user|sloc|dist|max': 'user|eloc_as_sloc|dist|max',
                           'user|sloc|dist|min': 'user|eloc_as_sloc|dist|min',
                           'user|sloc|dist|std': 'user|eloc_as_sloc|dist|std',
                           'user|sloc|time|mean': 'user|eloc_as_sloc|time|mean',
                           'user|sloc|deg|mean': 'user|eloc_as_sloc|deg|mean'},inplace=True)
    
    now = pd.merge(now, result, on=['userid', 'candidate_end_loc'], how='left')

    now.fillna(0,inplace=True)
    convertUint16(now, ['user|sloc|dist|mean', 'user|sloc|dist|max', 
                        'user|sloc|dist|min', 'user|sloc|dist|std','user|sloc|deg|mean'])
    convertUint8(now, ['user|sloc|time|mean','user|sloc|count'])    
    return now

    

#%user path feature
def getUserPathProfile(now, hist):
    result = hist.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False).agg({'orderid':'count','hour':[getAvgT],'deg':[getAvgD],'dist': [np.mean,'max', 'min']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid|count' : 'user|path|count',
                           'hour|getAvgT': 'user|path|time|mean',
                           'dist|mean':'user|path|dist|mean',
                           'dist|max': 'user|path|dist|max',
                           'dist|min': 'user|path|dist|min',
                           'deg|getAvgD': 'user|path|deg|mean'},inplace=True)
    now = pd.merge(now, result, on=['userid', 'geohashed_start_loc','candidate_end_loc'], how='left')
    now.fillna(0,inplace=True)
    convertUint16(now, ['user|path|dist|mean', 'user|path|dist|max', 
                        'user|path|dist|min', 'user|path|deg|mean'])
    convertUint8(now, ['user|path|time|mean','user|path|count'])  
    return now
    
#%User Reverse Path feature

def getUserRevPathProfile(now, hist):
    result = hist.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False).agg({'orderid':'count','hour':[getAvgT]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns = {'geohashed_start_loc':'candidate_end_loc',
                             'geohashed_end_loc':'geohashed_start_loc',
                             'orderid|count': 'user|RevPath|count',
                             'hour|getAvgT': 'user|RevPath|time|mean'},inplace=True)
    now = pd.merge(now,result,on=['userid','geohashed_start_loc','candidate_end_loc'],how='left')
    now.fillna(0,inplace=True)
    
    convertUint8(now, ['user|RevPath|time|mean','user|RevPath|count'])  
    return now


#%sloc feature
def getSlocProfile(now, hist):
    result = hist.groupby(['geohashed_start_loc'], as_index = False).agg({'orderid' : 'count', 'hour':[getAvgT] , 'deg':[getAvgD], 'dist':[np.mean,'max', 'min', 'std']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'orderid|count': 'sloc|count',
                           'dist|mean':'sloc|dist|mean',
                           'dist|max': 'sloc|dist|max',
                           'dist|min': 'sloc|dist|min',
                           'dist|std': 'sloc|dist|std',
                           'deg|getAvgD': 'sloc|deg|mean',
                           'hour|getAvgT' : 'sloc|time|mean'},inplace=True)
    now = pd.merge(now, result, on=['geohashed_start_loc'], how='left')
    #now['sloc|count'].fillna(0,inplace=True)
    
    result.rename(columns={'geohashed_start_loc' : 'candidate_end_loc',
                            'sloc|count': 'eloc_as_sloc|count',
                           'sloc|dist|mean':'eloc_as_sloc|dist|mean',
                           'sloc|dist|max': 'eloc_as_sloc|dist|max',
                           'sloc|dist|min': 'eloc_as_sloc|dist|min',
                           'sloc|dist|std': 'eloc_as_sloc|dist|std',
                           'sloc|time|mean': 'eloc_as_sloc|time|mean',
                           'sloc|deg|mean': 'eloc_as_sloc|deg|mean'},inplace=True)
    
    now = pd.merge(now, result, on=['candidate_end_loc'], how='left')
    
    now.fillna(0,inplace=True)
    convertUint16(now, ['sloc|count', 
                        'sloc|dist|mean', 'sloc|dist|max', 
                        'sloc|dist|min', 'sloc|deg|mean',
                        'eloc_as_sloc|count', 
                        'eloc_as_sloc|dist|mean', 'eloc_as_sloc|dist|max', 
                        'eloc_as_sloc|dist|min', 'eloc_as_sloc|deg|mean'])
    #convertUint8(now, ['eloc_as_sloc|time|mean','eloc_as_sloc|count']) 
    
    return now

#%eloc feature
def getElocProfile(now, hist):
    result = hist.groupby([ 'geohashed_end_loc'], as_index = False).agg({'orderid':'count','hour':[getAvgT], 'deg':[getAvgD], 'dist':[np.mean,'max', 'min', 'std']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid|count' : 'eloc|count',
                           'dist|mean':'eloc|dist|mean',
                           'dist|max': 'eloc|dist|max',
                           'dist|min': 'eloc|dist|min',
                           'dist|std': 'eloc|dist|std',
                           'hour|getAvgT': 'eloc|time|mean',
                           'deg|getAvgD': 'eloc|deg|mean'},inplace=True)
    now = pd.merge(now, result, on=['candidate_end_loc'], how='left')
    now.fillna(0,inplace=True)
    convertUint16(now, ['eloc|count', 
                        'eloc|dist|mean', 'eloc|dist|max', 
                        'eloc|dist|min', 'eloc|dist|std', 'eloc|deg|mean'])
    convertUint8(now, ['eloc|time|mean']) 
    return now

#%loc to loc (path) feature
'''
这个路径的历史平均时间
这个路径的历史次数
'''
def getPathProfile(now, hist):
    result = hist.groupby(['geohashed_start_loc', 'geohashed_end_loc'], 
                          as_index = False).agg({'hour':[getAvgT], 
                                          'orderid':'count',
                                          'deg':[getAvgD],
                                          'dist': [np.mean,'max', 'min']})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'orderid|count':'path_count',
                           'hour|getAvgT': 'path_avgTime',
                           'dist|mean':'path|dist|mean',
                           'dist|max': 'path|dist|max',
                           'dist|min': 'path|dist|min',
                           'deg|getAvgD': 'path|deg|mean'},inplace=True)
    now = pd.merge(now, result, on=['geohashed_start_loc', 'candidate_end_loc'], how='left')
    now.fillna(0,inplace=True)
    convertUint16(now, ['path|dist|mean', 'path|dist|max', 
                        'path|dist|min', 'path|deg|mean'])
    convertUint8(now, ['path_count','path_avgTime']) 
    return now

#User Reverse Path feature

def getRevPathProfile(now, hist):
    result = hist.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False).agg({'orderid':'count', 'hour':[getAvgT]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns = {'geohashed_start_loc':'candidate_end_loc',
                             'geohashed_end_loc':'geohashed_start_loc',
                             'orderid|count':'RevPath|count',
                             'hour|getAvgT': 'RevPath|time|mean'},inplace=True)
    now = pd.merge(now,result,on=['geohashed_start_loc','candidate_end_loc'],how='left')
    now.fillna(0,inplace=True)
    
    convertUint8(now, ['RevPath|count','RevPath|time|mean']) 
    return now


#%%

#==============================================================================
# 补充特征
#==============================================================================

def getUniCnt(x):
    return len(x.unique())

def getUserSloc(now, hist):
    result = hist.groupby(['geohashed_start_loc'], as_index = False).agg({'userid' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'userid|getUniCnt':'sloc|user|count'},inplace=True)
    now = pd.merge(now, result, on=['geohashed_start_loc'], how='left')
    return now


def getUserEloc(now, hist):
    result = hist.groupby(['geohashed_end_loc'], as_index = False).agg({'userid' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'userid|getUniCnt':'eloc|user|count'},inplace=True)
    now = pd.merge(now, result, on=['candidate_end_loc'], how='left')
    return now


def getStartEnd(now, hist):
    result = hist.groupby(['geohashed_end_loc'], as_index = False).agg({'geohashed_start_loc' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'geohashed_start_loc|getUniCnt':'eloc|start|count'},inplace=True)
    now = pd.merge(now, result, on=['candidate_end_loc'], how='left')
    
    
    result = hist.groupby(['geohashed_start_loc'], as_index = False).agg({'geohashed_end_loc' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc|getUniCnt':'sloc|end|count'},inplace=True)
    now = pd.merge(now, result, on=['geohashed_start_loc'], how='left')
    
    return now


def getUserStartEnd(now, hist):
    result = hist.groupby(['userid', 'geohashed_end_loc'], as_index = False).agg({'geohashed_start_loc' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'geohashed_start_loc|getUniCnt':'user|eloc|start|count'},inplace=True)
    now = pd.merge(now, result, on=['userid', 'candidate_end_loc'], how='left')
    
    
    result = hist.groupby(['userid', 'geohashed_start_loc'], as_index = False).agg({'geohashed_end_loc' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc|getUniCnt':'user|sloc|end|count'},inplace=True)
    now = pd.merge(now, result, on=['userid', 'geohashed_start_loc'], how='left')
    
    result = hist.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index = False).agg({'userid' : [getUniCnt]})
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result.rename(columns={'geohashed_end_loc':'candidate_end_loc',
                           'userid|getUniCnt':'start|end|user|count'},inplace=True)
    now = pd.merge(now, result, on=['geohashed_start_loc', 'candidate_end_loc'], how='left')
    
    return now
    


def gen_features(now, hist):
    printHeader('生成user sloc feature')
    now = getUserSlocProfile(now, hist)
    
    printHeader('生成user eloc feature')
    now = getUserElocProfile(now, hist)
    
    printHeader('生成user path feature')
    now = getUserPathProfile(now, hist)
    printHeader('生成user rev path feature')
    now = getUserRevPathProfile(now, hist)
    
    gc.collect()
    printHeader('生成 sloc feature')
    now = getSlocProfile(now, hist)
    printHeader('生成 eloc feature')
    now = getElocProfile(now, hist)
    printHeader('生成path feature')
    now = getPathProfile(now, hist)
    printHeader('生成 rev path feature')
    now = getRevPathProfile(now, hist)
    
    gc.collect()
    
    printHeader('生成user feature')
    now = getUserProfile(now, hist)
    
    printHeader('生成user sloc')
    now = getUserSloc(now, hist)
    
    printHeader('生成user eloc')
    now = getUserEloc(now, hist)
    
    printHeader('生成start end')
    now = getStartEnd(now, hist)
    
    printHeader('生成user start end')
    now = getUserStartEnd(now, hist)
    
    gc.collect()
    print 'gen feature done!'
    #now.fillna(0,inplace=True)
    return now

#%%

def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)
        
def chunk_dataframe(df, n):
    for i in range(0, len(df), n):
        yield df.iloc[i:i+n]


chunkSize = 10000000
chunks = []


#%% select sample
train, test = loadDataset()

#train, test = getTinyDS(train, test)
#h0 = train[train.day < 21]
#n0 = train[train.day >= 21]
#%%

print 'from %d to %d'%(args.day_start, args.day_end)
sample_file = ''
if args.day_start < 24:
    h0 = train[(train.day < args.day_start) | (train.day > args.day_end)]
    n0 = train[(train.day >= args.day_start) & (train.day <= args.day_end)]
    sample_file = 'tmp/make_sample_%d-%d'%(args.day_start, args.day_end)
    print 'num of orderid: %d'%n0.shape[0]
else:
    h0 = train
    n0 = test
    n0['day'][n0['day'] == 1] = 32
    n0['geohashed_end_loc'] = np.nan
    sample_file = 'tmp/make_sample_25-32'

del train, test

#%%
printHeader('构造候选集')
user_loc = get_user_loc(h0, n0)
stop3e = get_start_top10end(h0)
#%%

#if param == 'tiny':
#    df_all = ft.read_dataframe('tmp/train_tiny.feather')
#else:
#    df_all = ft.read_dataframe('tmp/train.feather') 


printHeader('make sample train')



if os.path.exists(sample_file):
    #df_all = pd.read_csv(sample_file)
    df_all = ft.read_dataframe(sample_file)
else:
    df_all = make_sample(n0)
    df_all.reset_index(inplace = True)
    del df_all['index']
    df_all.columns = df_all.columns.astype('str')
    df_all.to_feather(sample_file)
    print 'save finished!'


df_all = df_all[df_all.candidate_end_loc.notnull()]

if args.day_start > 24:
    df_all = df_all[(df_all.day >= args.day_start) & (df_all.day <= args.day_end)]


printHeader( 'gen sample train features')
df_all = gen_features(df_all, h0)






if args.day_start < 24:
    time_train = pd.read_csv('tmp/time_train_leak.csv')
    
    df_all = pd.merge(df_all, time_train, on = 'orderid', how = 'left')
    df_all['velocity'] = df_all['dist'] / df_all['delta_t_x'] * 3600
    
    num = float(df_all.orderid.unique().shape[0])
    print 'train orginal shape: %.3f'%(df_all['label'].sum() / num)
    df_all = df_all[((df_all.velocity > 20000) & (df_all.label == 1)) | (df_all.velocity <= 20000) | df_all.velocity.isnull()]
    print 'train after shape: %.3f'%(df_all['label'].sum() / num)

#print df_all.columns()


df_all.reset_index(inplace = True)
del df_all['index']
df_all.columns = df_all.columns.astype('str')
print df_all.columns


print 'begin to save df_all!'

save_features = ['label', 'orderid', 'userid', 'bikeid', 'biketype', 'dist', 'start','end',
                 'geohashed_start_loc', 'geohashed_end_loc', 'candidate_end_loc', 
                 'Direction', 'deg', 'day', 'date','time', 'starttime','hour',
                 
                 'user|count', 'user|dist|mean','user|dist|max','user|dist|min', 'user|dist|std',
                 
                 'user|eloc|count', 'user|eloc|time|mean', 'user|eloc|dist|mean','user|eloc|dist|max',
                 'user|eloc|dist|min', 'user|eloc|dist|std', 'user|eloc|deg|mean',
                 
                 'user|eloc_as_sloc|count', 'user|eloc_as_sloc|time|mean', 'user|eloc_as_sloc|dist|mean',
                 'user|eloc_as_sloc|dist|max', 'user|eloc_as_sloc|dist|min', 'user|eloc_as_sloc|deg|mean', 
                 
                 'user|sloc|count', 'user|sloc|time|mean', 'user|sloc|dist|mean','user|sloc|dist|max',
                 'user|sloc|dist|min', 'user|sloc|dist|std', 'user|sloc|deg|mean',
                 
                 'user|path|count', 'user|path|time|mean','user|path|dist|mean','user|path|dist|max','user|path|dist|min',
                 
                 'user|RevPath|count', 'user|RevPath|time|mean',
                 
                 'sloc|count','sloc|dist|mean','sloc|dist|max','sloc|dist|min', 'sloc|dist|std',
                 'sloc|deg|mean','sloc|time|mean', 
                 
                 'eloc|count','eloc|dist|mean','eloc|dist|max', 'eloc|dist|min', 'eloc|dist|std',
                 'eloc|time|mean','eloc|deg|mean',
                 
                 'eloc_as_sloc|count','eloc_as_sloc|dist|mean','eloc_as_sloc|dist|max', 
                 'eloc_as_sloc|dist|min', 'eloc_as_sloc|time|mean','eloc_as_sloc|deg|mean',
                 
                 'path_count', 'path_avgTime', 'path|dist|mean','path|dist|max','path|dist|min',
                 
                 'RevPath|count', 'RevPath|time|mean',
                 'delta_t1', 'delta_t2', 'user_hist', 'sloc|user|count','eloc|user|count',
                 'eloc|start|count', 'sloc|end|count', 'user|eloc|start|count', 'user|sloc|end|count',
                 'start|end|user|count']

df_all[save_features].to_feather(args.out_file)



