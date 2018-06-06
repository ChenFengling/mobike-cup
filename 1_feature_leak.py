#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:55:30 2017
leak velocty feature
@author: leo
"""
from util import *
import pandas as pd
import numpy as np
from tqdm import tqdm



train_ds, test_ds = loadDataset()


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

leak1 = get_leak_user(test_ds, 1800)
leak2 = get_leak_bike(test_ds, 7000)


tmp = pd.merge(leak2, leak1, how = 'outer', on ='orderid')
tmp['delta_t_x'][tmp.delta_t_x.isnull() & tmp.delta_t_y.notnull()] = tmp['delta_t_y'][tmp.delta_t_x.isnull() & tmp.delta_t_y.notnull()]
tmp['delta_t_y'][tmp.delta_t_x == tmp.delta_t_y] = np.nan

tmp[['orderid', 'delta_t_x']].to_csv('tmp/time_test_leak.csv', index = False)


leak1 = get_leak_user(train_ds, 1800)
leak2 = get_leak_bike(train_ds, 7000)


tmp = pd.merge(leak2, leak1, how = 'outer', on ='orderid')
tmp['delta_t_x'][tmp.delta_t_x.isnull() & tmp.delta_t_y.notnull()] = tmp['delta_t_y'][tmp.delta_t_x.isnull() & tmp.delta_t_y.notnull()]
tmp['delta_t_y'][tmp.delta_t_x == tmp.delta_t_y] = np.nan

tmp[['orderid', 'delta_t_x']].to_csv('tmp/time_train_leak.csv', index = False)

#%%
