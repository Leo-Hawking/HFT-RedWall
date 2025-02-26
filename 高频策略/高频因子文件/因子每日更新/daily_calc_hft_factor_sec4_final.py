import os
import random
import sys
import pickle as pkl
import warnings
import torch
import gc
import pandas as pd
import numpy as np
sys.path.append(r'/home/datamake118/equity_sec_strategy')
from sec_support import *
import math

def get_sec4_factor_all(date):
    #初始化
    factor_list = ['up_risk_ratio_10', 'down_risk_ratio_10', 'up_down_risk_ratio_10', 'up_risk_ratio_30', 'down_risk_ratio_30', 'up_down_risk_ratio_30', 'up_risk_ratio_90', \
        'down_risk_ratio_90', 'up_down_risk_ratio_90', 'up_risk_ratio_300', 'down_risk_ratio_300', 'up_down_risk_ratio_300', 'up_risk_ratio_900', 'down_risk_ratio_900', \
        'up_down_risk_ratio_900', 'illiquidity_10', 'cvilliquidity_10', 'illiquidity_30', 'cvilliquidity_30', 'illiquidity_90', 'cvilliquidity_90', 'illiquidity_300', 'cvilliquidity_300', \
        'illiquidity_900', 'cvilliquidity_900', 'diff_vol_mean_10', 'pv_corr_10', 'bias_10', 'pv_corr_trend_10', 'diff_vol_mean_30', 'pv_corr_30', 'bias_30', 'pv_corr_trend_30', 'diff_vol_mean_90', \
        'pv_corr_90', 'bias_90', 'pv_corr_trend_90', 'diff_vol_mean_300', 'pv_corr_300', 'bias_300', 'pv_corr_trend_300', 'diff_vol_mean_900', 'pv_corr_900', 'bias_900', 'pv_corr_trend_900', \
        'shortcut_10', 'shortcut_30', 'shortcut_90', 'shortcut_300', 'shortcut_900', 'negative_skewness_10', 'negative_skewness_30', 'negative_skewness_90', 'negative_skewness_300', \
        'negative_skewness_900', 'VCV_10', 'VCV_30', 'VCV_90', 'VCV_300', 'VCV_900', 'RTV_10', 'RVJ_10', 'RTV_30', 'RVJ_30', 'RTV_90', 'RVJ_90', 'RTV_300', 'RVJ_300', 'RTV_900', 'RVJ_900', \
        'RBV_10', 'RBV_30', 'RBV_90', 'RBV_300', 'RBV_900']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取数据
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()

    all_fea_dict = read_sec_table(date)
    pre_date = get_pre_date(date)
    for fea_name in ['close','open', 'vol', 'amt', 'num', 'high', 'low']:
        all_fea_dict[fea_name] = all_fea_dict[fea_name].reindex(columns=stock_list)
    ret = all_fea_dict['close'].pct_change().fillna(0)
    all_fea_dict['ret'] = ret
    get_sec4_factor_part1(all_fea_dict,SEC_FACTOR)
    get_sec4_factor_part2(all_fea_dict,SEC_FACTOR)
    get_sec4_factor_part3(all_fea_dict,SEC_FACTOR)
    get_sec4_factor_part4(all_fea_dict,SEC_FACTOR)
    get_sec4_factor_part5(all_fea_dict,SEC_FACTOR)
    return SEC_FACTOR

def get_sec4_factor_part1(all_fea_dict,SEC_FACTOR):
    ret = all_fea_dict['ret']
    ret_up = ret * (ret > 0).astype(int)
    ret_down = ret * (ret < 0).astype(int)
    
    for window in [10,30,90,300,900]:
        SEC_FACTOR.divid_save_factor(((ret_up ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba")/\
            (ret ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba")).replace([np.inf,-np.inf],np.nan), 'up_risk_ratio_' + str(window))
        SEC_FACTOR.divid_save_factor(((ret_down ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba")/\
            (ret ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba")).replace([np.inf,-np.inf],np.nan), 'down_risk_ratio_' + str(window))   
        SEC_FACTOR.divid_save_factor((((ret_up ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba") - (ret_down ** 2).rolling(window = window,min_periods = 1).sum(
            engine = "numba") )/(ret ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba")).fillna(0), 'up_down_risk_ratio_' + str(window))
    return

def get_sec4_factor_part2(all_fea_dict,SEC_FACTOR):
    ret = all_fea_dict['ret']
    illiquidity = ret.abs() / np.log(all_fea_dict['vol'] + 1)
    for window in [10,30,90,300,900]:
        SEC_FACTOR.divid_save_factor(illiquidity.rolling(window = window,min_periods = 1).mean(engine = "numba"), 'illiquidity_' + str(window)) 
        SEC_FACTOR.divid_save_factor((illiquidity.rolling(window = window,min_periods = 1).std(engine = "numba") /illiquidity.rolling(window = window,min_periods = 1).mean(
            engine = "numba")).replace([np.inf,-np.inf],np.nan), 'cvilliquidity_' + str(window))

    return

def get_sec4_factor_part3(all_fea_dict,SEC_FACTOR):
    ret = all_fea_dict['ret']
    vol = all_fea_dict['vol']
    diff = all_fea_dict['vol'].diff(1)
    price = all_fea_dict['close']
    for window in [10,30,90,300,900]:
        SEC_FACTOR.divid_save_factor((diff.abs().rolling(window = window,min_periods = 1).mean(engine = "numba") / \
            vol.rolling(window = window,min_periods = 1).mean(engine = "numba")).replace([np.inf,-np.inf],np.nan), 'diff_vol_mean_' + str(window))
        
        cov = vol.rolling(window = window,min_periods = 1).cov(price)
        vol_std = vol.rolling(window = window,min_periods = 1).std(engine = "numba") 
        price_std = price.rolling(window = window,min_periods = 1).std(engine = "numba")
        pv_corr = (cov/(vol_std * price_std)).replace([np.inf,-np.inf],np.nan)
        SEC_FACTOR.divid_save_factor(pv_corr, 'pv_corr_' + str(window))
        SEC_FACTOR.divid_save_factor(price / price.rolling(window = window,min_periods = 1).mean(engine = "numba")  - 1, 'bias_' + str(window))

        time = pv_corr.apply(lambda x:np.arange(len(ret)),axis = 0)
        cov = pv_corr.rolling(window = window,min_periods = 1).cov(time)
        var_time = time.rolling(window = window,min_periods = 1).var(engine = "numba")
        SEC_FACTOR.divid_save_factor(cov/var_time, 'pv_corr_trend_' + str(window))
    return

def get_sec4_factor_part4(all_fea_dict,SEC_FACTOR):
    shortcut = ((all_fea_dict['high'] - all_fea_dict['low']) / (all_fea_dict['close'] - all_fea_dict['open']).abs()).replace([np.inf,-np.inf],np.nan)
    for window in [10,30,90,300,900]:
        SEC_FACTOR.divid_save_factor(shortcut.rolling(window = window,min_periods = 1).mean(engine = "numba"), 'shortcut_' + str(window)) 
    
    ret = all_fea_dict['ret']
    for window in [10,30,90,300,900]:
        third_order_mat = ((ret.sub(ret.mean(axis = 1), axis = 0)) ** 3).rolling(window = window,min_periods = 1).sum(engine = "numba") 
        second_order_mat = ((ret.sub(ret.mean(axis = 1), axis = 0)) ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba") 
        SEC_FACTOR.divid_save_factor((third_order_mat/(second_order_mat ** 1.5)).replace([np.inf,-np.inf],np.nan), 'negative_skewness_' + str(window))
    
    for window in [10,30,90,300,900]:
        SEC_FACTOR.divid_save_factor((np.log(all_fea_dict['amt'] + 1).rolling(window = window,min_periods = 1).std(engine = "numba") / \
            np.log(all_fea_dict['amt'] + 1).rolling(window = window,min_periods = 1).mean(engine = "numba")).replace([np.inf,-np.inf],np.nan), 'VCV_' + str(window))
    
    return

def get_sec4_factor_part5(all_fea_dict,SEC_FACTOR):
    ret = all_fea_dict['ret']
    bp = 100 * ret
    V1 = ((bp ** 2) ** (1/3)).abs()
    V2 = ((bp.shift(1).fillna(0) ** 2) ** (1/3)).abs()
    V3 = ((bp.shift(2).fillna(0) ** 2) ** (1/3)).abs()
    miu = 2 ** (1/3) * math.gamma(5/6)/math.gamma(1/2)
    for window in [10,30,90,300,900]:
        RTV = (V1 * V2 * V3).rolling(window = window,min_periods = 1).sum(engine = "numba")  * miu ** (-3)
        SEC_FACTOR.divid_save_factor(RTV, 'RTV_' + str(window))
        RV = (bp ** 2).rolling(window = window,min_periods = 1).sum(engine = "numba") 
        RVJ = RV - RTV
        RVJ[RVJ < 0] = 0
        SEC_FACTOR.divid_save_factor(RVJ, 'RVJ_' + str(window))

    V1 = bp.abs()
    V2 = bp.shift(1).fillna(0).abs()
    for window in [10,30,90,300,900]:
        SEC_FACTOR.divid_save_factor((V1 * V2).rolling(window = window,min_periods = 1).sum(engine = "numba"), 'RBV_' + str(window)) 
    return

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_hft_factor_sec4'
    factor_path = r'/home/datamake118/datamake118_base/nas8/sec_factor4'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_sec4_factor_all),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)