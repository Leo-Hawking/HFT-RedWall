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

def calc_daily_factor_all1(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s'] + 1e-6
    amt_5s = all_fea_dict['amt_5s'] + 1e-6

    for lookback_m in [0.5,1,5,10]:
        window_period = int(lookback_m*12)
        window_period_name = str(int(lookback_m*60))
        ret_up = (ret_5s * (ret_5s > 0).astype(int)).rolling(window_period, min_periods=1).sum(engine='numba')
        ret_down = (ret_5s * (ret_5s < 0).astype(int)).rolling(window_period, min_periods=1).sum(engine='numba')
        second_data = ret_up/(1e-6+ret_up-ret_down)
        SEC_FACTOR.divid_save_factor(second_data, '%s_RSI'%window_period_name)

        second_data = (ret_5s*vol_5s).rolling(window_period, min_periods=1).sum(engine='numba')/vol_5s.rolling(window_period, min_periods=1).sum(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, '%s_wvol_ret'%window_period_name)

        second_data = ((np.log(price_5s).diff())**2).rolling(window_period, min_periods=1).sum(engine='numba')*ret_5s * 1e6
        SEC_FACTOR.divid_save_factor(second_data, '%s_wRV_ret'%window_period_name)

        rit = np.log(price_5s).diff()
        rit_up = (rit * (rit > 0).astype(int)).rolling(window_period, min_periods=1).sum(engine='numba')
        rit_down = (rit * (rit < 0).astype(int)).rolling(window_period, min_periods=1).sum(engine='numba')
        rit_all = (rit**2).rolling(window_period, min_periods=1).sum(engine='numba')+1e-6
        second_data = (rit_up-rit_down)/rit_all
        SEC_FACTOR.divid_save_factor(second_data, '%s_RSJ'%window_period_name)

        ret_sign = np.sign(ret_5s)
        second_data = (amt_5s*ret_sign).rolling(window_period,min_periods=1).mean(engine='numba') / amt_5s.rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, '%s_amount_dir'%window_period_name)

        logret_abs = np.log(np.abs(price_5s.pct_change(1))+1)
        second_data = (logret_abs/amt_5s).rolling(window_period,min_periods=1).mean(engine='numba') * 1e5
        SEC_FACTOR.divid_save_factor(second_data, '%s_liquity'%window_period_name)
    
    return

def calc_daily_factor_all2(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']

    rit = np.log(price_5s).diff()*100  #避免量纲过小
    rit_up = (rit > 0).astype(int)
    rit_down = (rit < 0).astype(int)
    rit_2 = rit**2
    rit_3 = rit**3

    for lookback_m in [0.5,1,5,10]:
        window_period = int(lookback_m*12)
        second_data = rit.rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_skew'%lookback_m)
        second_data = rit.rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_kurt'%lookback_m)

        second_data = (rit_2*rit_up).rolling(window_period,min_periods=1).mean(engine='numba') - (rit_2*rit_down).rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umd_mean'%lookback_m)
        second_data = (rit_2*rit_up).rolling(window_period,min_periods=1).std(engine='numba') - (rit_2*rit_down).rolling(window_period,min_periods=1).std(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umd_std'%lookback_m)

        second_data = rit_2.rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_mean'%lookback_m)
        second_data = rit_2.rolling(window_period,min_periods=1).std(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_std'%lookback_m)
        second_data = rit_2.rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_skew'%lookback_m)
        second_data = rit_2.rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_kurt'%lookback_m)

        second_data = rit_3.rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r3_%s_mean'%lookback_m)
        second_data = rit_3.rolling(window_period,min_periods=1).std(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r3_%s_std'%lookback_m)
        second_data = rit_3.rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'r3_%s_skew'%lookback_m)
        second_data = rit_3.rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'r3_%s_kurt'%lookback_m)
    
    return

def calc_daily_factor_all3(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']

    rit = np.log(price_5s).diff()*100  #避免量纲过小
    rit_up = (rit > 0).astype(int)
    rit_down = (rit < 0).astype(int)
    rit_2 = rit**2 + 1e-8

    for lookback_m in [0.5,1,5,10]:
        window_period = int(lookback_m*12)
        window_period_name = str(int(lookback_m*60))

        second_data = ((1/rit_2*rit_up).rolling(window_period,min_periods=1).std() - (1/rit_2*rit_down).rolling(window_period,min_periods=1).std()) * 1e-4
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umd_inv_std'%window_period_name)
        second_data = ((1/rit_2*rit_up).rolling(window_period,min_periods=1).mean(engine='numba') - (1/rit_2*rit_down).rolling(window_period,min_periods=1).mean(engine='numba')) * 1e-4
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umd_inv_mean'%window_period_name)

        second_data = (rit_2*rit_up).rolling(window_period,min_periods=1).skew() - (rit_2*rit_down).rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umd_skew'%window_period_name)
        second_data = (rit_2*rit_up).rolling(window_period,min_periods=1).kurt() - (rit_2*rit_down).rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umd_kurt'%window_period_name)
        second_data = rit.rolling(window_period,min_periods=1).quantile(0.2)/(rit.rolling(window_period,min_periods=1).quantile(0.95) - rit.rolling(window_period,min_periods=1).quantile(0.05) + 1e-6)
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_quan20_scale'%window_period_name)

    return

def calc_daily_factor_all4(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']

    rit = np.log(price_5s).diff()*100  #避免量纲过小
    vol_diff = np.log(vol_5s).diff()
    vol_up = (vol_diff > 0).astype(int)
    vol_down = (vol_diff < 0).astype(int)
    rit_2 = rit**2 + 1e-8

    for lookback_m in [0.5, 1, 5, 10]:
        window_period = int(lookback_m*12)
        window_period_name = str(int(lookback_m*60))

        second_data = ((1/rit_2*vol_up).rolling(window_period,min_periods=1).std(engine='numba') - (1/rit_2*vol_down).rolling(window_period,min_periods=1).std(engine='numba')) * 1e-4
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umdvol_inv_std'%window_period_name)
        second_data = ((1/rit_2*vol_up).rolling(window_period,min_periods=1).mean(engine='numba') - (1/rit_2*vol_down).rolling(window_period,min_periods=1).mean(engine='numba')) * 1e-4
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umdvol_inv_mean'%window_period_name)

        second_data = (rit_2*vol_up).rolling(window_period,min_periods=1).skew() - (rit_2*vol_down).rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umdvol_skew'%window_period_name)
        second_data = (rit_2*vol_up).rolling(window_period,min_periods=1).kurt() - (rit_2*vol_down).rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'r2_%s_umdvol_kurt'%window_period_name)

        second_data = (rit*vol_up).rolling(window_period,min_periods=1).mean(engine='numba') - (rit*vol_down).rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_umdvol_mean'%window_period_name)
        second_data = (rit*vol_up).rolling(window_period,min_periods=1).std(engine='numba') - (rit*vol_down).rolling(window_period,min_periods=1).std(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_umdvol_std'%window_period_name)
        second_data = (rit*vol_up).rolling(window_period,min_periods=1).skew() - (rit*vol_down).rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_umdvol_skew'%window_period_name)
        second_data = (rit*vol_up).rolling(window_period,min_periods=1).kurt() - (rit*vol_down).rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'r1_%s_umdvol_kurt'%window_period_name)
    
    return

def calc_daily_factor_all5(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']
    num_5s = all_fea_dict['num_5s']

    pamt = amt_5s / (num_5s + 1e-8)
    rit = np.log(price_5s).diff()*100  #避免量纲过小
    rit_up = (rit > 0).astype(int)
    rit_down = (rit < 0).astype(int)

    for lookback_m in [0.5, 1, 5, 10]:
        window_period = int(lookback_m*12)
        window_period_name = str(int(lookback_m*60))

        second_data = pamt.rolling(window_period,min_periods=1).std(engine='numba') / pamt.rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_std'%window_period_name)
        second_data = pamt.rolling(window_period,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_skew'%window_period_name)
        second_data = pamt.rolling(window_period,min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_kurt'%window_period_name)

        second_data = ((pamt*rit_up).rolling(window_period, min_periods=1).mean(engine='numba') - (pamt*rit_down).rolling(window_period, min_periods=1).mean(engine='numba')) / pamt.rolling(window_period, min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_umd_mean'%window_period_name)
        second_data = ((pamt*rit_up).rolling(window_period, min_periods=1).std(engine='numba') - (pamt*rit_down).rolling(window_period, min_periods=1).std(engine='numba')) / pamt.rolling(window_period, min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_umd_std'%window_period_name)
        second_data = (pamt*rit_up).rolling(window_period, min_periods=1).skew() - (pamt*rit_down).rolling(window_period, min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_umd_skew'%window_period_name)
        second_data = (pamt*rit_up).rolling(window_period, min_periods=1).kurt() - (pamt*rit_down).rolling(window_period, min_periods=1).kurt()
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_umd_kurt'%window_period_name)
        second_data = (pamt.rolling(window_period,min_periods=1).quantile(0.2) - pamt.rolling(window_period,min_periods=1).quantile(0.05)) / (
            pamt.rolling(window_period,min_periods=1).quantile(0.95) - pamt.rolling(window_period,min_periods=1).quantile(0.05) + 1e-6)
        SEC_FACTOR.divid_save_factor(second_data, 'pamt_%s_quan20_scale'%window_period_name)

    return

def calc_daily_factor_all6(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']
    num_5s = all_fea_dict['num_5s']

    pvol = vol_5s / (num_5s + 1e-8)
    pvol_diff = pvol.diff()
    vol_diff = vol_5s.diff()

    rit = np.log(price_5s).diff()*100  #避免量纲过小
    rit_up = (rit > 0).astype(int)
    rit_down = (rit < 0).astype(int)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    for lookback_m in [0.5, 1, 5, 10]:
        window_period = int(lookback_m*12)
        window_period_name = str(int(lookback_m*60))

        second_data = (vol_diff*rit).rolling(window_period, min_periods=1).mean(engine='numba') / (np.abs(vol_diff).rolling(window_period, min_periods=1).mean(engine='numba') + 1e-6)
        SEC_FACTOR.divid_save_factor(second_data, 'pv_%s_disacc'%window_period_name)
        second_data = vol_5s / (vol_5s.rolling(window_period, min_periods=1).median(engine='numba')+1e-6) * (rit_up-rit_down)
        second_data = sigmoid(second_data)
        SEC_FACTOR.divid_save_factor(second_data, 'abv_%s'%window_period_name)
        second_data = vol_5s / (vol_5s.rolling(window_period, min_periods=1).median(engine='numba')+1e-6) * (price_5s/price_5s.rolling(window_period, min_periods=1).median(engine='numba')-1)
        second_data = sigmoid(second_data)
        SEC_FACTOR.divid_save_factor(second_data, 'abv2_%s'%window_period_name)
        second_data = vol_5s / (vol_5s.rolling(window_period, min_periods=1).median(engine='numba')+1e-6) * (rit.diff().rolling(6).median(engine='numba'))
        second_data = sigmoid(second_data)
        SEC_FACTOR.divid_save_factor(second_data, 'abv3_%s'%window_period_name)

    return

def calc_daily_factor_all7(date, all_fea_dict, stock_list, SEC_FACTOR):
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']
    high = all_fea_dict['high']
    low = all_fea_dict['low']

    typ = (high+low+price)/3

    for lookback_m in [0.5, 1, 5, 10]:
        window_period = int(lookback_m*60)
        window_period_name = str(int(lookback_m*60))

        price_quan95 = price.rolling(window_period, min_periods=1).quantile(0.95)
        price_quan05 = price.rolling(window_period, min_periods=1).quantile(0.05)
        high_period = high.rolling(window_period, min_periods=1).max()
        low_period = low.rolling(window_period, min_periods=1).min()
        second_data = (price_quan95 - price_quan05) / (price_quan95 + price_quan05)
        SEC_FACTOR.divid_save_factor(second_data, 'CMO_%s'%window_period_name)
        second_data = (high_period - price) / (high_period - low_period)
        SEC_FACTOR.divid_save_factor(second_data, 'WR_%s'%window_period_name)
        second_data = (price - price.rolling(window_period,min_periods=1).mean(engine='numba').shift(window_period//2)) / price
        SEC_FACTOR.divid_save_factor(second_data, 'DPO_%s'%window_period_name)

        md = np.abs(typ.rolling(window_period,min_periods=1).mean(engine='numba') - price.rolling(window_period, min_periods=1).mean(engine='numba')).rolling(window_period,min_periods=1).mean(engine='numba') + 1e-8
        second_data = (typ-price.rolling(window_period, min_periods=1).mean(engine='numba')) / md
        SEC_FACTOR.divid_save_factor(second_data, 'CCI_%s'%window_period_name)
        ret_ = price.pct_change(window_period)
        second_data = (ret_ - ret_.rolling(window_period, min_periods=1).min()) / (ret_.rolling(window_period,min_periods=1).max()-ret_.rolling(window_period,min_periods=1).min() + 1e-8)
        SEC_FACTOR.divid_save_factor(second_data, 'POS_%s'%window_period_name)
        second_data = (high_period+low_period) / price
        SEC_FACTOR.divid_save_factor(second_data, 'DC_%s'%window_period_name)
    
    return

def get_hft_factor_all(date):
    #初始化
    basic_df = {}
    factor_list = ['30_RSI', '30_wvol_ret', '30_wRV_ret', '30_RSJ', '30_amount_dir', '30_liquity', '60_RSI', '60_wvol_ret', '60_wRV_ret', '60_RSJ', '60_amount_dir', '60_liquity', 
    '300_RSI', '300_wvol_ret', '300_wRV_ret', '300_RSJ', '300_amount_dir', '300_liquity', '600_RSI', '600_wvol_ret', '600_wRV_ret', '600_RSJ', '600_amount_dir', '600_liquity', 'r1_0.5_skew', 
    'r1_0.5_kurt', 'r2_0.5_umd_mean', 'r2_0.5_umd_std', 'r2_0.5_mean', 'r2_0.5_std', 'r2_0.5_skew', 'r2_0.5_kurt', 'r3_0.5_mean', 'r3_0.5_std', 'r3_0.5_skew', 'r3_0.5_kurt', 'r1_1_skew', 
    'r1_1_kurt', 'r2_1_umd_mean', 'r2_1_umd_std', 'r2_1_mean', 'r2_1_std', 'r2_1_skew', 'r2_1_kurt', 'r3_1_mean', 'r3_1_std', 'r3_1_skew', 'r3_1_kurt', 'r1_5_skew', 'r1_5_kurt', 
    'r2_5_umd_mean', 'r2_5_umd_std', 'r2_5_mean', 'r2_5_std', 'r2_5_skew', 'r2_5_kurt', 'r3_5_mean', 'r3_5_std', 'r3_5_skew', 'r3_5_kurt', 'r1_10_skew', 'r1_10_kurt', 'r2_10_umd_mean', 
    'r2_10_umd_std', 'r2_10_mean', 'r2_10_std', 'r2_10_skew', 'r2_10_kurt', 'r3_10_mean', 'r3_10_std', 'r3_10_skew', 'r3_10_kurt', 'r2_30_umd_inv_std', 'r2_30_umd_inv_mean', 'r2_30_umd_skew', 
    'r2_30_umd_kurt', 'r1_30_quan20_scale', 'r2_60_umd_inv_std', 'r2_60_umd_inv_mean', 'r2_60_umd_skew', 'r2_60_umd_kurt', 'r1_60_quan20_scale', 'r2_300_umd_inv_std', 'r2_300_umd_inv_mean', 
    'r2_300_umd_skew', 'r2_300_umd_kurt', 'r1_300_quan20_scale', 'r2_600_umd_inv_std', 'r2_600_umd_inv_mean', 'r2_600_umd_skew', 'r2_600_umd_kurt', 'r1_600_quan20_scale', 'r2_30_umdvol_inv_std', 
    'r2_30_umdvol_inv_mean', 'r2_30_umdvol_skew', 'r2_30_umdvol_kurt', 'r1_30_umdvol_mean', 'r1_30_umdvol_std', 'r1_30_umdvol_skew', 'r1_30_umdvol_kurt', 'r2_60_umdvol_inv_std', 'r2_60_umdvol_inv_mean', 
    'r2_60_umdvol_skew', 'r2_60_umdvol_kurt', 'r1_60_umdvol_mean', 'r1_60_umdvol_std', 'r1_60_umdvol_skew', 'r1_60_umdvol_kurt', 'r2_300_umdvol_inv_std', 'r2_300_umdvol_inv_mean', 'r2_300_umdvol_skew', 
    'r2_300_umdvol_kurt', 'r1_300_umdvol_mean', 'r1_300_umdvol_std', 'r1_300_umdvol_skew', 'r1_300_umdvol_kurt', 'r2_600_umdvol_inv_std', 'r2_600_umdvol_inv_mean', 'r2_600_umdvol_skew', 'r2_600_umdvol_kurt', 
    'r1_600_umdvol_mean', 'r1_600_umdvol_std', 'r1_600_umdvol_skew', 'r1_600_umdvol_kurt', 'pamt_30_std', 'pamt_30_skew', 'pamt_30_kurt', 'pamt_30_umd_mean', 'pamt_30_umd_std', 'pamt_30_umd_skew', 
    'pamt_30_umd_kurt', 'pamt_30_quan20_scale', 'pamt_60_std', 'pamt_60_skew', 'pamt_60_kurt', 'pamt_60_umd_mean', 'pamt_60_umd_std', 'pamt_60_umd_skew', 'pamt_60_umd_kurt', 'pamt_60_quan20_scale', 
    'pamt_300_std', 'pamt_300_skew', 'pamt_300_kurt', 'pamt_300_umd_mean', 'pamt_300_umd_std', 'pamt_300_umd_skew', 'pamt_300_umd_kurt', 'pamt_300_quan20_scale', 'pamt_600_std', 'pamt_600_skew', 'pamt_600_kurt', 
    'pamt_600_umd_mean', 'pamt_600_umd_std', 'pamt_600_umd_skew', 'pamt_600_umd_kurt', 'pamt_600_quan20_scale', 'pv_30_disacc', 'abv_30', 'abv2_30', 'abv3_30', 'pv_60_disacc', 'abv_60', 'abv2_60', 'abv3_60', 
    'pv_300_disacc', 'abv_300', 'abv2_300', 'abv3_300', 'pv_600_disacc', 'abv_600', 'abv2_600', 'abv3_600', 'CMO_30', 'WR_30', 'DPO_30', 'CCI_30', 'POS_30', 'DC_30', 'CMO_60', 'WR_60', 'DPO_60', 'CCI_60', 
    'POS_60', 'DC_60', 'CMO_300', 'WR_300', 'DPO_300', 'CCI_300', 'POS_300', 'DC_300', 'CMO_600', 'WR_600', 'DPO_600', 'CCI_600', 'POS_600', 'DC_600']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取数据
    all_fea_dict = read_sec_table(date)
    pre_date = get_pre_date(date)
    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()
    for fea_name in ['close', 'vol', 'amt', 'num', 'high', 'low']:
        all_fea_dict[fea_name] = all_fea_dict[fea_name].reindex(columns=stock_list)

    #计算5s统计量，存入all_fea_dict中
    sec_5s = [x for x in sec_params.sec_list_all if x%5==0]
    all_fea_dict['price_5s'] = all_fea_dict['close'].loc[sec_5s].reindex(columns=stock_list)
    all_fea_dict['ret_5s'] = all_fea_dict['price_5s'].pct_change().reindex(columns=stock_list)
    all_fea_dict['vol_5s'] = all_fea_dict['vol'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)
    all_fea_dict['amt_5s'] = all_fea_dict['amt'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)
    all_fea_dict['num_5s'] = all_fea_dict['num'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)

    #计算因子
    calc_daily_factor_all1(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all2(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all3(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all4(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all5(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all6(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all7(date, all_fea_dict, stock_list, SEC_FACTOR)

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'ysw_hft_factorall'
    factor_path = r'/home/datamake118/datamake118_base/nas0'
    assert os.path.exists(factor_path)
    
    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name+'_A'))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_hft_factor_all),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor_seperate(factor_group_name, factor_path, divid_num=100)