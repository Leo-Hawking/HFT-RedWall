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

def rolling_wsum(matrix, window_size):
    @jit(nopython=True)
    def rolling_wsum_jit(matrix, window_size):
        num_rows, num_cols = matrix.shape
        result = np.zeros_like(matrix)

        for col in range(num_cols):
            nearsum = 0.0
            nearwsum = 0.0
            for row in range(num_rows):
                nearwsum += window_size * matrix[row, col] - nearsum
                if row >= window_size:
                    nearsum -= matrix[row - window_size, col]
                nearsum += matrix[row, col]
                result[row, col] = nearwsum

        return result

    result = rolling_wsum_jit(matrix, window_size)
    result[np.abs(result) < 1e-9] = 0

    return result

def rolling_wmean(matrix,window_size):
    @jit(nopython=True)
    def rolling_wmean_jit(matrix, window_size):
        num_rows, num_cols = matrix.shape
        result = np.zeros_like(matrix)

        for col in range(num_cols):
            nearsum = 0.0
            nearwsum = 0.0
            for row in range(num_rows):
                nearwsum += window_size * matrix[row, col] - nearsum
                if row >= window_size:
                    nearsum -= matrix[row - window_size, col]
                nearsum += matrix[row, col]
                result[row, col] = nearwsum / min(row + 1, window_size)

        return result

    result = rolling_wmean_jit(matrix, window_size)
    result[np.abs(result) < 1e-9] = 0

    return result

def get_sec_factor1(date, all_fea_dict, stock_list, SEC_FACTOR):
    volume_df = all_fea_dict['vol'].reindex(columns=stock_list)
    apv_df = pd.DataFrame((rolling_mean(volume_df.values,30) / (rolling_mean(volume_df.values,300) + 1e-5)), index=sec_params.sec_list_all, columns=stock_list)
    apv_rank = apv_df.rank(axis=1, pct=True)
    SEC_FACTOR.divid_save_factor(apv_df.reindex(sec_params.sec_list),'APV')
    SEC_FACTOR.divid_save_factor((pd.DataFrame(rolling_mean(apv_rank.values,60),index=sec_params.sec_list_all,columns=stock_list) / \
                apv_rank.rolling(60,min_periods=1).std(engine='numba') + apv_rank.rolling(60,min_periods=1).kurt()).reindex(sec_params.sec_list),'PAPV')
    #开源证券重心因子
    ret_df = all_fea_dict['close'].pct_change().reindex(columns=stock_list).fillna(0)
    is_positive = (ret_df > 0).astype(int)
    ret_positive = ret_df * is_positive
    ret_negative = (ret_df * (1-is_positive)).abs()
    ret_abs = ret_df.abs()
    for period in [60,300,900]:
        norm1 = pd.DataFrame(rolling_mean(ret_abs.values,period),index=sec_params.sec_list_all,columns=stock_list)
        TGA_U = pd.DataFrame(rolling_wsum(ret_positive.values,period),index=sec_params.sec_list_all,columns=stock_list) / ((period+1)*period/2) / (norm1 + 1e-8)
        TGA_D = pd.DataFrame(rolling_wsum(ret_negative.values,period),index=sec_params.sec_list_all,columns=stock_list) / ((period+1)*period/2) / (norm1 + 1e-8)
        SEC_FACTOR.divid_save_factor((TGA_U - TGA_D).reindex(index=sec_params.sec_list),'TGA_diff_%d'%period)
        SEC_FACTOR.divid_save_factor((TGA_U - TGA_D).abs().reindex(index=sec_params.sec_list),'TGA_diffabs_%d'%period)
        del norm1, TGA_U, TGA_D

    return

def calc_daily_factor_all1(date, all_fea_dict, stock_list, SEC_FACTOR):    
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    num = all_fea_dict['num']

    price_5s = all_fea_dict['price_5s']
    ret_5s = all_fea_dict['ret_5s']
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']
    num_5s = all_fea_dict['num_5s'] + 1e-6
    
    pamt=(amt_5s/num_5s).fillna(method='ffill')
    amt_5s=amt_5s+1  # 避免为0放在分母上

    rit = np.log(price_5s).diff()  #避免量纲过小
    rit_up = (rit > 0).astype(int)
    rit_down = (rit < 0).astype(int)
    rit_up_value = rit * rit_up
    rit_down_value = -rit * rit_down
    
    ############################
    
    illiq=np.log(1+(np.abs(rit)/amt_5s).rolling(int(12*5),min_periods=1).mean(engine='numba'))  #用五分钟的平均刻画
    
    avg_vol=vol_5s.rolling(int(12*5),min_periods=1).mean(engine='numba')+1   #用五分钟的平均刻画
    avg_pamt=pamt.rolling(int(12*5),min_periods=1).mean(engine='numba')+1
    
    for lookback_m in [15,30,60]:  #抓短期
        window_period = int(lookback_m/5)
        second_data = illiq* ((rit_up*vol_5s).rolling(window_period,min_periods=1).mean(engine='numba')-(rit_down*vol_5s).rolling(window_period,min_periods=1).mean(engine='numba'))/avg_vol * 1e6
        SEC_FACTOR.divid_save_factor(second_data, '%s_w_liquity_exvolchg'%lookback_m)
        
        second_data = illiq* ((rit_up*pamt).rolling(window_period,min_periods=1).mean(engine='numba')-(rit_down*pamt).rolling(window_period,min_periods=1).mean(engine='numba'))/avg_pamt * 1e6 
        SEC_FACTOR.divid_save_factor(second_data, '%s_w_liquity_expamtchg'%lookback_m)
        
    for lookback_m in [30,60,300]:  #刻画相对长期两个方向的流动性因子
        window_period = int(lookback_m/5)
        up_illiq=np.log(1+(rit_up_value/amt_5s).rolling(window_period,min_periods=4).mean(engine='numba'))
        down_illiq=np.log(1+(rit_down_value/amt_5s).rolling(window_period,min_periods=4).mean(engine='numba'))
        
        SEC_FACTOR.divid_save_factor(up_illiq.loc[sec_params.sec_list] * 1e6,'%s_liquidity_p'%lookback_m)
        
        SEC_FACTOR.divid_save_factor(down_illiq.loc[sec_params.sec_list] * 1e6,'%s_liquidity_n'%lookback_m)
        
        second_data = up_illiq-down_illiq
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list] * 1e6,'%s_liquidity_pmn'%lookback_m)

    return

def calc_daily_factor_all2(date, all_fea_dict, stock_list, SEC_FACTOR):
    #######################
    #基础统计量板块
    #######################
    ##秒级基础数据
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']
    high = all_fea_dict['high']
    low = all_fea_dict['low']
    open_price = all_fea_dict['open']

    ##5s级基础数据，根据rolling对量、额、成交笔数进行聚合,适合不需要很细粒度的计算，注意此时5min的回看维度是60/5*5=60
    price_5s = all_fea_dict['price_5s']
    rit = np.log(price_5s).diff()
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']
    num_5s = all_fea_dict['num_5s'] + 1e-6
    high_5s = high.rolling(5).max().loc[sec_params.sec_list]
    low_5s = low.rolling(5).min().loc[sec_params.sec_list]
    open_5s = open_price.shift(4).loc[sec_params.sec_list]

    vwap_5s = all_fea_dict['vwap_5s']
    price_diff = np.abs(price_5s-open_5s)+1e-6
    middle_price = (price_5s+open_5s)/2

    rit_up_value = rit * (rit > 0).astype(int)
    rit_down_value = -rit * (rit < 0).astype(int)
    
    #计算因子
    up_dev = (1-(high_5s-price_5s)/price_diff).clip(0,1)**2*100  #平方以增加偏离惩罚
    down_dev = (1-(price_5s-low_5s)/price_diff).clip(0,1)**2*100

    up_grav = (1-(vwap_5s-middle_price)/price_diff).clip(0,2)**2*100 #允许反向增加权重
    down_grav = (1-(middle_price-vwap_5s)/price_diff).clip(0,2)**2*100

    avg_price = price_5s.rolling(12*5,min_periods=4).mean(engine='numba')

    for lookback_m in [15,30,60,300]:
        window_period = int(lookback_m/5)
        second_data = rit.diff().rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list] * 1e3, '%s_r_diff_mean'%lookback_m)

        if lookback_m > 15:
            second_data = rit.diff().rolling(window_period,min_periods=1).skew()
            SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_r_diff_skew'%lookback_m)

        #上涨区间内上影线大于收盘价，则扣分，因此上涨区间对因子贡献最强时必须收盘价为最高价
        second_data = (rit_up_value*up_dev).rolling(window_period,min_periods=1).mean(engine='numba')-(rit_down_value*down_dev).rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_pure_chg'%lookback_m)

        second_data = (rit_up_value*up_grav).rolling(window_period,min_periods=1).mean(engine='numba')-(rit_down_value*down_grav).rolling(window_period,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_gravity_chg'%lookback_m)

        second_data = (vwap_5s-middle_price).rolling(window_period,min_periods=1).mean(engine='numba')/avg_price
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list] * 1e3, '%s_gravity_dev'%lookback_m)

    return

def calc_daily_factor_all3(date, all_fea_dict, stock_list, SEC_FACTOR):
    #######################
    #基础统计量板块
    #######################
    ##秒级基础数据
    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']
    high = all_fea_dict['high']
    low = all_fea_dict['low']
    open_price = all_fea_dict['open']

    ##5s级基础数据，根据rolling对量、额、成交笔数进行聚合,适合不需要很细粒度的计算，注意此时5min的回看维度是60/5*5=60
    price_5s = all_fea_dict['price_5s']
    rit = np.log(price_5s).diff()
    vol_5s = all_fea_dict['vol_5s']
    amt_5s = all_fea_dict['amt_5s']
    num_5s = all_fea_dict['num_5s'] + 1e-6
    high_5s = high.rolling(5).max().loc[sec_params.sec_list]
    low_5s = low.rolling(5).min().loc[sec_params.sec_list]

    ############################
    high_diff=np.sign(high_5s.diff())
    low_diff = np.sign(low_5s.diff())
    all_diff=(high_diff+low_diff)/2
    
    all_diff_sign=all_diff.copy()
    all_diff_sign[np.abs(all_diff_sign)!=1]=np.nan
    all_diff_sign=all_diff_sign.fillna(method='ffill').diff().fillna(0)  #除了完全不被包含的都用前值填充
    
    all_diff_sign2=all_diff.copy().replace(0,np.nan).fillna(method='ffill').diff().fillna(0)    #保留原始值，只将完全被包含的用前值填充
    
    mid_hl=(high_5s+low_5s).diff()
    avg_midhl= np.abs(mid_hl).rolling(int(12*5),min_periods=1).mean(engine='numba')+1e-6

    for hl in [5,20,80]: #halflife，长期的
        window_period = int(hl/5)
        ewm_sign= all_diff_sign.ewm(halflife=window_period,min_periods=1).mean(engine='numba')
        ewm_sign2= all_diff_sign2.ewm(halflife=window_period,min_periods=1).mean()

        second_data = ewm_sign
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign_2diff_ewm_mean'%hl)

        second_data = ewm_sign2
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign2_2diff_ewm_mean'%hl)

        for lookback_m in [15,30,60]:  
            second_data = np.abs(ewm_sign)*ewm_sign.shift(1)/(all_diff_sign.rolling(int(lookback_m/5),min_periods=2).std(engine='numba')+1e-6)
            second_data = second_data.loc[sec_params.sec_list] * {15:1e-2,30:1e-1,60:1}[lookback_m] * {5:1e-1,20:1,80:1}[hl]
            SEC_FACTOR.divid_save_factor(second_data, '%s_%s_hl_sign_2diff_ewm_mean_divstd_sp'%(hl,lookback_m))

            second_data = np.abs(ewm_sign2)*ewm_sign2.shift(1)/(all_diff_sign2.rolling(int(lookback_m/5),min_periods=2).std(engine='numba')+1e-6)
            second_data = second_data.loc[sec_params.sec_list] * {15:1e-2,30:1e-1,60:1}[lookback_m] * {5:1e-1,20:1,80:1}[hl]
            SEC_FACTOR.divid_save_factor(second_data, '%s_%s_hl_sign2_2diff_ewm_mean_divstd_sp'%(hl,lookback_m))

    for lookback_m in [15,30,60,150]:  
        window_period = int(lookback_m/5)
        second_data = all_diff.rolling(window_period,min_periods=1).mean(engine='numba')  #不带量纲
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign_1diff_mean'%lookback_m)
        
        second_data = all_diff_sign.rolling(window_period,min_periods=1).mean(engine='numba')  #不带量纲
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign_2diff_mean'%lookback_m)
        
        second_data = all_diff_sign2.rolling(window_period,min_periods=1).mean(engine='numba')  #不带量纲
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign2_2diff_mean'%lookback_m)
        
        lwm_sign= pd.DataFrame(rolling_wmean(all_diff_sign.values,window_period),index=sec_params.sec_list,columns=stock_list)
        lwm_sign2= pd.DataFrame(rolling_wmean(all_diff_sign2.values,window_period),index=sec_params.sec_list,columns=stock_list)

        second_data = lwm_sign
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign_2diff_lwm_mean'%lookback_m) 

        second_data = lwm_sign2
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign2_2diff_lwm_mean'%lookback_m)

        second_data = np.abs(lwm_sign)*lwm_sign.shift(1)/(all_diff_sign.rolling(window_period,min_periods=1).std(engine='numba')+1e-6)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign_2diff_lwm_mean_divstd_sp'%lookback_m)

        second_data = np.abs(lwm_sign2)*lwm_sign2.shift(1)/(all_diff_sign2.rolling(window_period,min_periods=1).std(engine='numba')+1e-6)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_sign2_2diff_lwm_mean_divstd_sp'%lookback_m)
         
        #理想序列是000000020或者0000000-20，因此需要标准差尽可能小，并且平均值尽量大
        
        second_data = mid_hl.diff().rolling(window_period,min_periods=1).mean(engine='numba')/avg_midhl
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_middle_2diff_mean'%lookback_m)
        
        second_data = mid_hl.rolling(window_period,min_periods=1).mean(engine='numba')/avg_midhl
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_hl_middle_1diff_mean'%lookback_m)

    return

################
#和一般动量因子差异点主要源于归一化/做对比的是昨日成交量，注意到之前有类似的构造方法，分母使用的是过去一段时间的统计量；
#增加一些指数平均算子，缓解退化问题
################
def calc_daily_factor_all4(date, all_fea_dict, stock_list, SEC_FACTOR):
    #读取昨日数据
    pre_date = get_pre_date(date)
    pre_fea_dict = read_sec_table(pre_date)
    for fea_name in ['close', 'vol', 'amt', 'num', 'high', 'low']:
        pre_fea_dict[fea_name] = pre_fea_dict[fea_name].reindex(columns=stock_list)

    price = all_fea_dict['close']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    num = all_fea_dict['num']

    pre_vol = pre_fea_dict['vol']
    pre_amt = pre_fea_dict['amt']
    pre_num = pre_fea_dict['num']

    ##5s级基础数据，根据rolling对量、额、成交笔数进行聚合,适合不需要很细粒度的计算，注意此时5min的回看维度是60/5*5=60
    sec_5s = [x for x in sec_params.sec_list_all if x%5==0]
    price_5s = all_fea_dict['price_5s']
    pre_price_5s = pre_fea_dict['close'].loc[sec_5s].reindex(columns=stock_list).fillna(method='ffill')

    rit = np.log(price_5s).diff()
    pre_rit = np.log(pre_price_5s).diff()

    td_rit_up = (rit > 0).astype(int)
    td_rit_down = (rit < 0).astype(int)

    pre_rit_up = (pre_rit > 0).astype(int)
    pre_rit_down = (pre_rit < 0).astype(int)

    vol_5s = all_fea_dict['vol_5s']
    num_5s = all_fea_dict['num_5s'] + 1e-6

    pre_vol_5s = pre_fea_dict['vol'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)
    pre_num_5s = pre_fea_dict['num'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)

    #计算因子
    pre_vol = pre_vol_5s.mean() + 1e-8
    pre_up_vol = (pre_vol_5s*pre_rit_up).mean() + 1e-8
    pre_down_vol = (pre_vol_5s*pre_rit_down).mean() + 1e-8

    for lookback_m in [5,20,80]:
        window_period = int(lookback_m/5)
        ewm_td_up_vol=(vol_5s*td_rit_up).ewm(halflife=window_period).mean()
        ewm_td_down_vol=(vol_5s*td_rit_down).ewm(halflife=window_period).mean()
        second_data = (ewm_td_up_vol - ewm_td_down_vol) / pre_vol
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_ewm_umd_vol_ratio'%lookback_m)

    for lookback_m in [30,60,300]:
        window_period = int(lookback_m/5)
        td_up_vol=(vol_5s*td_rit_up).rolling(window_period,min_periods=1).mean()
        td_down_vol=(vol_5s*td_rit_down).rolling(window_period,min_periods=1).mean()

        second_data = td_up_vol / pre_up_vol - 1
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_up_vol_ratio'%lookback_m)
        second_data = td_down_vol / pre_down_vol - 1
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_down_vol_ratio'%lookback_m)
        second_data = (td_up_vol - td_down_vol) / pre_vol
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], '%s_umd_vol_ratio'%lookback_m)

    pre_num = pre_num_5s.mean() + 1e-8
    pre_up_num = (pre_num_5s*pre_rit_up).mean() + 1e-8
    pre_down_num = (pre_num_5s*pre_rit_down).mean() + 1e-8

    for lookback_m in [5,20,80]:
        window_period = int(lookback_m/5)
        ewm_td_up_num=(num_5s*td_rit_up).ewm(halflife=window_period).mean()
        ewm_td_down_num=(num_5s*td_rit_down).ewm(halflife=window_period).mean()
        second_data=(ewm_td_up_num-ewm_td_down_num)/pre_num
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_ewm_umd_num_ratio'%lookback_m)
    
    for lookback_m in [30,60,300]:
        window_period = int(lookback_m/5)
        td_up_num = (num_5s*td_rit_up).rolling(window_period,min_periods=1).mean(engine='numba')
        td_down_num = (num_5s*td_rit_down).rolling(window_period,min_periods=1).mean(engine='numba')

        second_data = td_up_num / pre_up_num - 1
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_up_num_ratio'%lookback_m)
        second_data = td_down_num / pre_down_num - 1
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_down_num_ratio'%lookback_m)
        second_data = (td_up_num - td_down_num) / pre_num
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_umd_num_ratio'%lookback_m)

    pre_rit_std=pre_rit.std() + 1e-8
    pre_up_rit=(pre_rit*pre_rit_up).std() + 1e-8
    pre_down_rit=(pre_rit*pre_rit_down).std() + 1e-8

    for lookback_m in [5,20,80]:
        window_period = int(lookback_m/5)
        ewm_td_up_rit = (rit*td_rit_up).ewm(halflife=window_period).std()
        ewm_td_down_rit = (rit*td_rit_down).ewm(halflife=window_period).std()
        second_data = (ewm_td_up_rit - ewm_td_down_rit) / pre_rit_std
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_ewm_umd_rit_vol_ratio'%lookback_m)

    for lookback_m in [30,60,300]:
        window_period = int(lookback_m/5)
        td_up_rit = (rit*td_rit_up).rolling(window_period,min_periods=1).std(engine='numba')
        td_down_rit = (rit*td_rit_down).rolling(window_period,min_periods=1).std(engine='numba')

        second_data = td_up_rit / pre_up_rit - 1
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_up_rit_vol_ratio'%lookback_m)
        second_data = td_down_rit / pre_down_rit - 1
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_down_rit_vol_ratio'%lookback_m)
        second_data = (td_up_rit - td_down_rit) / pre_rit_std
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list].fillna(0), '%s_umd_rit_vol_ratio'%lookback_m)

    return

def get_hft_factor_sec2(date):
    #初始化
    basic_df = {}
    factor_list = ['APV', 'PAPV', 'TGA_diff_60', 'TGA_diffabs_60', 'TGA_diff_300', 'TGA_diffabs_300', 'TGA_diff_900', 'TGA_diffabs_900', '15_w_liquity_exvolchg', '15_w_liquity_expamtchg', \
    '30_w_liquity_exvolchg', '30_w_liquity_expamtchg', '60_w_liquity_exvolchg', '60_w_liquity_expamtchg', '30_liquidity_p', '30_liquidity_n', '30_liquidity_pmn', '60_liquidity_p', \
    '60_liquidity_n', '60_liquidity_pmn', '300_liquidity_p', '300_liquidity_n', '300_liquidity_pmn', '15_r_diff_mean', '15_pure_chg', '15_gravity_chg', '15_gravity_dev', '30_r_diff_mean', \
    '30_r_diff_skew', '30_pure_chg', '30_gravity_chg', '30_gravity_dev', '60_r_diff_mean', '60_r_diff_skew', '60_pure_chg', '60_gravity_chg', '60_gravity_dev', '300_r_diff_mean', '300_r_diff_skew', \
    '300_pure_chg', '300_gravity_chg', '300_gravity_dev', '5_hl_sign_2diff_ewm_mean', '5_hl_sign2_2diff_ewm_mean', '5_15_hl_sign_2diff_ewm_mean_divstd_sp', '5_15_hl_sign2_2diff_ewm_mean_divstd_sp', \
    '5_30_hl_sign_2diff_ewm_mean_divstd_sp', '5_30_hl_sign2_2diff_ewm_mean_divstd_sp', '5_60_hl_sign_2diff_ewm_mean_divstd_sp', '5_60_hl_sign2_2diff_ewm_mean_divstd_sp', '20_hl_sign_2diff_ewm_mean', \
    '20_hl_sign2_2diff_ewm_mean', '20_15_hl_sign_2diff_ewm_mean_divstd_sp', '20_15_hl_sign2_2diff_ewm_mean_divstd_sp', '20_30_hl_sign_2diff_ewm_mean_divstd_sp', '20_30_hl_sign2_2diff_ewm_mean_divstd_sp', \
    '20_60_hl_sign_2diff_ewm_mean_divstd_sp', '20_60_hl_sign2_2diff_ewm_mean_divstd_sp', '80_hl_sign_2diff_ewm_mean', '80_hl_sign2_2diff_ewm_mean', '80_15_hl_sign_2diff_ewm_mean_divstd_sp', \
    '80_15_hl_sign2_2diff_ewm_mean_divstd_sp', '80_30_hl_sign_2diff_ewm_mean_divstd_sp', '80_30_hl_sign2_2diff_ewm_mean_divstd_sp', '80_60_hl_sign_2diff_ewm_mean_divstd_sp', \
    '80_60_hl_sign2_2diff_ewm_mean_divstd_sp', '15_hl_sign_1diff_mean', '15_hl_sign_2diff_mean', '15_hl_sign2_2diff_mean', '15_hl_sign_2diff_lwm_mean', '15_hl_sign2_2diff_lwm_mean', \
    '15_hl_sign_2diff_lwm_mean_divstd_sp', '15_hl_sign2_2diff_lwm_mean_divstd_sp', '15_hl_middle_2diff_mean', '15_hl_middle_1diff_mean', '30_hl_sign_1diff_mean', '30_hl_sign_2diff_mean', \
    '30_hl_sign2_2diff_mean', '30_hl_sign_2diff_lwm_mean', '30_hl_sign2_2diff_lwm_mean', '30_hl_sign_2diff_lwm_mean_divstd_sp', '30_hl_sign2_2diff_lwm_mean_divstd_sp', '30_hl_middle_2diff_mean', \
    '30_hl_middle_1diff_mean', '60_hl_sign_1diff_mean', '60_hl_sign_2diff_mean', '60_hl_sign2_2diff_mean', '60_hl_sign_2diff_lwm_mean', '60_hl_sign2_2diff_lwm_mean', '60_hl_sign_2diff_lwm_mean_divstd_sp', \
    '60_hl_sign2_2diff_lwm_mean_divstd_sp', '60_hl_middle_2diff_mean', '60_hl_middle_1diff_mean', '150_hl_sign_1diff_mean', '150_hl_sign_2diff_mean', '150_hl_sign2_2diff_mean', '150_hl_sign_2diff_lwm_mean', \
    '150_hl_sign2_2diff_lwm_mean', '150_hl_sign_2diff_lwm_mean_divstd_sp', '150_hl_sign2_2diff_lwm_mean_divstd_sp', '150_hl_middle_2diff_mean', '150_hl_middle_1diff_mean', '5_ewm_umd_vol_ratio', \
    '20_ewm_umd_vol_ratio', '80_ewm_umd_vol_ratio', '30_up_vol_ratio', '30_down_vol_ratio', '30_umd_vol_ratio', '60_up_vol_ratio', '60_down_vol_ratio', '60_umd_vol_ratio', '300_up_vol_ratio', \
    '300_down_vol_ratio', '300_umd_vol_ratio', '5_ewm_umd_num_ratio', '20_ewm_umd_num_ratio', '80_ewm_umd_num_ratio', '30_up_num_ratio', '30_down_num_ratio', '30_umd_num_ratio', '60_up_num_ratio', \
    '60_down_num_ratio', '60_umd_num_ratio', '300_up_num_ratio', '300_down_num_ratio', '300_umd_num_ratio', '5_ewm_umd_rit_vol_ratio', '20_ewm_umd_rit_vol_ratio', '80_ewm_umd_rit_vol_ratio', \
    '30_up_rit_vol_ratio', '30_down_rit_vol_ratio', '30_umd_rit_vol_ratio', '60_up_rit_vol_ratio', '60_down_rit_vol_ratio', '60_umd_rit_vol_ratio', '300_up_rit_vol_ratio', '300_down_rit_vol_ratio', \
    '300_umd_rit_vol_ratio']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()

    #读取数据
    all_fea_dict = read_sec_table(date)
    pre_date = get_pre_date(date)
    for fea_name in ['close', 'vol', 'amt', 'num', 'high', 'low']:
        all_fea_dict[fea_name] = all_fea_dict[fea_name].reindex(columns=stock_list)

    #计算5s统计量，存入all_fea_dict中
    sec_5s = [x for x in sec_params.sec_list_all if x%5==0]
    all_fea_dict['price_5s'] = all_fea_dict['close'].loc[sec_5s].reindex(columns=stock_list).fillna(method='ffill')
    all_fea_dict['ret_5s'] = all_fea_dict['price_5s'].pct_change().reindex(columns=stock_list)
    all_fea_dict['vol_5s'] = all_fea_dict['vol'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)
    all_fea_dict['amt_5s'] = all_fea_dict['amt'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)
    all_fea_dict['num_5s'] = all_fea_dict['num'].rolling(5).sum(engine='numba').loc[sec_5s].reindex(columns=stock_list)
    all_fea_dict['vwap_5s'] = (all_fea_dict['amt_5s'] / all_fea_dict['vol_5s']).replace([0,np.inf,-np.inf],np.nan).fillna(method='ffill')

    #计算因子
    get_sec_factor1(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all1(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all2(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all3(date, all_fea_dict, stock_list, SEC_FACTOR)
    calc_daily_factor_all4(date, all_fea_dict, stock_list, SEC_FACTOR)

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'hft_sec_factor2'
    factor_path = r'/home/datamake118/datamake118_base/nas6/sec_factor3'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_hft_factor_sec2),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)