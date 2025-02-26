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

def get_sec3_factor_all(date):
    #初始化
    factor_list = ['twap_APRR_10', 'vwap_APRR_10', 'twap_APRR_30', 'vwap_APRR_30', 'twap_APRR_90', 'vwap_APRR_90', 'twap_APRR_300', 'vwap_APRR_300', 'twap_APRR_900', 'vwap_APRR_900', \
        'RVJN_10', 'SRVJ_10', 'RVJN_30', 'SRVJ_30', 'RVJN_90', 'SRVJ_90', 'RVJN_300', 'SRVJ_300', 'RVJN_900', 'SRVJ_900', 'trend_ratio_10', 'trend_ratio_30', 'trend_ratio_90', \
        'trend_ratio_300', 'trend_ratio_900', 'ATR_10', 'ATR_30', 'ATR_90', 'ATR_300', 'ATR_900', 'top_10_percent_product_10', 'top_10_percent_product_30', 'top_10_percent_product_90', \
        'top_10_percent_product_300', 'top_10_percent_product_900', 'frighten_mean_10', 'frighten_std_10', 'FP_beta_10', 'frighten_mean_30', 'frighten_std_30', 'FP_beta_30', 'frighten_mean_90', \
        'frighten_std_90', 'FP_beta_90', 'frighten_mean_300', 'frighten_std_300', 'FP_beta_300', 'frighten_mean_900', 'frighten_std_900', 'FP_beta_900', 'consistent_UPCV_10', \
        'consistent_DPCV_10', 'consistent_UPCV_30', 'consistent_DPCV_30', 'consistent_UPCV_90', 'consistent_DPCV_90', 'consistent_UPCV_300', 'consistent_DPCV_300', \
        'consistent_UPCV_900', 'consistent_DPCV_900', 'negative_illiquidity_10', 'negative_illiquidity_30', 'negative_illiquidity_90', 'negative_illiquidity_300', \
        'negative_illiquidity_900', 'fuzzy_correlation_10', 'fuzzy_correlation_30', 'fuzzy_correlation_90', 'fuzzy_correlation_300', 'fuzzy_correlation_900', \
        'price_time_slope_10', 'price_time_slope_30', 'price_time_slope_90', 'price_time_slope_300', 'price_time_slope_900', 'mx_drawdown_10', \
        'mx_drawdown_30', 'mx_drawdown_90', 'mx_drawdown_300', 'mx_drawdown_900']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取数据
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()

    all_fea_dict = read_sec_table(date)
    pre_date = get_pre_date(date)
    for fea_name in ['close','open', 'vol', 'amt', 'num', 'high', 'low']:
        all_fea_dict[fea_name] = all_fea_dict[fea_name].reindex(columns=stock_list)
    
    get_sec3_factor_part1(all_fea_dict,SEC_FACTOR)
    # get_sec3_factor_part2(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part3(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part4(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part5(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part6(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part7(all_fea_dict,SEC_FACTOR,daily_support)
    get_sec3_factor_part8(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part9(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part10(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part11(all_fea_dict,SEC_FACTOR)
    # get_sec3_factor_part12(all_fea_dict,SEC_FACTOR)
    get_sec3_factor_part13(all_fea_dict,SEC_FACTOR)
    return SEC_FACTOR

def get_sec3_factor_part1(all_fea_dict,SEC_FACTOR):
    close = all_fea_dict['close']
    high = all_fea_dict['high']
    low = all_fea_dict['low']
    vol = all_fea_dict['vol']
    amt = all_fea_dict['amt']
    
    for period in [10,30,90,300,900]:
        rolling_high = high.rolling(period,min_periods=1).max()
        rolling_low = low.rolling(period,min_periods=1).min()
        vwap = amt.rolling(period,min_periods=1).sum(engine='numba') / vol.rolling(period,min_periods=1).sum(engine='numba')
        vwap = vwap.replace([0,np.inf,-np.inf],np.nan).fillna(method='ffill')

        second_data = (close.rolling(period,min_periods=1).mean(engine='numba') - rolling_low) / (rolling_high - rolling_low)
        SEC_FACTOR.divid_save_factor(second_data, 'twap_APRR_%s'%period)
        second_data = (vwap - rolling_low) / (rolling_high - rolling_low)
        SEC_FACTOR.divid_save_factor(second_data, 'vwap_APRR_%s'%period)
    return

# def get_sec3_factor_part2(all_fea_dict,SEC_FACTOR):
#     #成交占比偏度因子
#     for period in [10,30,90,300,900]:
#         vol_ratio = all_fea_dict['vol'] / all_fea_dict['vol'].rolling(period, min_periods=1).sum(engine='numba')
#         skew = vol_ratio.rolling(period, min_periods=1).skew()
#         SEC_FACTOR.divid_save_factor(skew, f'vol_ratio_skew_{period}')
#         kurt = vol_ratio.rolling(period, min_periods=1).kurt()
#         SEC_FACTOR.divid_save_factor(kurt, f'vol_ratio_kurt_{period}')
#     return

def get_sec3_factor_part3(all_fea_dict,SEC_FACTOR):
    #下行跳跃波动率因子，0611
    #上下行跳跃波动不对称性因子，0615
    #用于计算绝对矩的函数，保留下来但不实际使用
    '''
    import numpy as np
    from scipy.integrate import quad
    from fractions import Fraction

    # 定义标准正态分布的概率密度函数
    def normal_distribution(x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    # 计算绝对矩
    def absolute_moment(n):
        # 定义积分函数
        integrand = lambda x: np.abs(x)**n * normal_distribution(x)
        # 进行数值积分
        result, _ = quad(integrand, -np.inf, np.inf)
        return result

    absolute_moment(2/3) # = 0.8023805748753304
    '''
    ret = all_fea_dict['close'].pct_change()
    ret2 = ret**2
    ret_abs = ret.abs()
    ret_tmp = ret.abs()**(2/3) * (ret.shift(1).abs()**(2/3)) * (ret.shift(3).abs()**(2/3))
    ret_up = (ret > 0).astype(int)
    ret_down = (ret < 0).astype(int)
    miu = (0.8023805748753304)**(-3)
    for period in [10,30,90,300,900]:
        RS_up = (ret2 * ret_up).rolling(period, min_periods=1).sum(engine='numba')
        RS_down = (ret2 * ret_down).rolling(period, min_periods=1).sum(engine='numba')
        IV = miu * (ret_tmp).rolling(period-2, min_periods=1).sum(engine='numba') / 2
        RVJP = RS_up - IV
        RVJP = (RVJP * (RVJP > 0).astype(int)) * 1e6
        RVJN = RS_down - IV
        RVJN = (RVJN * (RVJN > 0).astype(int)) * 1e6
        SEC_FACTOR.divid_save_factor(RVJN, 'RVJN_%s'%period)
        SEC_FACTOR.divid_save_factor(RVJP-RVJN, 'SRVJ_%s'%period)

    return

def get_sec3_factor_part4(all_fea_dict,SEC_FACTOR):
    #趋势占比因子，0613
    price_diff = all_fea_dict['close'].diff().abs()
    for period in [10,30,90,300,900]:
        factor = all_fea_dict['close'].diff(period)
        factor = factor / price_diff.rolling(period-1, min_periods=1).sum(engine='numba')
        SEC_FACTOR.divid_save_factor(factor, f'trend_ratio_{period}')

    return

def get_sec3_factor_part5(all_fea_dict,SEC_FACTOR):
    #平均真实波幅因子，0614
    #三个矩阵，求每个对应元素的最大值
    def max_matrix(matrix1, matrix2, matrix3):
        return np.maximum(np.maximum(matrix1, matrix2), matrix3)
    TR = max_matrix(all_fea_dict['high'] - all_fea_dict['low'], (all_fea_dict['high'] - all_fea_dict['close']).abs(), (all_fea_dict['low'] - all_fea_dict['close']).abs())
    for period in [10,30,90,300,900]:
        factor = TR.rolling(period, min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(factor, f'ATR_{period}')

    return

def get_sec3_factor_part6(all_fea_dict,SEC_FACTOR):
    #最大涨幅因子，0624
    import numpy as np
    from numba import njit

    ret = all_fea_dict['close'].pct_change().fillna(0)
    @njit
    def top_10_percent_product(data):
        n = data.shape[0]
        k = data.shape[1]
        top_10_idx = {10:1,30:3,90:9,300:30,900:90}[n]
        result = np.zeros(k)
        
        for j in range(k):
            sorted_data = np.sort(data[:, j])
            top_10_data = sorted_data[-top_10_idx:]
            result[j] = np.prod(1 + top_10_data)
        
        return result

    def rolling_top_10_percent_product(price, period):
        n = price.shape[0]
        k = price.shape[1]
        result = np.zeros((n - period + 1, k))
        zeros = np.zeros((period - 1, k))
        
        for i in range(n - period + 1):
            window_data = price[i:i+period]
            result[i] = top_10_percent_product(window_data)
        
        return np.vstack([zeros, result])
    for period in [10,30,90,300,900]:
        factor = pd.DataFrame(rolling_top_10_percent_product(ret.values, period), index=ret.index, columns=ret.columns)
        SEC_FACTOR.divid_save_factor(factor, f'top_10_percent_product_{period}')

    return

def get_sec3_factor_part7(all_fea_dict,SEC_FACTOR,daily_support):
    #注意力衰减-惊恐因子，0627
    #贝塔因子，0628
    mv_pre = daily_support['mv'].copy()
    ret = all_fea_dict['close'].pct_change().fillna(0)
    ret_market = ret @ mv_pre / mv_pre.sum()

    frighten = ((ret.T - ret_market).abs() / (ret.T.abs() + ret_market.abs() + 0.1)).T
    frighten = frighten - (frighten.shift(1) + frighten.shift(2)) / 2
    frighten = frighten * (frighten > 0).astype(int)
    for period in [10,30,90,300,900]:
        tmp = frighten * ret
        mean = tmp.rolling(period, min_periods=1).mean(engine='numba') * 1e6
        std = tmp.rolling(period, min_periods=1).std(engine='numba') * 1e6
        SEC_FACTOR.divid_save_factor(mean, f'frighten_mean_{period}')
        SEC_FACTOR.divid_save_factor(std, f'frighten_std_{period}')

        #贝塔因子
        if period > 1:
            corr = ret.rolling(period, min_periods=1).corr(ret_market)
            ret_std = ret.rolling(period, min_periods=1).std(engine='numba')
            beta = corr * (ret_std.T / ret_market.rolling(period, min_periods=1).std(engine='numba')).T
            SEC_FACTOR.divid_save_factor(beta, f'FP_beta_{period}')
    return

def get_sec3_factor_part8(all_fea_dict,SEC_FACTOR):
    #一致买入交易因子，0629
    close_open = (all_fea_dict['close'] - all_fea_dict['open']).abs()
    high_low = (all_fea_dict['high'] - all_fea_dict['low']).abs()
    alpha = 0.6
    consistent_bar = (close_open > alpha * high_low).astype(int)
    ret = all_fea_dict['close'].pct_change().fillna(0)
    ret_up = (ret > 0).astype(int)
    ret_down = (ret < 0).astype(int)
    for period in [10,30,90,300,900]:
        vol_sum = all_fea_dict['vol'].rolling(period, min_periods=1).sum(engine='numba')
        factor_up = (all_fea_dict['vol'] * consistent_bar * ret_up).rolling(period, min_periods=1).sum(engine='numba')
        factor_down = (all_fea_dict['vol'] * consistent_bar * ret_down).rolling(period, min_periods=1).sum(engine='numba')
        SEC_FACTOR.divid_save_factor(factor_up / vol_sum, f'consistent_UPCV_{period}')
        SEC_FACTOR.divid_save_factor(factor_down / vol_sum, f'consistent_DPCV_{period}')
    return

def get_sec3_factor_part9(all_fea_dict,SEC_FACTOR):
    #负收益非流动性因子
    ret = all_fea_dict['close'].pct_change().fillna(0)
    ret_down_value = (ret * (ret < 0).astype(int)).abs()
    for period in [10,30,90,300,900]:
        factor = ret_down_value.rolling(period, min_periods=1).sum(engine='numba') / (all_fea_dict['amt'].rolling(period, min_periods=1).sum(engine='numba') / 1e6)
        SEC_FACTOR.divid_save_factor(factor, f'negative_illiquidity_{period}')
    return

def get_sec3_factor_part10(all_fea_dict,SEC_FACTOR):
    #模糊关联度因子，0710
    ret = all_fea_dict['close'].pct_change().fillna(0)
    for period in [10,30,90,300,900]:
        tmp = ret.rolling(period, min_periods=1).std(engine='numba') * 1e3
        SEC_FACTOR.divid_save_factor(tmp, f'fuzzy_correlation_{period}')
    return

def get_sec3_factor_part11(all_fea_dict,SEC_FACTOR):
    #成交量潮汐价格变动速率因子，0716
    def count_slope(tmp):
        MIN,MIN_IDX = 0,0
        MAX,MAX_IDX = 0,0
        for i in range(1,len(tmp)):
            if tmp[i] < tmp[MIN_IDX]:
                MIN,MIN_IDX = tmp[i],i
            if tmp[i] > tmp[MAX_IDX]:
                MAX,MAX_IDX = tmp[i],i
        if MAX_IDX == MIN_IDX:
            return 0
        else: return (MAX-MIN)/(MAX_IDX-MIN_IDX)

    for period in [10,30,90,300,900]:
        #求过去price的最快变化速率
        tmp = all_fea_dict['close'].rolling(period,min_periods=5).apply(count_slope, raw=True, engine='numba')
        SEC_FACTOR.divid_save_factor(tmp, f'price_time_slope_{period}')

    return

# def get_sec3_factor_part12(all_fea_dict,SEC_FACTOR):
#     #佳庆离散指标，0717
#     REM = (all_fea_dict['high'] - all_fea_dict['low']).rolling(15, min_periods=1).mean(engine='numba')
#     REM2 = (all_fea_dict['high'] - all_fea_dict['low']).rolling(90, min_periods=1).mean(engine='numba')
#     factor = (REM-REM2) / REM2
#     SEC_FACTOR.divid_save_factor(factor, 'JQ_REM')
#     return

def get_sec3_factor_part13(all_fea_dict,SEC_FACTOR):
    #最大回撤因子，0718
    for period in [10,30,90,300,900]:
        max_price = all_fea_dict['close'].rolling(period, min_periods=1).max(engine='numba')
        factor = ((max_price - all_fea_dict['close']) / max_price) * 1e3
        SEC_FACTOR.divid_save_factor(factor, f'mx_drawdown_{period}')
    return

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_hft_factor_sec3'
    factor_path = r'/home/datamake118/datamake118_base/nas8/sec_factor4'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_sec3_factor_all),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)