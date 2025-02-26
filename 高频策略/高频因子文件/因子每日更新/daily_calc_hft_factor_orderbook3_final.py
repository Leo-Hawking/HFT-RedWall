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
warnings.filterwarnings('ignore')

####################################
#函数区
####################################
# @njit
# def calculate_kurtosis(data):
#     if data.size == 0:
#         return np.nan
#     # 计算均值和标准差
#     mean = np.mean(data)
#     std = np.std(data)
#     # 计算峰度
#     kurtosis = np.sum((data - mean)**4) / (len(data) * std**4) - 3 if std > 0 else np.nan
#     return kurtosis

# @njit
# def calculate_skewness(rates):
#     if rates.size == 0:
#         return np.nan
#     mean = np.mean(rates)
#     diffs = rates - mean
#     adjusted_sum = np.sum(diffs ** 3)
#     m3 = adjusted_sum / rates.size
#     m2 = np.sum(diffs ** 2) / rates.size
#     return m3 / (m2 ** (3 / 2)) if m2 > 0 else np.nan

@njit
def calculate_skewness_pandas(data):
    if data.size == 0:
        return np.nan
    n = data.size
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.sqrt(np.sum(data ** 2) / (n - 1) - (n / (n - 1)) * (mean ** 2))
    skew = n * (np.sum(data ** 3) - 3 * mean * np.sum(data ** 2) + 2 * n * mean ** 3) / ((n - 1) * (n - 2) * std ** 3) if std > 0 else np.nan

    return skew

@njit
def calculate_kurtosis_pandas(data):
    if data.size == 0:
        return np.nan
    n = data.size
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.sqrt(np.sum(data ** 2) / (n - 1) - (n / (n - 1)) * (mean ** 2))
    a = (n + 1) * n / ((n - 1) * (n - 2) * (n - 3))
    b = (n - 1) ** 2 / ((n - 2) * (n - 3))
    kurt = a * ((data ** 4).sum() - 4 * mean * (data ** 3).sum() + 6 * (mean ** 2) * (data ** 2).sum() - 3 * n * mean ** 4) / (std ** 4) - 3 * b if std > 0 else np.nan
    
    return kurt

@njit
def simple_linear_regression(x, y):
    """
    简单线性回归，返回斜率和截距。
    x: 自变量，档位索引
    y: 因变量，价格或深度
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x*y)
    
    # 计算斜率和截距
    beta = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    intercept = (sum_y - beta*sum_x) / n
    # 计算残差
    residuals = y - (beta * x + intercept)
    ss_res = np.sum(residuals**2)  # 残差平方和
    ss_tot = np.sum((y - np.mean(y))**2)  # 总平方和
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return np.array([beta, intercept, ss_res, r_squared])

@njit
def calculate_regression_metrics_to_each_row(arr):
    """
    对每一行数据计算线性回归参数。
    arr: 二维数组，每一行是一个观测值序列。
    """
    results = np.empty((arr.shape[0], 4), dtype=np.float64)  # 用于存储斜率和截距
    x = np.arange(1, arr.shape[1] + 1)  # 档位索引，作为自变量
    for i in range(arr.shape[0]):
        results[i] = simple_linear_regression(x, arr[i])
    return results


@njit
def apply_to_each_row(arr, func, output_num = 1):
    # 创建一个足够大的数组来存储每行的输出
    results = np.empty((arr.shape[0], output_num), dtype=np.float64)
    for i in range(arr.shape[0]):
        # 分别赋值
        output = func(arr[i])
        results[i] = output
    return results

# 自定义的寻峰函数，认为间隔足够大的大值是峰
@njit
def simplified_find_peaks(row, height, distance):
    peaks = []
    last_peak_index = -distance  # 初始化为负距离，以确保第一个元素可以被考虑
    for i in range(len(row)):
        if row[i] > height and (i - last_peak_index) >= distance:
            peaks.append(i)
            last_peak_index = i
    return peaks

@njit
def process_row(row):
    global_mean = np.mean(row)
    global_std = np.std(row)
    threshold = global_mean + 2 * global_std
    peaks = simplified_find_peaks(row, threshold, 5)
    
    length = len(peaks)
    mean_depth = np.mean(row[np.array(peaks)]) if length > 0 else 0
    result = np.array([float(length), mean_depth, length * mean_depth])
    return result

warnings.filterwarnings('ignore')

def calc_orderbook_factor3(date):
    #初始化
    factor_list = [
        # 买卖一档价差力量
        'buy_gap_power_immediate', 'sell_gap_power_immediate', 'buy_gap_power', 'sell_gap_power',
        # 买卖穿透档位点数
        'sell_cross_point_1', 'buy_cross_point_1', 'sell_cross_point_5', 'buy_cross_point_5', 'sell_cross_point_15', 'buy_cross_point_15', 'sell_cross_point_30', 'buy_cross_point_30',
        # 买卖穿透力量
        'sell_cross_1', 'buy_cross_1', 'sell_cross_5', 'buy_cross_5', 'sell_cross_15', 'buy_cross_15', 'sell_cross_30', 'buy_cross_30',
        # 买单和卖单最大交易量所在的档位
        'deepest_buy_level', 'deepest_sell_level', 
        # 前五个和前十个买卖档位中最大交易量所在的档位
        'deepest_level_5', 'deepest_level_10', 
        # 全部买卖档位中最大交易量所在的档位
        'deepest_level_all', 
        # 基于给定阈值和距离计算的行情峰值数量、平均深度及总深度
        'peak_num', 'peak_mean_depth', 'peak_depth_sum', 
        # 卖单交易量的线性回归参数
        'sell_volume_beta', 'sell_volume_intercept', 'sell_volume_residuals', 'sell_volume_r2', 
        # 买单交易量的线性回归参数
        'buy_volume_beta', 'buy_volume_intercept', 'buy_volume_residuals', 'buy_volume_r2', 
        # 卖价和买价的线性回归参数
        'sell_price_beta', 'sell_price_intercept', 'sell_price_residuals', 'sell_prices_r2', 
        'buy_price_beta', 'buy_price_intercept', 'buy_price_residuals', 'buy_prices_r2', 
        # 全部价格的线性回归参数
        'all_price_beta', 'all_price_intercept', 'all_price_residuals', 'all_prices_r2', 
        # 第二大交易量与最大交易量之间的差异
        'second_deepest_diff', 
        # 最大交易量与平均交易量的比率
        'mean_diff_ratio', 
        # 交易深度的统计参数
        'depth_std', 'depth_mean', 'depth_kurt', 'depth_skew', 
        # 买卖价格差的统计参数
        'buy_price_diff_std', 'sell_price_diff_std', 'buy_price_diff_mean', 'sell_price_diff_mean', 
        'buy_price_diff_kurt', 'sell_price_diff_kurt', 'buy_price_diff_skew', 'sell_price_diff_skew'
    ]
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()
    pre_close = daily_support['preclose'].copy()
    pre_vol = daily_support['vol5'].copy() /5 / 14400  # 前5日的总成交量，这块应除以5再除以秒数

    #分股票计算因子
    factor_dict = {}
    for factor in factor_list:
        factor_dict[factor] = {}
    lob_code_list = sorted(os.listdir(os.path.join(sec_params.lob_path,date)))
    lob_code_list = sorted([int(x[:6]) for x in lob_code_list])

    for stock in stock_list:
        if stock in lob_code_list and not np.isnan(pre_close[stock]) and not np.isnan(pre_vol[stock]) and pre_close[stock]!=0 and pre_vol[stock]!=0:
            try:
                lobdata_c = read_divid_lobdata(date, stock, expand_tick=20)
            except:
                print(f'Error in reading lobdata: {date}-{stock}')
                continue
            lobdata_c['mid_price'] = ((lobdata_c['sp_1'] + lobdata_c['bp_1']) / 2).shift(1)
            lobdata_c['mid_price_roll'] = lobdata_c['mid_price'].rolling(10, min_periods=1).mean()
            #计算买卖gap力量
            factor_dict['buy_gap_power_immediate'][stock] = ((lobdata_c['mid_price'] - lobdata_c['bp_1']) / pre_close[stock]).reindex(sec_params.sec_list)
            factor_dict['sell_gap_power_immediate'][stock] = ((lobdata_c['sp_1'] - lobdata_c['mid_price']) / pre_close[stock]).reindex(sec_params.sec_list)
            factor_dict['buy_gap_power'][stock] = ((lobdata_c['mid_price_roll'] - lobdata_c['bp_1']) / pre_close[stock]).reindex(sec_params.sec_list)
            factor_dict['sell_gap_power'][stock] = ((lobdata_c['sp_1'] - lobdata_c['mid_price_roll']) / pre_close[stock]).reindex(sec_params.sec_list)

            #计算买卖穿透力量
            lobdata_shift = lobdata_c.shift(1)
            lobdata_shift['bp_1_now'] = lobdata_c['bp_1']
            lobdata_shift['sp_1_now'] = lobdata_c['sp_1']
            sell_cross_df = (lobdata_shift[['bp_%s'%str(i) for i in range(1,6)]].T > lobdata_shift['bp_1_now']).astype(int).T
            buy_cross_df = (lobdata_shift[['sp_%s'%str(i) for i in range(1,6)]].T < lobdata_shift['sp_1_now']).astype(int).T
            lobdata_shift['sell_cross_point'] = sell_cross_df.sum(axis=1)
            lobdata_shift['buy_cross_point'] = buy_cross_df.sum(axis=1)
            lobdata_shift['sell_cross'] = (sell_cross_df * lobdata_shift[['bv_%s'%str(i) for i in range(1,6)]].values).sum(axis=1) / pre_vol[stock]
            lobdata_shift['buy_cross'] = (buy_cross_df * lobdata_shift[['sv_%s'%str(i) for i in range(1,6)]].values).sum(axis=1) / pre_vol[stock]
            for period in [1,5,15,30]:
                factor_dict['sell_cross_point_%s'%period][stock] = lobdata_shift['sell_cross_point'].rolling(period, min_periods=1).sum().reindex(sec_params.sec_list)
                factor_dict['buy_cross_point_%s'%period][stock] = lobdata_shift['buy_cross_point'].rolling(period, min_periods=1).sum().reindex(sec_params.sec_list)
                factor_dict['sell_cross_%s'%period][stock] = lobdata_shift['sell_cross'].rolling(period, min_periods=1).sum().reindex(sec_params.sec_list)
                factor_dict['buy_cross_%s'%period][stock] = lobdata_shift['buy_cross'].rolling(period, min_periods=1).sum().reindex(sec_params.sec_list)
            
            lobdata_c = lobdata_c.reindex(sec_params.sec_list)
            buy_volumes = lobdata_c[['bv_%s'%str(i) for i in range(1,21)]].values / pre_vol[stock]
            sell_volumes = lobdata_c[['sv_%s'%str(i) for i in range(1,21)]].values / pre_vol[stock]
            buy_prices = lobdata_c[['bp_%s'%str(i) for i in range(1,21)]].values / pre_close[stock]
            sell_prices = lobdata_c[['sp_%s'%str(i) for i in range(1,21)]].values / pre_close[stock]

            # 计算买方和卖方最深的档位
            factor_dict['deepest_buy_level'][stock] = np.argmax(buy_volumes, axis=1) + 1
            factor_dict['deepest_sell_level'][stock] = np.argmax(sell_volumes, axis=1) + 1

            # 合并买方和卖方深度，计算五档和十档的最深值
            combined_volumes = np.hstack([buy_volumes, sell_volumes])
            combined_prices = np.hstack((sell_prices[:,::-1], buy_prices))
            factor_dict['deepest_level_5'][stock] = np.argmax(np.hstack([buy_volumes[:,:5],sell_volumes[:,:5]]), axis=1) + 1
            factor_dict['deepest_level_10'][stock] = np.argmax(np.hstack([buy_volumes[:,:10],sell_volumes[:,:10]]), axis=1) + 1
            factor_dict['deepest_level_all'][stock] = np.argmax(combined_volumes, axis=1) + 1

            # 计算峰值因子
            peak_factor = apply_to_each_row(combined_volumes, process_row, 3)
            factor_dict['peak_num'][stock] = peak_factor[:,0]
            factor_dict['peak_mean_depth'][stock] = peak_factor[:,1]
            factor_dict['peak_depth_sum'][stock] = peak_factor[:,2]

            #计算回归因子
            sell_volume_reg_factor = calculate_regression_metrics_to_each_row(sell_volumes[:,::-1])
            factor_dict['sell_volume_beta'][stock] = sell_volume_reg_factor[:,0]
            factor_dict['sell_volume_intercept'][stock] = sell_volume_reg_factor[:,1]
            factor_dict['sell_volume_residuals'][stock] = sell_volume_reg_factor[:,2] / 1e4
            factor_dict['sell_volume_r2'][stock] = sell_volume_reg_factor[:,3]
            buy_volume_reg_factor = calculate_regression_metrics_to_each_row(buy_volumes)
            factor_dict['buy_volume_beta'][stock] = buy_volume_reg_factor[:,0]
            factor_dict['buy_volume_intercept'][stock] = buy_volume_reg_factor[:,1]
            factor_dict['buy_volume_residuals'][stock] = buy_volume_reg_factor[:,2] / 1e4
            factor_dict['buy_volume_r2'][stock] = buy_volume_reg_factor[:,3]

            sell_price_reg_factor = calculate_regression_metrics_to_each_row(sell_prices[:,::-1])
            factor_dict['sell_price_beta'][stock] = sell_price_reg_factor[:,0]
            factor_dict['sell_price_intercept'][stock] = sell_price_reg_factor[:,1]
            factor_dict['sell_price_residuals'][stock] = sell_price_reg_factor[:,2] * 1e4
            factor_dict['sell_price_residuals'][stock][np.abs(factor_dict['sell_price_residuals'][stock]) < 1e-8] = 0 
            factor_dict['sell_prices_r2'][stock] = sell_price_reg_factor[:,3]
            buy_price_reg_factor = calculate_regression_metrics_to_each_row(buy_prices)
            factor_dict['buy_price_beta'][stock] = buy_price_reg_factor[:,0]
            factor_dict['buy_price_intercept'][stock] = buy_price_reg_factor[:,1]
            factor_dict['buy_price_residuals'][stock] = buy_price_reg_factor[:,2] * 1e4
            factor_dict['buy_price_residuals'][stock][np.abs(factor_dict['buy_price_residuals'][stock]) < 1e-8] = 0 
            factor_dict['buy_prices_r2'][stock] = buy_price_reg_factor[:,3]
            all_price_reg_factor = calculate_regression_metrics_to_each_row(combined_prices)
            factor_dict['all_price_beta'][stock] = all_price_reg_factor[:,0]
            factor_dict['all_price_intercept'][stock] = all_price_reg_factor[:,1]
            factor_dict['all_price_residuals'][stock] = all_price_reg_factor[:,2] * 1e4
            factor_dict['all_price_residuals'][stock][np.abs(factor_dict['all_price_residuals'][stock]) < 1e-8] = 0 
            factor_dict['all_prices_r2'][stock] = all_price_reg_factor[:,3]

            #第二大交易量与最大交易量之间的差异
            sorted_volumes = np.sort(combined_volumes, axis=1)
            factor_dict['second_deepest_diff'][stock] = (sorted_volumes[:, -1] - sorted_volumes[:, -2])

            # Calculate mean difference ratio
            mean_volumes = np.mean(combined_volumes, axis=1, where=combined_volumes!=0)
            factor_dict['mean_diff_ratio'][stock] = sorted_volumes[:, -1] / mean_volumes

            #计算基础统计量
            factor_dict['depth_std'][stock] = np.std(combined_volumes, axis=1)
            factor_dict['depth_mean'][stock] = np.mean(combined_volumes, axis=1)
            factor_dict['depth_kurt'][stock] = apply_to_each_row(combined_volumes, calculate_kurtosis_pandas).squeeze(axis=1)
            factor_dict['depth_skew'][stock] = apply_to_each_row(combined_volumes, calculate_skewness_pandas).squeeze(axis=1)

            #计算价差
            buy_price_diffs = np.diff(buy_prices, axis=1)
            sell_price_diffs = np.diff(sell_prices, axis=1)
            factor_dict['buy_price_diff_std'][stock] = np.std(buy_price_diffs, axis=1)
            factor_dict['buy_price_diff_std'][stock][np.abs(factor_dict['buy_price_diff_std'][stock]) < 1e-8] = 0
            factor_dict['sell_price_diff_std'][stock] = np.std(sell_price_diffs, axis=1)
            factor_dict['sell_price_diff_std'][stock][np.abs(factor_dict['sell_price_diff_std'][stock]) < 1e-8] = 0
            factor_dict['buy_price_diff_mean'][stock] = np.mean(buy_price_diffs, axis=1)
            factor_dict['sell_price_diff_mean'][stock] = np.mean(sell_price_diffs, axis=1)
            factor_dict['buy_price_diff_kurt'][stock] = apply_to_each_row(buy_price_diffs, calculate_kurtosis_pandas).squeeze(axis=1)
            factor_dict['sell_price_diff_kurt'][stock] = apply_to_each_row(sell_price_diffs, calculate_kurtosis_pandas).squeeze(axis=1)
            factor_dict['buy_price_diff_skew'][stock] = apply_to_each_row(buy_price_diffs, calculate_skewness_pandas).squeeze(axis=1)
            factor_dict['sell_price_diff_skew'][stock] = apply_to_each_row(sell_price_diffs, calculate_skewness_pandas).squeeze(axis=1)

        else:
            for factor in factor_list:
                factor_dict[factor][stock] = np.array([np.nan]*len(sec_params.sec_list))
    target_sec_list = sec_params.sec_list
    for factor in factor_list:
        df = pd.DataFrame(factor_dict[factor], index=target_sec_list)
        SEC_FACTOR.divid_save_factor(df, factor)
    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_orderbook_factorall3'
    factor_path = r'/home/datamake118/datamake118_base/nas8/sec_factor4'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,calc_orderbook_factor3),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)