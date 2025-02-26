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

def get_order_basic_factor(date):
    #初始化
    basic_df = {}
    factor_list = ['order_bpvcorr_5', 'order_buyamt_ratio_10', 'order_buyamt_ratio_300', 'order_buyamt_ratio_5', 'order_buyamt_ratio_60', 'order_buyamt_ratio_900', \
    'order_buyvol_concen_10', 'order_buyvol_concen_300', 'order_buyvol_concen_5', 'order_buyvol_concen_60', 'order_buyvol_concen_900', 'order_bvol_strength_10', \
    'order_bvol_strength_300', 'order_bvol_strength_5', 'order_bvol_strength_60', 'order_bvol_strength_900', 'order_spvcorr_5', 'order_svol_strength_10', 'order_svol_strength_300', \
    'order_svol_strength_5', 'order_svol_strength_60', 'order_svol_strength_900', 'order_tpvcorr_5', 'order_vol_concen_10', 'order_vol_concen_300', 'order_vol_concen_5', 'order_vol_concen_60', \
    'order_vol_concen_900', 'order_vol_std_10', 'order_vol_std_300', 'order_vol_std_5', 'order_vol_std_60', 'order_vol_std_900', 'order_vwap_bratio_10', 'order_vwap_bratio_300', 'order_vwap_bratio_5', \
    'order_vwap_bratio_60', 'order_vwap_bratio_900', 'order_vwap_sratio_10', 'order_vwap_sratio_300', 'order_vwap_sratio_5', 'order_vwap_sratio_60', 'order_vwap_sratio_900']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)
    period_list = [5,10,60,300,900]

    #读取数据
    order_data = read_order_data(date, cancel=False)

    #计算总体数据
    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()
    buy = (order_data['functionCode'] == 'B').astype(int)
    order_data['total_buyAmt'] = order_data['orderAmt'] * buy
    order_data['vol_d100'] = order_data['orderVolume'] / 100
    order_data['vol_d100_sq'] = order_data['vol_d100'] ** 2
    order_data['vol_buy_d100'] = order_data['vol_d100'] * buy
    order_data['vol_buy_d100_sq'] = order_data['vol_d100_sq'] * buy
    grouped_data = order_data.groupby(['code', 'second']).agg(totalamt_sum=('orderAmt', 'sum'),
                                                            totalamt_buy_sum=('total_buyAmt', 'sum'),
                                                            vol_d100=('vol_d100', 'sum'),
                                                            vol_d100_sq=('vol_d100_sq', 'sum'),
                                                            vol_buy_d100=('vol_buy_d100', 'sum'),
                                                            vol_buy_d100_sq=('vol_buy_d100_sq','sum')).reset_index()
    # Create pivot tables
    basic_df['totalamt_sum'] = grouped_data.pivot_table(index='second', columns='code', values='totalamt_sum',
                                                        fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['totalamt_buy_sum'] = grouped_data.pivot_table(index='second', columns='code', values='totalamt_buy_sum',
                                                            fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['totalamt_sell_sum'] = basic_df['totalamt_sum'] - basic_df['totalamt_buy_sum']
    basic_df['vol_d100_sum'] = grouped_data.pivot_table(index='second', columns='code', values='vol_d100',
                                                        fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['vol_d100_sq_sum'] = grouped_data.pivot_table(index='second', columns='code', values='vol_d100_sq',
                                                            fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['vol_buy_d100_sum'] = grouped_data.pivot_table(index='second', columns='code', values='vol_buy_d100',
                                                            fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['vol_buy_d100_sq_sum'] = grouped_data.pivot_table(index='second', columns='code', values='vol_buy_d100_sq',
                                                                fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['vol_sell_d100_sum'] = basic_df['vol_d100_sum'] - basic_df['vol_buy_d100_sum']

    for period in period_list:
        #每秒订单的统计量指标
        vol_d100_mean = basic_df['vol_d100_sum'].rolling(period,min_periods=1).mean()
        vol_d100_sum = basic_df['vol_d100_sum'].rolling(period,min_periods=1).sum()
        amt_sum = basic_df['totalamt_sum'].rolling(period,min_periods=1).sum() + 1e-5
        total_vwap = np.log(basic_df['totalamt_sum'].rolling(period,min_periods=1).sum() / (vol_d100_sum * 100).replace(0,np.nan))
        buy_vwap = np.log(basic_df['totalamt_buy_sum'].rolling(period,min_periods=1).sum() / (basic_df['vol_buy_d100_sum'].rolling(period,min_periods=1).sum() * 100).replace(0,np.nan))
        sell_vwap = np.log(basic_df['totalamt_sell_sum'].rolling(period,min_periods=1).sum() / (basic_df['vol_sell_d100_sum'].rolling(period,min_periods=1).sum() * 100).replace(0,np.nan))
        #成交量变异系数
        SEC_FACTOR.divid_save_factor(basic_df['vol_d100_sum'].rolling(period,min_periods=1).std(engine='numba') / (vol_d100_mean + 1e-5), 'order_vol_std_%s'%str(period))
        #金额占比
        SEC_FACTOR.divid_save_factor(basic_df['totalamt_buy_sum'].rolling(period,min_periods=1).sum(engine='numba') / amt_sum, 'order_buyamt_ratio_%s'%str(period))
        #集中度因子
        SEC_FACTOR.divid_save_factor(basic_df['vol_d100_sq_sum'].rolling(period,min_periods=1).sum(engine='numba') / (vol_d100_sum ** 2 + 1e-5), 'order_vol_concen_%s'%str(period))
        SEC_FACTOR.divid_save_factor(basic_df['vol_buy_d100_sq_sum'].rolling(period,min_periods=1).sum(engine='numba') / (vol_d100_sum ** 2 + 1e-5), 'order_buyvol_concen_%s'%str(period))
        #价差偏离因子
        SEC_FACTOR.divid_save_factor(total_vwap - buy_vwap, 'order_vwap_bratio_%s'%str(period))
        SEC_FACTOR.divid_save_factor(sell_vwap - total_vwap, 'order_vwap_sratio_%s'%str(period))
        #买入卖出金额强度
        SEC_FACTOR.divid_save_factor(basic_df['vol_buy_d100_sum'].rolling(period,min_periods=1).mean(engine='numba') / \
                    (basic_df['vol_buy_d100_sum'].rolling(period,min_periods=1).std(engine='numba') + 1e-5), 'order_bvol_strength_%s'%str(period))
        SEC_FACTOR.divid_save_factor(basic_df['vol_sell_d100_sum'].rolling(period,min_periods=1).mean(engine='numba') / \
                    (basic_df['vol_sell_d100_sum'].rolling(period,min_periods=1).std(engine='numba') + 1e-5), 'order_svol_strength_%s'%str(period))

        if period == 5:
            tpvcorr = (total_vwap * basic_df['vol_d100_sum']).rolling(period).mean(engine='numba') - \
                                (total_vwap.rolling(period).mean(engine='numba') * basic_df['vol_d100_sum'].rolling(period).mean(engine='numba'))
            tpvcorr = tpvcorr / (total_vwap.rolling(period).std(engine='numba') * basic_df['vol_d100_sum'].rolling(period).std(engine='numba') + 1e-5)
            SEC_FACTOR.divid_save_factor(tpvcorr * 100, 'order_tpvcorr_%s'%str(period))
            bpvcorr = (buy_vwap * basic_df['vol_buy_d100_sum']).rolling(period).mean(engine='numba') - \
                                (buy_vwap.rolling(period).mean(engine='numba') * basic_df['vol_buy_d100_sum'].rolling(period).mean(engine='numba'))
            bpvcorr = bpvcorr / (buy_vwap.rolling(period).std(engine='numba') * basic_df['vol_buy_d100_sum'].rolling(period).std(engine='numba') + 1e-5)
            SEC_FACTOR.divid_save_factor(bpvcorr * 100, 'order_bpvcorr_%s'%str(period))
            spvcorr = (sell_vwap * basic_df['vol_sell_d100_sum']).rolling(period).mean(engine='numba') - \
                                (sell_vwap.rolling(period).mean(engine='numba') * basic_df['vol_sell_d100_sum'].rolling(period).mean(engine='numba'))
            spvcorr = spvcorr / (sell_vwap.rolling(period).std(engine='numba') * basic_df['vol_sell_d100_sum'].rolling(period).std(engine='numba') + 1e-5)
            SEC_FACTOR.divid_save_factor(spvcorr * 100, 'order_spvcorr_%s'%str(period))

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_order_basic'
    factor_path = r'/home/datamake118/datamake118_base/nas2/sec_factor2'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_order_basic_factor),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)