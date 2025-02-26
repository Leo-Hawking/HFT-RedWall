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

def get_trans_basic_factor(date):
    #初始化
    basic_df = {}
    factor_list = ['trans_buy_aratio_10','trans_buy_aratio_300','trans_buy_aratio_5','trans_buy_aratio_60','trans_buy_aratio_900','trans_buy_vratio_10',\
            'trans_buy_vratio_300','trans_buy_vratio_5','trans_buy_vratio_60','trans_buy_vratio_900','trans_buyvol_concen2_10','trans_buyvol_concen2_300','trans_buyvol_concen2_5',\
            'trans_buyvol_concen2_60','trans_buyvol_concen2_900','trans_buyvol_concen_10','trans_buyvol_concen_300','trans_buyvol_concen_5','trans_buyvol_concen_60','trans_buyvol_concen_900',\
            'trans_bvol_strength_10','trans_bvol_strength_300','trans_bvol_strength_5','trans_bvol_strength_60','trans_bvol_strength_900','trans_svol_strength_10','trans_svol_strength_300',\
            'trans_svol_strength_5','trans_svol_strength_60','trans_svol_strength_900','trans_vol_concen_10','trans_vol_concen_300','trans_vol_concen_5','trans_vol_concen_60',\
            'trans_vol_concen_900','trans_vol_kurt_10','trans_vol_kurt_300','trans_vol_kurt_5','trans_vol_kurt_60','trans_vol_kurt_900','trans_vol_skew_10','trans_vol_skew_300',\
            'trans_vol_skew_5','trans_vol_skew_60','trans_vol_skew_900','trans_vol_std_10','trans_vol_std_300','trans_vol_std_5','trans_vol_std_60','trans_vol_std_900','trans_vwap_bratio_10',\
            'trans_vwap_bratio_300','trans_vwap_bratio_5','trans_vwap_bratio_60','trans_vwap_bratio_900','trans_vwap_sratio_10','trans_vwap_sratio_300','trans_vwap_sratio_5',\
            'trans_vwap_sratio_60','trans_vwap_sratio_900']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)
    period_list = [5,10,60,300,900]

    # 读取数据
    trans_data = read_trans_data(date, cancel=False)
    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()
    new_idx = pd.MultiIndex.from_product([stock_list, sec_params.sec_list_all], names=['code', 'second'])

    # Calculate additional columns in a vectorized manner
    trans_data['tradeVolume_d100'] = trans_data['tradeVolume'] / 100
    trans_data['tradeVolume_d100_2'] = trans_data['tradeVolume_d100'] ** 2

    # Create boolean masks
    mask_S = trans_data['bsFlag'] == 'S'
    mask_B = trans_data['bsFlag'] == 'B'
    # Use np.where to efficiently set values based on conditions
    trans_data['amount_S'] = np.where(mask_S, trans_data['tradeAmt'], 0)
    trans_data['amount_B'] = np.where(mask_B, trans_data['tradeAmt'], 0)
    trans_data['tradeVolume_S'] = np.where(mask_S, trans_data['tradeVolume'], 0)
    trans_data['tradeVolume_B'] = np.where(mask_B, trans_data['tradeVolume'], 0)
    trans_data['tradeVolume_d100_2_S'] = np.where(mask_S, trans_data['tradeVolume_d100_2'], 0)
    trans_data['tradeVolume_d100_2_B'] = np.where(mask_B, trans_data['tradeVolume_d100_2'], 0)

    # # Create trans_type directly without using intermediate column
    # tmp_trans['trans_type'] = np.where(mask_big, 'big', 'normal')

    # Group by and sum in a single step for better performance
    tmp_second = trans_data.groupby(['code', 'second']).agg(
        tradeVolume_d100=('tradeVolume_d100', 'sum'),
        tradeVolume_d100_2=('tradeVolume_d100_2', 'sum'),
        tradeVolume_B=('tradeVolume_B', 'sum'),
        tradeVolume_S=('tradeVolume_S', 'sum'),
        amount=('tradeAmt', 'sum'),
        amount_B=('amount_B', 'sum'),
        amount_S=('amount_S', 'sum'),
        tradeVolume_d100_2_B=('tradeVolume_d100_2_B', 'sum'),
        tradeVolume_d100_2_S=('tradeVolume_d100_2_S', 'sum'),
    ).reindex(new_idx).fillna(0).reset_index()

    # Calculate vwap in a vectorized manner
    tmp_second['vwap'] = tmp_second['amount'] / (tmp_second['tradeVolume_d100'] * 100)
    tmp_second['vwap'] = tmp_second['vwap'].replace([-np.inf,np.inf],np.nan)

    # Forward-fill missing values in vwap column within each code group
    tmp_second['vwap'] = tmp_second.groupby('code')['vwap'].fillna(method='ffill')

    #合并基础数据
    basic_df['tradeVolume_d100'] = tmp_second.pivot_table(index='second', columns='code', values='tradeVolume_d100',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['tradeVolume_d100_2'] = tmp_second.pivot_table(index='second', columns='code', values='tradeVolume_d100_2',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['tradeVolume_B'] = tmp_second.pivot_table(index='second', columns='code', values='tradeVolume_B',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['tradeVolume_S'] = tmp_second.pivot_table(index='second', columns='code', values='tradeVolume_S',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list)
    basic_df['amount'] = tmp_second.pivot_table(index='second', columns='code', values='amount',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['amount_B'] = tmp_second.pivot_table(index='second', columns='code', values='amount_B',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['amount_S'] = tmp_second.pivot_table(index='second', columns='code', values='amount_S',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['tradeVolume_d100_2_B'] = tmp_second.pivot_table(index='second', columns='code', values='tradeVolume_d100_2_B',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['tradeVolume_d100_2_S'] = tmp_second.pivot_table(index='second', columns='code', values='tradeVolume_d100_2_S',fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)

    #计算因子
    for period in period_list:
        vol_d100_sum = pd.DataFrame(rolling_sum(basic_df['tradeVolume_d100'].values, period),index=sec_params.sec_list_all, columns=stock_list)
        vol_buy_sum = pd.DataFrame(rolling_sum(basic_df['tradeVolume_B'].values, period),index=sec_params.sec_list_all, columns=stock_list)
        vol_sell_sum = pd.DataFrame(rolling_sum(basic_df['tradeVolume_S'].values, period),index=sec_params.sec_list_all, columns=stock_list)
        amt_sum = pd.DataFrame(rolling_sum(basic_df['amount'].values, period),index=sec_params.sec_list_all, columns=stock_list)
        amt_buy_sum = pd.DataFrame(rolling_sum(basic_df['amount_B'].values, period),index=sec_params.sec_list_all, columns=stock_list)
        amt_sell_sum = pd.DataFrame(rolling_sum(basic_df['amount_S'].values, period),index=sec_params.sec_list_all, columns=stock_list)
        total_vwap = np.log(amt_sum / (vol_d100_sum * 100).replace(0,np.nan))
        total_vwap = total_vwap.replace([-np.inf,np.inf],np.nan)
        buy_vwap = np.log(amt_buy_sum / (vol_buy_sum).replace(0,np.nan))
        buy_vwap = buy_vwap.replace([-np.inf,np.inf],np.nan)
        sell_vwap = np.log(amt_sell_sum / (vol_sell_sum).replace(0,np.nan))
        sell_vwap = sell_vwap.replace([-np.inf,np.inf],np.nan)
        #计算订单统计量指标
        SEC_FACTOR.divid_save_factor(basic_df['tradeVolume_d100'].rolling(period,min_periods=1).std(engine='numba'), 'trans_vol_std_%s'%str(period))
        SEC_FACTOR.divid_save_factor(basic_df['tradeVolume_d100'].rolling(period,min_periods=1).skew(), 'trans_vol_skew_%s'%str(period))
        SEC_FACTOR.divid_save_factor(basic_df['tradeVolume_d100'].rolling(period,min_periods=1).kurt(), 'trans_vol_kurt_%s'%str(period))
        #计算主买比例因子
        SEC_FACTOR.divid_save_factor(vol_buy_sum / (vol_d100_sum * 100 + 1e-8), 'trans_buy_vratio_%s'%str(period))
        SEC_FACTOR.divid_save_factor(amt_buy_sum / (amt_sum + 1e-8), 'trans_buy_aratio_%s'%str(period))
        #计算集中度因子
        SEC_FACTOR.divid_save_factor(pd.DataFrame(rolling_sum(basic_df['tradeVolume_d100_2'].values, period),index=sec_params.sec_list_all, columns=stock_list) / (vol_d100_sum ** 2 + 1e-8), 'trans_vol_concen_%s'%str(period))
        SEC_FACTOR.divid_save_factor(pd.DataFrame(rolling_sum(basic_df['tradeVolume_d100_2_B'].values, period),index=sec_params.sec_list_all, columns=stock_list) / (vol_d100_sum ** 2 + 1e-8), 'trans_buyvol_concen_%s'%str(period))
        SEC_FACTOR.divid_save_factor(pd.DataFrame(rolling_sum(basic_df['tradeVolume_d100_2_B'].values, period),index=sec_params.sec_list_all, columns=stock_list) / ((vol_buy_sum / 100) ** 2 + 1e-8), 'trans_buyvol_concen2_%s'%str(period))
        #计算价差偏离因子
        SEC_FACTOR.divid_save_factor(total_vwap - buy_vwap, 'trans_vwap_bratio_%s'%str(period))
        SEC_FACTOR.divid_save_factor(sell_vwap - total_vwap, 'trans_vwap_sratio_%s'%str(period))
        #买入卖出金额强度
        SEC_FACTOR.divid_save_factor(basic_df['tradeVolume_B'].rolling(period,min_periods=1).mean(engine='numba') / (basic_df['tradeVolume_B'].rolling(period,min_periods=1).std(engine='numba') + 1e-8), 'trans_bvol_strength_%s'%str(period))
        SEC_FACTOR.divid_save_factor(basic_df['tradeVolume_S'].rolling(period,min_periods=1).mean(engine='numba') / (basic_df['tradeVolume_S'].rolling(period,min_periods=1).std(engine='numba') + 1e-8), 'trans_svol_strength_%s'%str(period))

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_trans_basic'
    factor_path = r'/home/datamake118/datamake118_base/nas2/sec_factor2'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_trans_basic_factor),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)