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

def get_order_factor1(date):
    #初始化
    basic_df = {}
    factor_list = ['exlarge_buy_ratio2_1', 'exlarge_buy_ratio2_10', 'exlarge_buy_ratio2_300', 'exlarge_buy_ratio2_60', 'exlarge_buy_ratio3_1', 'exlarge_buy_ratio3_10', 'exlarge_buy_ratio3_300', \
    'exlarge_buy_ratio3_60', 'exlarge_buy_ratio_1', 'exlarge_buy_ratio_10', 'exlarge_buy_ratio_300', 'exlarge_buy_ratio_60', 'exlarge_ratio_1', 'exlarge_ratio_10', 'exlarge_ratio_300', 'exlarge_ratio_60', \
    'large_buy_ratio2_1', 'large_buy_ratio2_10', 'large_buy_ratio2_300', 'large_buy_ratio2_60', 'large_buy_ratio3_1', 'large_buy_ratio3_10', 'large_buy_ratio3_300', 'large_buy_ratio3_60', \
    'large_buy_ratio_1', 'large_buy_ratio_10', 'large_buy_ratio_300', 'large_buy_ratio_60', 'large_ratio_1', 'large_ratio_10', 'large_ratio_300', 'large_ratio_60', 'medium_buy_ratio2_1', 'medium_buy_ratio2_10', \
    'medium_buy_ratio2_300', 'medium_buy_ratio2_60', 'medium_buy_ratio3_1', 'medium_buy_ratio3_10', 'medium_buy_ratio3_300', 'medium_buy_ratio3_60', 'medium_buy_ratio_1', 'medium_buy_ratio_10', \
    'medium_buy_ratio_300', 'medium_buy_ratio_60', 'medium_ratio_1', 'medium_ratio_10', 'medium_ratio_300', 'medium_ratio_60', 'small_buy_ratio2_1', 'small_buy_ratio2_10', 'small_buy_ratio2_300', \
    'small_buy_ratio2_60', 'small_buy_ratio3_1', 'small_buy_ratio3_10', 'small_buy_ratio3_300', 'small_buy_ratio3_60', 'small_buy_ratio_1', 'small_buy_ratio_10', 'small_buy_ratio_300', 'small_buy_ratio_60', \
    'small_ratio_1', 'small_ratio_10', 'small_ratio_300', 'small_ratio_60']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)
    period_list = [1,10,60,300]

    #读取数据
    order_data = read_order_data(date, cancel=False)

    #计算总体数据
    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()
    order_data['exlarge_order'] = (order_data['orderAmt'] >= 1000000).astype(int)
    order_data['large_order'] = ((order_data['orderAmt'] >= 200000) & (order_data['orderAmt'] < 1000000)).astype(int)
    order_data['medium_order'] = ((order_data['orderAmt'] >= 40000) & (order_data['orderAmt'] < 200000)).astype(int)
    order_data['small_order'] = (order_data['orderAmt'] < 40000).astype(int)
    order_data['total_buyAmt'] = order_data['orderAmt'] * (order_data['functionCode']=='B').astype(int)
    grouped_data = order_data.groupby(['code', 'second']).agg(totalamt_sum=('orderAmt', 'sum'),
                                                            totalamt_buy_sum=('total_buyAmt', 'sum')).reset_index()
    # Create pivot tables
    basic_df['totalamt_sum'] = grouped_data.pivot_table(index='second', columns='code', values='totalamt_sum',
                                                        fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)
    basic_df['totalamt_buy_sum'] = grouped_data.pivot_table(index='second', columns='code', values='totalamt_buy_sum',
                                                            fill_value=0).reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)

    for category in ['exlarge','large','medium','small']:
        order_data[f'{category}Amt'] = order_data[f'{category}_order'] * order_data['orderAmt']
        order_data[f'{category}_buyAmt'] = order_data[f'{category}Amt'] * (order_data['functionCode']=='B').astype(int)

        for key in [f'{category}amt_sum',f'{category}amt_buy_sum']:
            keyto = {f'{category}amt_sum':f'{category}Amt',f'{category}amt_buy_sum':f'{category}_buyAmt'}[key]
            basic_df[key] = order_data.groupby(['code', 'second'])[keyto].sum().reset_index() \
                .pivot_table(index='second', columns='code', values=keyto) \
                .reindex(index=sec_params.sec_list_all, columns=stock_list).fillna(0)

        for period in period_list:
            totalsum_roll = rolling_sum(basic_df['totalamt_sum'].values, period) + 1e-5
            totalsum_buy_roll = rolling_sum(basic_df['totalamt_buy_sum'].values, period) + 1e-5
            targetsum_roll = rolling_sum(basic_df[f'{category}amt_sum'].values, period)
            targetsum_buy_roll = rolling_sum(basic_df[f'{category}amt_buy_sum'].values, period)
            # Calculate factors and add to the dictionary
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_roll / totalsum_roll,index=sec_params.sec_list_all, columns=stock_list), f'{category}_ratio_{period}')
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_buy_roll / totalsum_roll,index=sec_params.sec_list_all, columns=stock_list), f'{category}_buy_ratio_{period}')
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_buy_roll / (targetsum_roll + 1e-5),index=sec_params.sec_list_all, columns=stock_list), f'{category}_buy_ratio2_{period}')
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_buy_roll / totalsum_buy_roll,index=sec_params.sec_list_all, columns=stock_list), f'{category}_buy_ratio3_{period}')

        del order_data[f'{category}Amt'], order_data[f'{category}_buyAmt']
        del totalsum_roll, totalsum_buy_roll, targetsum_roll, targetsum_buy_roll

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_order_ls1'
    factor_path = r'/home/datamake118/datamake118_base/nas2/sec_factor2'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_order_factor1),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)