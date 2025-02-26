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

def get_trans_factor1(date):
    # 逐笔成交大小单因子计算
    # 初始化
    factor_dict = {}
    basic_df = {}
    factor_list = ['large_buy_ratio2_1','large_buy_ratio2_10','large_buy_ratio2_300','large_buy_ratio2_60','large_buy_ratio3_1','large_buy_ratio3_10','large_buy_ratio3_300','large_buy_ratio3_60',
 'large_buy_ratio_1','large_buy_ratio_10','large_buy_ratio_300','large_buy_ratio_60','large_pd_diff_1','large_pd_diff_10','large_pd_diff_300','large_pd_diff_60','large_pd_ratio_1','large_pd_ratio_10',
 'large_pd_ratio_300','large_pd_ratio_60','large_ratio_1','large_ratio_10','large_ratio_300','large_ratio_60','medium_buy_ratio2_1','medium_buy_ratio2_10','medium_buy_ratio2_300','medium_buy_ratio2_60',
 'medium_buy_ratio3_1','medium_buy_ratio3_10','medium_buy_ratio3_300','medium_buy_ratio3_60','medium_buy_ratio_1','medium_buy_ratio_10','medium_buy_ratio_300','medium_buy_ratio_60','medium_pd_diff_1',
 'medium_pd_diff_10','medium_pd_diff_300','medium_pd_diff_60','medium_pd_ratio_1','medium_pd_ratio_10','medium_pd_ratio_300','medium_pd_ratio_60','medium_ratio_1','medium_ratio_10','medium_ratio_300',
 'medium_ratio_60','small_buy_ratio2_1','small_buy_ratio2_10','small_buy_ratio2_300','small_buy_ratio2_60','small_buy_ratio3_1','small_buy_ratio3_10','small_buy_ratio3_300','small_buy_ratio3_60',
 'small_buy_ratio_1','small_buy_ratio_10','small_buy_ratio_300','small_buy_ratio_60','small_pd_diff_1','small_pd_diff_10','small_pd_diff_300','small_pd_diff_60','small_pd_ratio_1','small_pd_ratio_10',
 'small_pd_ratio_300','small_pd_ratio_60','small_ratio_1','small_ratio_10','small_ratio_300','small_ratio_60']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    # Load data
    trans_data = read_trans_data(date, cancel=False)

    # Log-transform tradePrice
    trans_data['lnprice'] = np.log(trans_data['tradePrice'])

    # Calculate tradeAmt, amtmean, and amtstd
    trans_data['tradeAmt'] = trans_data['tradePrice'] * trans_data['tradeVolume']
    trans_data['amtmean'] = trans_data.groupby(['code', 'second'])['tradeAmt'].transform('mean')
    trans_data['amtstd'] = trans_data.groupby(['code', 'second'])['tradeAmt'].transform('std')

    # Create trade size categories
    trans_data['large_trade'] = (trans_data['tradeAmt'] >= (trans_data['amtmean'] + trans_data['amtstd'])).astype(int)
    trans_data['small_trade'] = (trans_data['tradeAmt'] < trans_data['amtmean']).astype(int)
    trans_data['medium_trade'] = ((trans_data['large_trade'] == 0) & (trans_data['small_trade'] == 0)).astype(int)
    trans_data['pricediff'] = trans_data.groupby('code')['lnprice'].diff()

    # Calculate total data
    #读取daily_support
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()
    period_list = [1,10,60,300]

    # Calculate total buyAmt, totalAmt_pd
    trans_data['total_buyAmt'] = trans_data['tradeAmt'] * (trans_data['bsFlag'] == 'B').astype(int)
    trans_data['totalAmt_pd'] = trans_data['tradeAmt'] * trans_data['pricediff']

    # Group by code and second, then calculate total amounts and pivot tables
    for key in ['totalamt_sum', 'totalamt_buy_sum', 'totalamt_pd_sum']:
        keyto = {'totalamt_sum':'tradeAmt','totalamt_buy_sum':'total_buyAmt','totalamt_pd_sum':'totalAmt_pd'}[key]
        basic_df[key] = trans_data.groupby(['code', 'second'])[keyto].sum().reset_index() \
            .pivot_table(index='second', columns='code', values=keyto) \
            .reindex(index=sec_params._sec_list_all, columns=stock_list).fillna(0)

    # Define trade size categories
    trade_categories = ['large', 'medium', 'small']

    for category in trade_categories:
        # Create new columns for the given category
        trans_data[f'{category}Amt'] = trans_data[f'{category}_trade'] * trans_data['tradeAmt']
        trans_data[f'{category}_buyAmt'] = trans_data[f'{category}Amt'] * (trans_data['bsFlag'] == 'B').astype(int)
        trans_data[f'{category}Amt_pd'] = trans_data[f'{category}Amt'] * trans_data['pricediff']

        # Group by code and second, then calculate total amounts and pivot tables
        for key in [f'{category}amt_sum', f'{category}amt_buy_sum', f'{category}amt_pd_sum']:
            keyto = {f'{category}amt_sum':f'{category}Amt',f'{category}amt_buy_sum':f'{category}_buyAmt',f'{category}amt_pd_sum':f'{category}Amt_pd'}[key]
            basic_df[key] = trans_data.groupby(['code', 'second'])[keyto].sum().reset_index() \
                .pivot_table(index='second', columns='code', values=keyto) \
                .reindex(index=sec_params._sec_list_all, columns=stock_list).fillna(0)

        # Calculate factors for each period
        for period in period_list:
            totalsum_roll = rolling_sum(basic_df['totalamt_sum'].values, period) + 1e-5
            totalsum_buy_roll = rolling_sum(basic_df['totalamt_buy_sum'].values, period) + 1e-5
            targetsum_roll = rolling_sum(basic_df[f'{category}amt_sum'].values, period)
            targetsum_buy_roll = rolling_sum(basic_df[f'{category}amt_buy_sum'].values, period)
            targetsum_pd_roll = rolling_sum(basic_df[f'{category}amt_pd_sum'].values, period)

            # Calculate factors and add to the dictionary
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_roll / totalsum_roll, index=sec_params._sec_list_all, columns=stock_list), f'{category}_ratio_{period}')
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_buy_roll / totalsum_roll, index=sec_params._sec_list_all, columns=stock_list), f'{category}_buy_ratio_{period}')
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_buy_roll / (targetsum_roll + 1e-5), index=sec_params._sec_list_all, columns=stock_list), f'{category}_buy_ratio2_{period}')
            SEC_FACTOR.divid_save_factor(pd.DataFrame(targetsum_buy_roll / totalsum_buy_roll, index=sec_params._sec_list_all, columns=stock_list), f'{category}_buy_ratio3_{period}')
            tmp = pd.DataFrame((targetsum_pd_roll / (targetsum_roll + 1e-5)) * 1e3, index=sec_params._sec_list_all, columns=stock_list)
            SEC_FACTOR.divid_save_factor(tmp, f'{category}_pd_ratio_{period}')
            SEC_FACTOR.divid_save_factor(tmp - \
                            pd.DataFrame((rolling_sum(basic_df['totalamt_pd_sum'].values, period) /
                            totalsum_roll) * 1e3, index=sec_params._sec_list_all, columns=stock_list), f'{category}_pd_diff_{period}')

        # Remove unnecessary columns and variables
        del trans_data[f'{category}Amt'], trans_data[f'{category}_buyAmt'], trans_data[f'{category}Amt_pd']
        del basic_df[f'{category}amt_sum'], basic_df[f'{category}amt_buy_sum'], basic_df[f'{category}amt_pd_sum']

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'yy_trans_ls1'
    factor_path = r'/home/datamake118/datamake118_base/nas2/sec_factor2'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240726',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,get_trans_factor1),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)