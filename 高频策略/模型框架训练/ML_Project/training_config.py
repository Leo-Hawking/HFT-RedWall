import pandas as pd
import pickle as pkl
import numpy as np
import datetime

def calc_time_series(contain_all=False):
    time_series=[]  #所有的秒序列,不包括集合竞价
    early_thre=30 if not contain_all else 15
    late_thre=57 if not contain_all else 60
    for i in [9,10,11,13,14]:
        for j in range(60):
            if i==9 and j<early_thre:
                continue
            elif i==11 and j>=30:
                continue
            elif i==14 and j>=late_thre:
                continue
            time_series.extend([int(str(i)+'0'*(2-len(str(j)))+str(j)+'0'*(2-len(str(x)))+str(x)) for x in range(60)])
    return time_series

class Dataset:
    min_se = None
    max_se = None
    nan_ratio=None
    
    with open(r"/home/datamake94/data_nb0/support_data/citic_2.pkl", 'rb') as f:
        industry_table = pkl.load(f)
    industry_table.index = pd.to_datetime(
        industry_table.index).strftime('%Y%m%d')
    industry_table.columns = [int(x) for x in industry_table.columns]

class params:
    factor_path_dict = {
        'ysw_orderbook1':r'/home/intern1/hft_database/nas0/ysw_orderbook_factorall1',
        'ysw_pv_A':r'/home/intern1/hft_database/nas0/ysw_hft_factorall_A',
        'ysw_pv_B':r'/home/intern1/hft_database/nas0/ysw_hft_factorall_B',
        'ysw_graph':r'/home/intern1/hft_database/nas7/sec_factor/ysw_factor_graph2',
        'ysw_orderbook2':r'/home/intern1/hft_database/nas7/sec_factor/ysw_orderbook_factorall2',
        'ysw_pv2':r'/home/intern1/hft_database/nas6/sec_factor3/hft_sec_factor2',
        'yy_order_basic':r'/home/intern1/hft_database/nas2/sec_factor2/yy_order_basic',
        'yy_order_ls1':r'/home/intern1/hft_database/nas2/sec_factor2/yy_order_ls1',
        'yy_trans_basic':r'/home/intern1/hft_database/nas2/sec_factor2/yy_trans_basic',
        'yy_trans_ls1':r'/home/intern1/hft_database/nas2/sec_factor2/yy_trans_ls1',
        'yy_orderbook3':r'/home/intern1/hft_database/nas8/sec_factor4/yy_orderbook_factorall3',
        #'yy_pv3':r'/home/intern1/hft_database/nas8/sec_factor4/yy_hft_factor_sec3', 
        'yy_pv4':r'/home/intern1/hft_database/nas8/sec_factor4/yy_hft_factor_sec4',
#         'yy_orderagg1':r'/home/intern1/hft_database/nas2/sec_factor2/yy_orderagg1',
#         'yy_cancel':r'/home/intern1/hft_database/nas8/sec_factor4/yy_cancel1',
    }

    ic_ls_path = r'/home/intern1/data0/sec_factor_corrdf/ic_ls_corr_final_0509'
    period_list=[93000,100000]

    future_ret_path_dict= {'1m':r'/home/intern1/data6/sec_ret/future_ob_60return',
                          '3m':r'/home/intern1/data6/sec_ret/future_ob_180return',
                        '5m':r'/home/intern1/data6/sec_ret/future_ob_300return',
                        '15m':r'/home/intern1/data6/sec_ret/future_ob_900return',
                        '15s':r'/home/intern1/data6/sec_ret/future_ob_15return',
                        '1d':r'/home/intern1/data6/sec_ret/future_ob_1d'}
    
    ic_ls_future_ret_list=['15s','1m','5m']
    
    _model_path = r'/home/datamake94/data_nb7/sec_model_all_final'
    
    _min_max_path = r'/home/intern1/hft_factor_comb/max_min_/max_min_final_0323' 
    #_min_max_model_path = r'/home/intern1/hft_factor_comb/max_min_/max_min_model_final3'
    
    static_factor_path= r'/home/datamake94/data_nb8/min_factor_agg/static_factor_path2'
    #ml_factor_path= r'/home/intern1/data4/sec_score_output/deeplob_1m_oos'
    sec_valid_path= r'/home/datamake94/data_nb6/sec_valid_stock'  #训练时使用

    tmpfs_path = r'/home/datamake94/data_nb9/sec_factor_tmp'
        
    sec_list_dict={}
    sec_list_period_dict={}
    
    sec_list = calc_time_series().copy()
    sec_list_all=sorted([x for x in sec_list if x >= 93200 and x <= 144955])
    sec_list_all=[x for x in sec_list_all if x % 5==0]

    sec_list_factor=sorted([x for x in sec_list if x >= 93000 and x <= 145455])  #所有因子时间维的索引，均一致
    sec_list_factor=[x for x in sec_list_factor if x % 5==0]
    
    for future_period in future_ret_path_dict.keys():
        if future_period!='15m':
            sec_list_=sorted([x for x in sec_list if x >= 93200 and x <= 144955 and x % 5==0])
        else:
            sec_list_=sorted([x for x in sec_list if x >= 93200 and x <= 143955 and x % 5==0])
        sec_list_dict[future_period]=sec_list_
        sec_list_period_dict[future_period]={93000:[x for x in sec_list_ if x<100000],
                                             100000:[x for x in sec_list_ if x>=100000]}    
   
    random_num=int(len(sec_list_period_dict['5m'][93000])*0.3) 