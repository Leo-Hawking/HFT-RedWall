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

class params:
    factor_out_path = '/home/datamake94/data_nb8'
    
    factor_path_dict = {
        'OLD_FACTOR':rf'{factor_out_path}/min_factor_agg'
        # 'gp1':rf'{factor_out_path}/gp1_select',
    }
    static_factor_path= r'/home/datamake94/data_nb8/min_factor_agg/static_factor_path2'

    future_ret_path_dict= {'1d':r'/home/datamake94/data_nb8/min_factor_agg/min_ret/1d_exret',
                          '3d':r'/home/datamake94/data_nb8/min_factor_agg/min_ret/3d_exret',
                        '30':r'/home/datamake94/data_nb8/min_factor_agg/min_ret/30_exret',}
    
    ic_ls_future_ret_list=['1d','3d','30']
    
    _model_path = r'/home/datamake94/data_nb0/模型库/min_model_update20241015'
    
    _min_max_path = r'/home/datamake94/data_nb8/min_factor_agg/min_max_dict_norm' 
    
    min_valid_path= r'/home/datamake94/data_nb8/min_valid_stock'  #训练时使用

    tmpfs_path = r'/home/datamake94/data_nb9/min_factor_tmp'

    select_factor_path = r'/home/datamake94/可转债高频策略/factor_engineering/select_factor_20240818'
        
    sec_list_dict={}
    sec_list_period_dict={}
    
    sec_list = calc_time_series().copy()
    sec_list = [x//100 for x in sec_list if x%100 == 0]
    sec_list_all=sorted([x for x in sec_list if x >= 930 and x < 1455])
    sec_list_all=[x for x in sec_list_all]

    with open(r'/home/datamake94/data_nb8/min_factor_agg/all_time.pkl','rb') as f:
        valid_time = pkl.load(f)
    sec_list_factor_old = valid_time[:-1]
    sec_list_factor=sorted([x for x in sec_list if x >= 930 and x < 1455])  #所有因子时间维的索引，均一致
    sec_list_factor=[x for x in sec_list_factor]

    sec_list_ret = sorted([x for x in sec_list if x >= 930 and x < 1455])  #所有收益率时间维的索引，均一致
    sec_list_ret=[x for x in sec_list_ret]

    sec_list_validts = sorted([x for x in sec_list if x >= 930 and x < 1455]) #所有收益率时间维的索引，均一致
    
    for future_period in future_ret_path_dict.keys():
        if future_period!='15m':
            sec_list_=sorted([x for x in sec_list if x >= 93200 and x <= 144959])
        else:
            sec_list_=sorted([x for x in sec_list if x >= 93200 and x <= 143959])
        sec_list_dict[future_period]=sec_list_
        sec_list_period_dict[future_period]={93000:[x for x in sec_list_ if x<100000],
                                             100000:[x for x in sec_list_ if x>=100000],
                                            'allday':sec_list_}    
    sec_list_period_dict['interval1'] = sorted([x for x in sec_list if x >= 930 and x < 1000])
    sec_list_period_dict['interval2'] = sorted([x for x in sec_list if x >= 1000 and x < 1130])
    sec_list_period_dict['interval3'] = sorted([x for x in sec_list if x >= 1300 and x < 1455])
   
    # random_num=int(len(sec_list_period_dict['5m'][93000])*0.3) 
    tmpfs_fast = True