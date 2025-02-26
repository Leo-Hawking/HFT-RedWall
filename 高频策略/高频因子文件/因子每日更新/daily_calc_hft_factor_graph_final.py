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

class graph:
    with open(r'/home/datamake118/basic_data/tonglian_data2/support_data/citic_2.pkl', 'rb') as f:
        industry_table = pkl.load(f)
    industry_table.index = pd.to_datetime(
        industry_table.index).strftime('%Y%m%d')
    industry_table.columns = [int(x) for x in industry_table.columns]
    
    with open(r'/home/datamake118/basic_data/tonglian_data2/ohlc_fea/NEG_MARKET_VALUE.pkl', 'rb') as f:
        neg_mktvalue = pkl.load(f)
    neg_mktvalue.index = pd.to_datetime(
        neg_mktvalue.index).strftime('%Y%m%d')
    neg_mktvalue.columns = [int(x) for x in neg_mktvalue.columns]
    
    neg_mkt_group=pd.DataFrame(8,index=neg_mktvalue.columns,columns=neg_mktvalue.index)
    neg_mkt_group[pd.isna(neg_mktvalue.T)]=np.nan

    for i in range(6,-1,-1):
        neg_mkt_group[neg_mktvalue.T>=neg_mktvalue.quantile(1-2**i/100,axis=1)]=i+1

def df_to_tensor(df):
    return torch.from_numpy(df.to_numpy()).float()

def get_correlation_matrix(pre_date):
    close=pd.read_feather(r'/home/datamake118/datamake118_base/nas3/sec_fea_table/sec_%s_table/%s.fea'%('close',pre_date)).set_index('second')
    stock_list=close.columns.astype(int)
    ret_ts=df_to_tensor(close.fillna(method='ffill').loc[sec_params.sec_list].pct_change(12).iloc[::12,:].dropna(how='all').fillna(0))  #5s bar的close
    result=ret_ts.view((ret_ts.shape[0],ret_ts.shape[1],1))*ret_ts.view((ret_ts.shape[0],1,ret_ts.shape[1]))
    conv=result.mean(dim=0)-ret_ts.mean(dim=0).view(-1,1)*ret_ts.mean(dim=0).view(1,-1)
    rho=conv/torch.sqrt(conv.diag()).view(-1,1)/torch.sqrt(conv.diag()).view(1,-1)
    matrix=pd.DataFrame(rho.numpy(),index=stock_list,columns=stock_list)
    return matrix

def get_target_matrix(date):
    pre_date=get_pre_date(date)
    matrix={}
    matrix['cotrading']=pd.read_feather(r'/home/datamake118/datamake118_base/nas3/cotrade_matrix/%s.fea'%pre_date).set_index('code')
    matrix['cotrading'].columns=matrix['cotrading'].columns.astype(int)
    industry_slice = graph.industry_table.loc[pre_date].dropna()
    stock_ind=pd.get_dummies(industry_slice)
    matrix['industry']=stock_ind@stock_ind.T
    mkt_slice = graph.neg_mkt_group[pre_date].dropna()
    stock_mkt=pd.get_dummies(mkt_slice)
    matrix['mkt_value']=stock_mkt@stock_mkt.T
    matrix['rho']=get_correlation_matrix(pre_date)
    
    for mtx in matrix.keys(): #抹去对角线元素
        matrix[mtx] = matrix[mtx].astype(float)
        for i in range(matrix[mtx].shape[1]):
            matrix[mtx].iloc[i,i] = 0
    return matrix

#####################
#图谱2：直接考察共同成交矩阵和相关性矩阵，将关系较弱的联系点去除，只保留最强的样本点
#对于正向成交和负向成交均考虑（相关性矩阵）
#####################
def calc_daily_graph_func(date):
    #初始化
    basic_df = {}
    factor_list = ['p_vwap_ori_30', 'graph_industry_p_vwap_thresholding_30', 'graph_industry_cubic_p_vwap_thresholding_30', 'graph_industry_wsquare_p_vwap_thresholding_30', 
    'graph_industry_wabs_p_vwap_thresholding_30', 'graph_rho_pos_p_vwap_thresholding_30', 'graph_rho_pos_cubic_p_vwap_thresholding_30', 
    'graph_rho_pos_wsquare_p_vwap_thresholding_30', 'graph_rho_pos_wabs_p_vwap_thresholding_30', 'graph_rho_neg_p_vwap_thresholding_30', 
    'graph_rho_neg_cubic_p_vwap_thresholding_30', 'graph_rho_neg_wsquare_p_vwap_thresholding_30', 'graph_rho_neg_wabs_p_vwap_thresholding_30', 
    'graph_cotrading_pos_p_vwap_thresholding_30', 'graph_cotrading_pos_cubic_p_vwap_thresholding_30', 'graph_cotrading_pos_wsquare_p_vwap_thresholding_30', 
    'graph_cotrading_pos_wabs_p_vwap_thresholding_30', 'p_vwap_ori_60', 'graph_industry_p_vwap_thresholding_60', 'graph_industry_cubic_p_vwap_thresholding_60', 
    'graph_industry_wsquare_p_vwap_thresholding_60', 'graph_industry_wabs_p_vwap_thresholding_60', 'graph_rho_pos_p_vwap_thresholding_60', 
    'graph_rho_pos_cubic_p_vwap_thresholding_60', 'graph_rho_pos_wsquare_p_vwap_thresholding_60', 'graph_rho_pos_wabs_p_vwap_thresholding_60', 
    'graph_rho_neg_p_vwap_thresholding_60', 'graph_rho_neg_cubic_p_vwap_thresholding_60', 'graph_rho_neg_wsquare_p_vwap_thresholding_60', 
    'graph_rho_neg_wabs_p_vwap_thresholding_60', 'graph_cotrading_pos_p_vwap_thresholding_60', 'graph_cotrading_pos_cubic_p_vwap_thresholding_60', 
    'graph_cotrading_pos_wsquare_p_vwap_thresholding_60', 'graph_cotrading_pos_wabs_p_vwap_thresholding_60', 'p_vwap_ori_300', 'graph_industry_p_vwap_thresholding_300', 
    'graph_industry_cubic_p_vwap_thresholding_300', 'graph_industry_wsquare_p_vwap_thresholding_300', 'graph_industry_wabs_p_vwap_thresholding_300', 
    'graph_rho_pos_p_vwap_thresholding_300', 'graph_rho_pos_cubic_p_vwap_thresholding_300', 'graph_rho_pos_wsquare_p_vwap_thresholding_300', 
    'graph_rho_pos_wabs_p_vwap_thresholding_300', 'graph_rho_neg_p_vwap_thresholding_300', 'graph_rho_neg_cubic_p_vwap_thresholding_300', 
    'graph_rho_neg_wsquare_p_vwap_thresholding_300', 'graph_rho_neg_wabs_p_vwap_thresholding_300', 'graph_cotrading_pos_p_vwap_thresholding_300', 
    'graph_cotrading_pos_cubic_p_vwap_thresholding_300', 'graph_cotrading_pos_wsquare_p_vwap_thresholding_300', 'graph_cotrading_pos_wabs_p_vwap_thresholding_300']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #横截面因子，需要提前获取股票池
    daily_support = read_daily_support(date)
    stock_list = daily_support.index.tolist()

    #读取数据
    all_fea_dict = read_sec_table(date)
    pre_date = get_pre_date(date)
    for fea_name in ['close', 'vol', 'amt', 'num', 'high', 'low']:
        all_fea_dict[fea_name] = all_fea_dict[fea_name].reindex(columns=stock_list)
    #读取图谱数据
    ###################################
    #这块可以我在前一天晚上算好然后直接传到共享盘，get_target_matrix函数写成读取的逻辑就行
    ###################################
    matrix=get_target_matrix(date)
    for mtx in matrix.keys():
        matrix[mtx] = matrix[mtx].reindex(index=stock_list,columns=stock_list).fillna(0)

    ##############################################
    #yy:注意，这段代码实盘的时候不需要添加，实盘的matrix是我提前算到这个位置的。
    ##############################################
    #'''
    matrix_new={}
    
    rho_1=matrix['rho'].quantile(0.01)
    rho_99=matrix['rho'].quantile(0.99)
    cotrading_99=matrix['cotrading'].quantile(0.99)
    
    matrix_new['industry']=matrix['industry'].copy()
    matrix_new['rho_pos']=matrix['rho'].copy()
    matrix_new['rho_neg']=matrix['rho'].copy()
    matrix_new['cotrading_pos']=matrix['cotrading'].copy()  #以上参与加权的均只有1/100（二级行业是100个左右，不均匀）
    
    matrix_new['rho_neg'][matrix_new['rho_pos']>rho_1]=0
    matrix_new['rho_pos'][matrix_new['rho_pos']<rho_99]=0
    matrix_new['cotrading_pos'][matrix_new['cotrading_pos']<cotrading_99]=0
    
    matrix=matrix_new
        
    for mtx in ['industry','rho_pos','rho_neg','cotrading_pos']:
        matrix[mtx] = matrix[mtx]/(np.abs(matrix[mtx]).sum() + 1e-8)  #不乘以-1，并且对角线不设为1，后面也要对应调整
    #'''
    ##############################################
    #结束
    ##############################################
    sec_5s = [x for x in sec_params.sec_list_all if x%5==0]
    price=all_fea_dict['close'].fillna(method='ffill')
    price_5s=all_fea_dict['close'].loc[sec_5s].fillna(method='ffill')
    vol_5s=all_fea_dict['vol'].rolling(5).sum(engine='numba').loc[sec_5s]
    amt_5s=all_fea_dict['amt'].rolling(5).sum(engine='numba').loc[sec_5s]
    
    log_price=np.log(price_5s)
    rit=log_price.diff()

    for lookback_m in [30,60,300]:
        window_period = int(lookback_m//5)
        window_period_name = str(lookback_m)
        vwap_ratio= (price_5s/(amt_5s.rolling(window_period,min_periods=1).sum(engine='numba')/vol_5s.rolling(window_period,min_periods=1).sum(engine='numba'))-1).fillna(0)*100
        vwap_ratio = vwap_ratio.loc[sec_params.sec_list] 
        SEC_FACTOR.divid_save_factor(vwap_ratio, 'p_vwap_ori_'+window_period_name)

        for mtx in ['industry','rho_pos','rho_neg','cotrading_pos']:
            second_data=vwap_ratio-vwap_ratio @ matrix[mtx]
            SEC_FACTOR.divid_save_factor(second_data, 'graph_%s_p_vwap_thresholding_%s'%(mtx,window_period_name))
            
            second_data=vwap_ratio**3-vwap_ratio**3 @ matrix[mtx]  #三次方以加强因子本身的量纲，隐性地用本身的二次方进行加权
            SEC_FACTOR.divid_save_factor(second_data, 'graph_%s_cubic_p_vwap_thresholding_%s'%(mtx,window_period_name))
            
            second_data=vwap_ratio-(vwap_ratio**3 @ matrix[mtx])/(vwap_ratio**2 @ matrix[mtx]) #显性地使用本身的二次方加权
            SEC_FACTOR.divid_save_factor(second_data, 'graph_%s_wsquare_p_vwap_thresholding_%s'%(mtx,window_period_name))
            
            second_data=vwap_ratio-(vwap_ratio*np.abs(vwap_ratio) @ matrix[mtx])/(np.abs(vwap_ratio) @ matrix[mtx]) #显性地使用值的绝对值加权
            SEC_FACTOR.divid_save_factor(second_data, 'graph_%s_wabs_p_vwap_thresholding_%s'%(mtx,window_period_name))

    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'ysw_factor_graph2'
    factor_path = r'/home/datamake118/datamake118_base/nas7/sec_factor'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,calc_daily_graph_func),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)