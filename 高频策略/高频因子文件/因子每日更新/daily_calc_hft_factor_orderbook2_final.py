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

def load_factor6(date, fea_, SEC_FACTOR):        
    avg_price=(fea_['bp_1']+fea_['sp_1'])/2
    vwap=fea_['amt']/fea_['vol']
    vwap1=vwap.fillna(method='ffill')
    vwap2=vwap.combine_first(fea_['close'])
    
    avg_price_rolling=(avg_price+avg_price.shift(1))/2
    
    ############################
    for lookback_m in [5,15,30,60]:
        second_data=(fea_['close']/avg_price-1).rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'MPB_%s_mean'%lookback_m)
        
        second_data=(vwap1/avg_price_rolling-1).rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'MPB2_%s_mean'%lookback_m)
        
        second_data=(vwap2/avg_price_rolling-1).rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'MPB3_%s_mean'%lookback_m)
        
        second_data=(fea_['close']/avg_price-1).rolling(lookback_m,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'MPB_%s_skew'%lookback_m)
        
    return

def load_factor7(date, fea_, SEC_FACTOR):    
    bvol=0
    svol=0
    w_bvol=0
    w_svol=0
    
    for s in range(1,6):
        bvol+=fea_['bv_%s'%s]
        svol+=fea_['sv_%s'%s]
        w_bvol+=fea_['bv_%s'%s]*(1.2-0.2*s)
        w_svol+=fea_['sv_%s'%s]*(1.2-0.2*s)
    
    avg_bvol=bvol.rolling(60,min_periods=1).mean(engine='numba')
    avg_svol=svol.rolling(60,min_periods=1).mean(engine='numba')
    
    vol_ratio=(bvol-svol)/(bvol+svol)
    w_vol_ratio=(w_bvol-w_svol)/(w_bvol+w_svol)
    
    delta_vol_ratio=vol_ratio-w_vol_ratio
    
    ############################
    for lookback_m in [5,15,30,60]:
        second_data= bvol.rolling(lookback_m,min_periods=1).mean(engine='numba')/avg_bvol-svol.rolling(lookback_m,min_periods=1).mean(engine='numba')/avg_svol
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'rvol_%s_mean'%lookback_m)
        
        second_data= w_bvol.rolling(lookback_m,min_periods=1).mean(engine='numba')/avg_bvol-w_svol.rolling(lookback_m,min_periods=1).mean(engine='numba')/avg_svol
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'w_rvol_%s_mean'%lookback_m)
        
        second_data= vol_ratio.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'rvol_ratio_%s_mean'%lookback_m)
        
        second_data= w_vol_ratio.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'w_rvol_ratio_%s_mean'%lookback_m)
        
        second_data= delta_vol_ratio.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'delta_rvol_ratio_%s_mean'%lookback_m)
        
    return

def load_factor9(date, fea_, SEC_FACTOR):
    bvol_diff=0
    bvol=0
    svol_diff=0
    svol=0
    bp1_diff=fea_['bp_1'].diff()
    sp1_diff=fea_['sp_1'].diff()
    
    for s in range(2,6):
        svol_diff+=fea_['sv_%s'%s].diff()*(8-s)  
        svol+=fea_['sv_%s'%s]*(8-s)  
        bvol_diff+=fea_['bv_%s'%s].diff()*(8-s)
        bvol+=fea_['bv_%s'%s]*(8-s)  
    
    bs_vol_scale1=(np.abs(svol_diff)+np.abs(bvol_diff)).rolling(60,min_periods=1).mean(engine='numba')
    bs_vol_scale2=(bvol+svol).rolling(60,min_periods=1).mean(engine='numba')
    
    bp1_flat = (np.abs(bp1_diff) < 1e-6).astype(int)
    sp1_flat = (np.abs(sp1_diff) < 1e-6).astype(int)
    
    b_flat_vol_diff= (bp1_flat*bvol_diff).replace(0,np.nan)  #只看价格不变区间内的情况
    s_flat_vol_diff= (sp1_flat*svol_diff).replace(0,np.nan)
    
    ############################
    for lookback_m in [30,60]: #不要用5s和15s，0太多
        
        second_data= (b_flat_vol_diff.rolling(lookback_m,min_periods=1).mean(engine='numba')-s_flat_vol_diff.rolling(lookback_m,min_periods=1).mean(engine='numba'))/bs_vol_scale1
        SEC_FACTOR.divid_save_factor(second_data.fillna(0).loc[sec_params.sec_list],'ec1_%s_mean'%lookback_m)
        
        second_data= (b_flat_vol_diff.rolling(lookback_m,min_periods=1).mean(engine='numba')-s_flat_vol_diff.rolling(lookback_m,min_periods=1).mean(engine='numba'))/bs_vol_scale2
        SEC_FACTOR.divid_save_factor(second_data.fillna(0).loc[sec_params.sec_list],'ec2_%s_mean'%lookback_m)
        
    return

def load_factor10(date, fea_, SEC_FACTOR):
    bamt=0
    bvol=0
    samt=0
    svol=0
    
    for s in range(1,6):
        bamt+=fea_['bp_%s'%s]*fea_['bv_%s'%s]
        samt+=fea_['sp_%s'%s]*fea_['sv_%s'%s]
        bvol+=fea_['bv_%s'%s]
        svol+=fea_['sv_%s'%s]
    
    avg_price=(fea_['bp_1']+fea_['sp_1'])/2
    vwap_b=bamt/bvol
    vwap_s=samt/svol
    rvwap_b=(avg_price-vwap_b)/avg_price
    rvwap_s=(vwap_s-avg_price)/avg_price
    
    core_diff=(rvwap_b-rvwap_s)/(rvwap_b+rvwap_s)
    core_diff2=rvwap_b-rvwap_s
    
    ############################
    for lookback_m in [5,15,30,60]:
        second_data= core_diff.diff(lookback_m)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'vwap_diff_%s'%lookback_m)
        
        second_data= core_diff2.diff(lookback_m)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'vwap2_diff_%s'%lookback_m)
        
        second_data= vwap_b.pct_change(lookback_m)+ vwap_s.pct_change(lookback_m)  #没有去量纲，携带了中间价的动量
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'vwap_chg_%s'%lookback_m)
        
        second_data= avg_price.pct_change(lookback_m)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'mp_%s'%lookback_m)
        
    return

def load_factor11(date, fea_, SEC_FACTOR):
    vwap=fea_['amt']/fea_['vol']
    vwap=vwap.combine_first(fea_['close'])
        
    bp_spread=vwap-fea_['bp_1']
    sp_spread=vwap-fea_['sp_1']
    
    ############################
    for lookback_m in [5,15,30,60]:
        second_data=fea_['bv_1'].rolling(lookback_m,min_periods=1).corr(bp_spread)-fea_['sv_1'].rolling(lookback_m,min_periods=1).corr(sp_spread)
        SEC_FACTOR.divid_save_factor(second_data.replace([np.inf,-np.inf],np.nan).loc[sec_params.sec_list],'bspvcorr1_%s'%lookback_m)
        
        second_data=fea_['bv_1'].rolling(lookback_m,min_periods=1).corr(bp_spread)+fea_['sv_1'].rolling(lookback_m,min_periods=1).corr(sp_spread)
        SEC_FACTOR.divid_save_factor(second_data.replace([np.inf,-np.inf],np.nan).loc[sec_params.sec_list],'bspvcorr2_%s'%lookback_m)
        
    return

def load_factor12(date, fea_, SEC_FACTOR):
    vwap=fea_['amt']/fea_['vol']
    vwap=vwap.combine_first(fea_['close'])
    
    avg_price=(fea_['bp_1']+fea_['sp_1'])/2
    bs_vol_scale=(fea_['bv_1']+fea_['sv_1']).rolling(60,min_periods=1).mean(engine='numba')
    
    bp_spread=(vwap-fea_['bp_1'])/avg_price
    sp_spread=(fea_['sp_1']-vwap)/avg_price
    
    bspv1=bp_spread*fea_['bv_1']-sp_spread*fea_['sv_1']
    bspv2=bp_spread*fea_['sv_1']-sp_spread*fea_['bv_1']
    
    ############################
    for lookback_m in [5,15,30,60]:
        second_data=bspv1.rolling(lookback_m,min_periods=1).mean(engine='numba')/bs_vol_scale
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'bspv1_%s'%lookback_m)
        
        second_data=bspv2.rolling(lookback_m,min_periods=1).mean(engine='numba')/bs_vol_scale
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list],'bspv2_%s'%lookback_m)

    return

def calc_orderbook_final2(date):
    #初始化
    factor_list = ['MPB_5_mean', 'MPB2_5_mean', 'MPB3_5_mean', 'MPB_5_skew', 'MPB_15_mean', 'MPB2_15_mean', 'MPB3_15_mean', 'MPB_15_skew', 'MPB_30_mean', 'MPB2_30_mean', \
    'MPB3_30_mean', 'MPB_30_skew', 'MPB_60_mean', 'MPB2_60_mean', 'MPB3_60_mean', 'MPB_60_skew', 'rvol_5_mean', 'w_rvol_5_mean', 'rvol_ratio_5_mean', 'w_rvol_ratio_5_mean', \
    'delta_rvol_ratio_5_mean', 'rvol_15_mean', 'w_rvol_15_mean', 'rvol_ratio_15_mean', 'w_rvol_ratio_15_mean', 'delta_rvol_ratio_15_mean', 'rvol_30_mean', 'w_rvol_30_mean', \
    'rvol_ratio_30_mean', 'w_rvol_ratio_30_mean', 'delta_rvol_ratio_30_mean', 'rvol_60_mean', 'w_rvol_60_mean', 'rvol_ratio_60_mean', 'w_rvol_ratio_60_mean', 'delta_rvol_ratio_60_mean', \
    'ec1_30_mean', 'ec2_30_mean', 'ec1_60_mean', 'ec2_60_mean', 'vwap_diff_5', 'vwap2_diff_5', 'vwap_chg_5', 'mp_5', 'vwap_diff_15', 'vwap2_diff_15', 'vwap_chg_15', 'mp_15', 'vwap_diff_30', \
    'vwap2_diff_30', 'vwap_chg_30', 'mp_30', 'vwap_diff_60', 'vwap2_diff_60', 'vwap_chg_60', 'mp_60', 'bspvcorr1_5', 'bspvcorr2_5', 'bspvcorr1_15', 'bspvcorr2_15', 'bspvcorr1_30', 'bspvcorr2_30', \
    'bspvcorr1_60', 'bspvcorr2_60', 'bspv1_5', 'bspv2_5', 'bspv1_15', 'bspv2_15', 'bspv1_30', 'bspv2_30', 'bspv1_60', 'bspv2_60']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取数据
    fea_ = read_orderbook_data(date)
    basic_fea = read_sec_table(date)
    fea_.update(basic_fea)

    #计算因子
    load_factor6(date, fea_, SEC_FACTOR)
    load_factor7(date, fea_, SEC_FACTOR)
    load_factor9(date, fea_, SEC_FACTOR)
    load_factor10(date, fea_, SEC_FACTOR)
    load_factor11(date, fea_, SEC_FACTOR)
    load_factor12(date, fea_, SEC_FACTOR)
    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'ysw_orderbook_factorall2'
    factor_path = r'/home/datamake118/datamake118_base/nas7/sec_factor'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,calc_orderbook_final2),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)