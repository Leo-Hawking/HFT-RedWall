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

def load_factor1(date, fea_, SEC_FACTOR):
    bp1_diff=fea_['bp_1'].diff()
    sp1_diff=fea_['sp_1'].diff()
    bv1_diff=fea_['bv_1'].diff()
    bv1_shift=fea_['bv_1'].shift(1)
    sv1_diff=fea_['sv_1'].diff()
    sv1_shift=fea_['sv_1'].shift(1)

    bp1_up = (bp1_diff>0).astype(int)
    bp1_down = (bp1_diff<0).astype(int)
    bp1_flat = (np.abs(bp1_diff)<=1e-8).astype(int)

    sp1_up = (sp1_diff>0).astype(int)
    sp1_down = (sp1_diff<0).astype(int)
    sp1_flat = (np.abs(sp1_diff)<=1e-8).astype(int)

    bs_vol_scale = (fea_['bv_1']+fea_['sv_1']).rolling(60, min_periods=1).mean(engine='numba')

    OFI1_up_g1=-bp1_down*bv1_shift+bp1_flat*bv1_diff
    OFI1_down_g1=-sp1_up*sv1_shift+sp1_flat*sv1_diff
    OFI2_up_g1=OFI1_up_g1+bp1_up*fea_['bv_1']
    OFI2_down_g1=OFI1_down_g1+sp1_down*fea_['sv_1']

    ############################
    for lookback_m in [5,15,30,60]:
        second_data= (OFI1_up_g1.rolling(lookback_m,min_periods=1).mean(engine='numba')-OFI1_down_g1.rolling(lookback_m,min_periods=1).mean(engine='numba'))/bs_vol_scale  
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OFI_gear1_%s_mean'%lookback_m)

        second_data= (OFI2_up_g1.rolling(lookback_m,min_periods=1).mean(engine='numba')-OFI2_down_g1.rolling(lookback_m,min_periods=1).mean(engine='numba'))/bs_vol_scale
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OFI2_gear1_%s_mean'%lookback_m)
        
        second_data= (OFI1_up_g1.rolling(lookback_m,min_periods=1).std(engine='numba')-OFI1_down_g1.rolling(lookback_m,min_periods=1).std(engine='numba'))/bs_vol_scale   
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OFI_gear1_%s_std'%lookback_m)

        second_data= (OFI2_up_g1.rolling(lookback_m,min_periods=1).std(engine='numba')-OFI2_down_g1.rolling(lookback_m,min_periods=1).std(engine='numba'))/bs_vol_scale
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OFI2_gear1_%s_std'%lookback_m)
        
        second_data= OFI1_up_g1.rolling(lookback_m,min_periods=1).skew()-OFI1_down_g1.rolling(lookback_m,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OFI_gear1_%s_skew'%lookback_m)

        second_data= OFI2_up_g1.rolling(lookback_m,min_periods=1).skew()-OFI2_down_g1.rolling(lookback_m,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OFI2_gear1_%s_skew'%lookback_m)

    return

def load_factor2(date, fea_, SEC_FACTOR):
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
    MCI_b= rvwap_b/bvol
    rvwap_s=(vwap_s-avg_price)/avg_price
    MCI_s=rvwap_s/svol
    
    MCI_IMB= (MCI_b-MCI_s)/(MCI_b+MCI_s)

    ############################
    for lookback_m in [5,15,30,60]:
        second_data= MCI_IMB.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'MCI_IMB_%s'%lookback_m)
        
        second_data= rvwap_b.rolling(lookback_m,min_periods=1).mean(engine='numba')-rvwap_s.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'VWAP_IMB_%s_mean'%lookback_m)
        
        second_data= rvwap_b.rolling(lookback_m,min_periods=1).std(engine='numba')-rvwap_s.rolling(lookback_m,min_periods=1).std(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'VWAP_IMB_%s_std'%lookback_m)
        
        second_data= rvwap_b.rolling(lookback_m,min_periods=1).skew()-rvwap_s.rolling(lookback_m,min_periods=1).skew()
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'VWAP_IMB_%s_skew'%lookback_m)

    return

def load_factor3(date, fea_, SEC_FACTOR):
    bvol=0
    svol=0
    for s in range(1,6):
        bvol+=fea_['bv_%s'%s]
        svol+=fea_['sv_%s'%s]
    
    avg_vol=bvol.rolling(60,min_periods=1).mean(engine='numba')+svol.rolling(60,min_periods=1).mean(engine='numba')
    
    vol_bs_ratio= bvol/svol

    ############################
    for lookback_m in [5,15,30,60]:
        second_data= (bvol.rolling(lookback_m,min_periods=1).median(engine='numba')-svol.rolling(lookback_m,min_periods=1).median(engine='numba'))/avg_vol
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OBS_50quan_%s'%lookback_m)
        
        second_data= (bvol.rolling(lookback_m,min_periods=1).quantile(0.8)-svol.rolling(lookback_m,min_periods=1).quantile(0.8))/avg_vol
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OBS_80quan_%s'%lookback_m)
        
        second_data= vol_bs_ratio.rolling(lookback_m,min_periods=1).quantile(0.8)-(1/vol_bs_ratio).rolling(lookback_m,min_periods=1).quantile(0.8)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'OBS2_80quan_%s'%lookback_m)

    return

def load_factor4(date, fea_, SEC_FACTOR):
    bvol=0
    svol=0
    bvol_diff=0
    svol_diff=0
    bvol_diff_abs=0
    svol_diff_abs=0
    
    for s in range(2,6):
        bvol+=fea_['bv_%s'%s]
        bvol_diff+=fea_['bv_%s'%s]-fea_['bv_%s'%(s-1)]
        bvol_diff_abs+=np.abs(fea_['bv_%s'%s]-fea_['bv_%s'%(s-1)])
        svol+=fea_['sv_%s'%s]
        svol_diff=fea_['sv_%s'%s]-fea_['sv_%s'%(s-1)]
        svol_diff_abs+=np.abs(fea_['sv_%s'%s]-fea_['sv_%s'%(s-1)])
        
    slope_b=(fea_['bp_1']-fea_['bp_5'])/bvol
    slope_s=(fea_['sp_5']-fea_['sp_1'])/svol
    
    bvol_1_ratio=fea_['bv_1']/bvol
    svol_1_ratio=fea_['sv_1']/svol
    
    bvol_chg=bvol_diff/bvol_diff_abs
    svol_chg=svol_diff/svol_diff_abs
    
    avg_slope=slope_b.rolling(60,min_periods=1).mean(engine='numba')+slope_s.rolling(60,min_periods=1).mean(engine='numba')

    ############################
    for lookback_m in [5,15,30,60]:
        second_data= (slope_b.rolling(lookback_m,min_periods=1).mean(engine='numba')-slope_s.rolling(lookback_m,min_periods=1).mean(engine='numba'))/avg_slope
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'slope_%s_mean'%lookback_m)
        
        second_data= (slope_b.rolling(lookback_m,min_periods=1).median(engine='numba')-slope_s.rolling(lookback_m,min_periods=1).median(engine='numba'))/avg_slope
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'slope_%s_median'%lookback_m)
        
        second_data= (bvol_1_ratio.rolling(lookback_m,min_periods=1).median(engine='numba')-svol_1_ratio.rolling(lookback_m,min_periods=1).median(engine='numba'))
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'vol1_ratio_%s_median'%lookback_m)
        
        second_data= bvol_chg.rolling(lookback_m,min_periods=1).mean(engine='numba')-svol_chg.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'vol_chg_%s_mean'%lookback_m)
        
        second_data= bvol_chg.rolling(lookback_m,min_periods=1).median(engine='numba')-svol_chg.rolling(lookback_m,min_periods=1).median(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'vol_chg_%s_median'%lookback_m)

    return

def load_factor5(date, fea_, SEC_FACTOR):
    bvol=0
    svol=0
    w_bvol=0
    w_svol=0
    pbvol=0
    psvol=0
    w_pbvol=0
    w_psvol=0
    max_pbvol=0
    max_psvol=0
    
    for s in range(1,6):
        pbvol_s=fea_['bv_%s'%s]/fea_['bn_%s'%s]
        pbvol+= pbvol_s
        w_pbvol+= fea_['bp_%s'%s]*pbvol_s
        bvol+= fea_['bv_%s'%s]
        w_bvol+= fea_['bp_%s'%s]*fea_['bv_%s'%s]
        max_pbvol=(max_pbvol+pbvol_s+np.abs(max_pbvol-pbvol_s))/2
        
        psvol_s=fea_['sv_%s'%s]/fea_['sn_%s'%s]
        psvol+= psvol_s
        w_psvol+= fea_['sp_%s'%s]*psvol_s
        svol+= fea_['sv_%s'%s]
        w_svol+= fea_['sp_%s'%s]*fea_['sv_%s'%s]
        max_psvol=(max_psvol+psvol_s+np.abs(max_psvol-psvol_s))/2
        
    avg_price=(fea_['bp_1']+fea_['sp_1'])/2
    
    pb_vwap=w_pbvol/pbvol
    ps_vwap=w_psvol/psvol
    b_vwap=w_bvol/bvol
    s_vwap=w_svol/svol
    
    pb_imb= (avg_price-pb_vwap)/avg_price
    ps_imb= (ps_vwap-avg_price)/avg_price
    b_imb= (avg_price-b_vwap)/avg_price
    s_imb= (s_vwap-avg_price)/avg_price
    
    max_pvol_ratio= (max_pbvol-max_psvol)/(max_pbvol+max_psvol)

    ############################
    for lookback_m in [5,15,30,60]:
        second_data= pb_imb.rolling(lookback_m,min_periods=1).mean(engine='numba')-ps_imb.rolling(lookback_m,min_periods=1).mean(engine='numba')
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'PVWAP_IMB_%s_mean'%lookback_m)
        
        second_data=pb_imb.rolling(lookback_m,min_periods=1).corr(b_imb)-ps_imb.rolling(lookback_m,min_periods=1).corr(s_imb)
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'vn_%s_corr'%lookback_m)
        
        second_data=max_pvol_ratio.rolling(lookback_m,min_periods=1).mean()
        SEC_FACTOR.divid_save_factor(second_data.loc[sec_params.sec_list], 'pvol_ratio_%s_mean'%lookback_m)

    return

def calc_orderbook_final(date):
    #初始化
    factor_list = ['OFI_gear1_5_mean', 'OFI2_gear1_5_mean', 'OFI_gear1_5_std', 'OFI2_gear1_5_std', 'OFI_gear1_5_skew', 'OFI2_gear1_5_skew', 'OFI_gear1_15_mean', 'OFI2_gear1_15_mean', \
        'OFI_gear1_15_std', 'OFI2_gear1_15_std', 'OFI_gear1_15_skew', 'OFI2_gear1_15_skew', 'OFI_gear1_30_mean', 'OFI2_gear1_30_mean', 'OFI_gear1_30_std', 'OFI2_gear1_30_std', 'OFI_gear1_30_skew', \
        'OFI2_gear1_30_skew', 'OFI_gear1_60_mean', 'OFI2_gear1_60_mean', 'OFI_gear1_60_std', 'OFI2_gear1_60_std', 'OFI_gear1_60_skew', 'OFI2_gear1_60_skew', 'MCI_IMB_5', 'VWAP_IMB_5_mean', \
        'VWAP_IMB_5_std', 'VWAP_IMB_5_skew', 'MCI_IMB_15', 'VWAP_IMB_15_mean', 'VWAP_IMB_15_std', 'VWAP_IMB_15_skew', 'MCI_IMB_30', 'VWAP_IMB_30_mean', 'VWAP_IMB_30_std', 'VWAP_IMB_30_skew', \
        'MCI_IMB_60', 'VWAP_IMB_60_mean', 'VWAP_IMB_60_std', 'VWAP_IMB_60_skew', 'OBS_50quan_5', 'OBS_80quan_5', 'OBS2_80quan_5', 'OBS_50quan_15', 'OBS_80quan_15', 'OBS2_80quan_15', 'OBS_50quan_30', \
        'OBS_80quan_30', 'OBS2_80quan_30', 'OBS_50quan_60', 'OBS_80quan_60', 'OBS2_80quan_60', 'slope_5_mean', 'slope_5_median', 'vol1_ratio_5_median', 'vol_chg_5_mean', 'vol_chg_5_median', 'slope_15_mean', \
        'slope_15_median', 'vol1_ratio_15_median', 'vol_chg_15_mean', 'vol_chg_15_median', 'slope_30_mean', 'slope_30_median', 'vol1_ratio_30_median', 'vol_chg_30_mean', 'vol_chg_30_median', 'slope_60_mean', \
        'slope_60_median', 'vol1_ratio_60_median', 'vol_chg_60_mean', 'vol_chg_60_median', 'PVWAP_IMB_5_mean', 'vn_5_corr', 'pvol_ratio_5_mean', 'PVWAP_IMB_15_mean', 'vn_15_corr', 'pvol_ratio_15_mean', \
        'PVWAP_IMB_30_mean', 'vn_30_corr', 'pvol_ratio_30_mean', 'PVWAP_IMB_60_mean', 'vn_60_corr', 'pvol_ratio_60_mean']
    SEC_FACTOR = sec_combine_factor(date, preload_factor_list=True, factor_list=factor_list)

    #读取数据
    fea_ = read_orderbook_data(date)
    #计算因子
    load_factor1(date, fea_, SEC_FACTOR)
    load_factor2(date, fea_, SEC_FACTOR)
    load_factor3(date, fea_, SEC_FACTOR)
    load_factor4(date, fea_, SEC_FACTOR)
    load_factor5(date, fea_, SEC_FACTOR)
    return SEC_FACTOR

if __name__ == '__main__':
    #初始化
    factor_group_name = 'ysw_orderbook_factorall1'
    factor_path = r'/home/datamake118/datamake118_base/nas0'
    assert os.path.exists(factor_path)

    now_date_list = os.listdir(os.path.join(factor_path, factor_group_name))
    now_date_list = sorted([x for x in now_date_list if '.' not in x])
    end_date = sys.argv[1]
    target_date_list = get_datelist('20240430',end_date)
    update_date_list = [x for x in target_date_list if x not in now_date_list]

    workers = int(sys.argv[2])
    dl = DataLoaderX(SDataset(update_date_list,calc_orderbook_final),collate_fn=lambda x:x[0],batch_size=1,num_workers=workers,shuffle=False,drop_last=False)
    
    for batch in tqdm(dl,desc='正在更新因子：%s'%factor_group_name,total=len(dl)):
        SEC_FACTOR = batch
        SEC_FACTOR.save_factor(factor_group_name, factor_path)