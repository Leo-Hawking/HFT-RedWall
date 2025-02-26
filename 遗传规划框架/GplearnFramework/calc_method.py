#%%
import operator
import traceback
import sys
import gc
from my_deap import creator, base, tools, gp #遗传规划函数
from basic_functions import *
from gp_functions import *
from gp_config import *    #基本配置
import pickle as pkl
import re 
import os


#%%
# def get_ic_hedgeret_result(formula, FS_dict):
#     ic_list, ric_list, hedgeret_list, styleexp_list = [], [], [], []
#     for date in FS_dict.keys():
#         ic, ric, hedgeret, styleexp = FS_dict[date].get_evaluation_result(formula)
#         ic_list.append(ic)
#         ric_list.append(ric)
#         hedgeret_list.append(hedgeret)
#         styleexp_list.append(styleexp)
#     ic_score = np.array(ic_list).mean()
#     ric_score = np.array(ric_list).mean()
#     hedgeret_score = np.array([x[-1]-x[0] for x in hedgeret_list]).mean()
#     hedgeret_score = hedgeret_score * (2*np.sign(ic_score)-1)
#     longret_score = np.array([x[{0:0,1:-1}[int(ic_score>0)]] for x in hedgeret_list]).mean()
#     style_location = sum(styleexp_list) / len(styleexp_list)
#     return ic_score, ric_score, hedgeret_score, longret_score, style_location

def fit_evaluation(args, FS_dict, individual):
    #fac_exp=str(individual)
    # gp_factor_sample = exec_fac_exp(base_data_slice_list, fac_exp)
    # ic, hedge_ret = calc_ic_hedge_ret(gp_factor_sample, base_data_slice_list, group_num=args.group_num)
    
    # func = my_compile(expr=individual)   # 因子值计算函数  # 先计算小样本的ic
    # gp_factor_sample = func(factor_sample_dict)
    fac_exp=str(individual)
    # try:
    length=calc_formula_length(fac_exp)
    
    if length<2 or length>15:  #长度过短或过长，直接淘汰
        return np.array([0,0,0,0,0]), np.array([[0,0]]*5)

    ic, hedgeret, longret, style_location = get_ic_hedgeret_result(fac_exp, FS_dict)
    
    # if abs(ic) >= args.ic_thres or hedge_ret>=args.ls_thres:  #小样本过检，对大样本重新计算一次更新
    #     gp_factor = formula_to_value(fac_exp,factor_dict, args.large_sample_num)
    #     ic, hedge_ret= calc_ic_hedge_ret(gp_factor, factor_dict, label_name=args.label, group_num=args.group_num)
    
    # #计算相关性
    # gp_factor_sample = sample_to_corr(gp_factor_sample)
    # hist_corr_list = []
    # print(len(TempData.factor_values_hist))
    # for fac_values in TempData.factor_values_hist.values():
    #     hist_corr_list.append(cs_corr(fac_values, gp_factor_sample).nanmean())
        
    # max_corr=torch.tensor(0)
    
    abs_ic = ic * (2*np.sign(ic)-1)
    
    # 核心的score函数
    # score = abs_ic + hedge_ret*args.ls_weight - length * args.parsimony_coef  #初始打分
    score = max(abs_ic, hedgeret*args.ls_weight) - length * args.parsimony_coef  #初始打分

    # if len(hist_corr_list)>0:
    # #if True:
    #     #nan_num=np.sum(np.isnan(hist_corr_list))
    #     #nan_ratio=nan_num/len(hist_corr_list)  
    #     max_corr=torch.max(torch.abs(torch.stack(hist_corr_list)))
        
    #     if max_corr > args.corr_thres2: #过高重复，直接清除该个体
    #         return (0, 0, 0, 0, 0, 0)
        
    #     if (max_corr > args.corr_thres1 and abs_ic < 2*args.ic_thres and (not hedge_ret >2*args.ls_thres)):  
    #         score=min(torch.tensor(args.score_thres)-0.005,score)
    #         abs_ic=min(torch.tensor(args.ic_thres)-0.005,abs_ic)
    return torch.tensor([score, abs_ic, hedgeret, ic, longret]).numpy(), style_location.numpy()

# def calc_ic_hedge_ret(fac_data,factor_dict,
#                       label_name='1d_exret',
#                       group_num=50):
#     '''遗传规划中计算适应度的函数'''
    
#     date_num=fac_data.shape[0]
    
#     # for i in range(date_num): fac_data[i] = fac_data[i][30:, :]

#     # for i in range(date_num):
#     #     print('fac_data.shape:', fac_data[i][30:, :].shape)
#     #     print('factor_dict.shape:', factor_dict[label_name][i].shape)
#     #     # print(fac_data[i][30:, :])
#     ic = torch.stack([cs_corr(fac_data[i][30:, :][GpConfig.start_ind:GpConfig.end_ind, :],
#     factor_dict[label_name][i][GpConfig.start_ind:GpConfig.end_ind, :]) for i in range(date_num)]).nanmean()
#     # print('ic:')
#     # print(torch.stack([cs_corr(fac_data[i][30:, :],factor_dict[label_name][i]) for i in range(date_num)]))
#     # print(torch.stack([cs_corr(fac_data[i][30:, :],factor_dict[label_name][i]) for i in range(date_num)]))

#     hedge_ret=[]
#     for i in range(date_num):
#         group_mean = cs_group_mean(
#             fac_data[i][30:, :][GpConfig.start_ind:GpConfig.end_ind, :],
#             factor_dict[label_name][i][GpConfig.start_ind:GpConfig.end_ind, :],
#             None, group_num)
#         hedge_ret.append((group_mean[:,-1]-group_mean[:,0])/2)
#     hedge_ret=torch.stack(hedge_ret).nanmean()
#     return ic, hedge_ret*torch.sign(ic)

def calc_nan_proportion(factor):
    count_tensor=torch.isfinite(factor).sum(axis=-1)
    return (count_tensor==0).sum()

# def fit_evaluation(args, factor_dict, factor_sample_dict, individual):
#     #fac_exp=str(individual)
#     # gp_factor_sample = exec_fac_exp(base_data_slice_list, fac_exp)
#     # ic, hedge_ret = calc_ic_hedge_ret(gp_factor_sample, base_data_slice_list, group_num=args.group_num)
    
#     # func = my_compile(expr=individual)   # 因子值计算函数  # 先计算小样本的ic
#     # gp_factor_sample = func(factor_sample_dict)
#     fac_exp=str(individual)
#     # try:
#     length=calc_formula_length(fac_exp)
    
#     if length<2 or length>15:  #长度过短或过长，直接淘汰
#         return (0, 0, 0, 0, 0, 0)
    
#     gp_factor_sample = formula_to_value(fac_exp,factor_sample_dict, args.small_sample_num)
#     nan_sum=calc_nan_proportion(gp_factor_sample[0])  #挑第一日的截面计算缺失比例
#     nan_ratio=nan_sum/GpConfig.intraday_length
    
#     if nan_sum> args.nan_sum*3:
#         return (0, 0, 0, 0, 0, 0)  #针对早盘进行训练，因此要求不能有太多的空值
    
#     ic, hedge_ret= calc_ic_hedge_ret(gp_factor_sample, factor_sample_dict, label_name=args.label, group_num=args.group_num)               
#     print('ic_init:', ic)
#     print('hedge_ret_init:', hedge_ret)
    
#     if abs(ic) >= args.ic_thres or hedge_ret>=args.ls_thres:  #小样本过检，对大样本重新计算一次更新
#         gp_factor = formula_to_value(fac_exp,factor_dict, args.large_sample_num)
#         ic, hedge_ret= calc_ic_hedge_ret(gp_factor, factor_dict, label_name=args.label, group_num=args.group_num)
    
#     if hedge_ret < args.ls_thres/3:  #多空太差，直接淘汰
#         return (0, 0, 0, 0, 0, 0)
    
#     #计算相关性
#     gp_factor_sample = sample_to_corr(gp_factor_sample)
#     hist_corr_list = []
#     print(len(TempData.factor_values_hist))
#     for fac_values in TempData.factor_values_hist.values():
#         hist_corr_list.append(cs_corr(fac_values, gp_factor_sample).nanmean())
        
#     max_corr=torch.tensor(0)
    
#     abs_ic = abs(ic) 
    
#     # 核心的score函数
#     # score = abs_ic + hedge_ret*args.ls_weight - length * args.parsimony_coef  #初始打分
#     score = max(abs_ic, hedge_ret*args.ls_weight) - length * args.parsimony_coef  #初始打分

#     if len(hist_corr_list)>0:
#     #if True:
#         #nan_num=np.sum(np.isnan(hist_corr_list))
#         #nan_ratio=nan_num/len(hist_corr_list)  
#         max_corr=torch.max(torch.abs(torch.stack(hist_corr_list)))
        
#         if max_corr > args.corr_thres2: #过高重复，直接清除该个体
#             return (0, 0, 0, 0, 0, 0)
        
#         if (max_corr > args.corr_thres1 and abs_ic < 2*args.ic_thres and (not hedge_ret >2*args.ls_thres)):  
#             score=min(torch.tensor(args.score_thres)-0.005,score)
#             abs_ic=min(torch.tensor(args.ic_thres)-0.005,abs_ic)
            
#     print(fac_exp)
#     print(torch.stack([score, abs_ic, hedge_ret, ic, max_corr, nan_ratio]).numpy().tolist())
#     return torch.stack([score, abs_ic, hedge_ret, ic, max_corr, nan_ratio]).numpy()
    
#     # except Exception as e:
#     #     print("An error occurred:", e)
#     #     traceback_info = traceback.extract_tb(sys.exc_info()[2])
#     #     filename, line_number, function_name, text = traceback_info[-1]
#     #     print("The wrong formula is: ", fac_exp)
#     #     print("Exception occurred in file:", filename)
#     #     print("On line number:", line_number)
#     #     print("In function:", function_name)
#     #     print("Exception line:", text)
#     #     del traceback_info
#     #     gc.collect()
#     #     torch.cuda.empty_cache()