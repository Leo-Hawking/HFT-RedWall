#%%
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from datetime import datetime
import operator
from multiprocessing import Pool
import re
import sys
sys.path.append(r'/home/datamake118/equity_min_strategy/GplearnFramework')
from tqdm import tqdm
import joblib
import random
import math
import torch
from gp_config import *    #基本配置
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore")

#%%
'''
读写函数
'''
# def my_compile(expr):
#     '''
#     编译表达式，将因子名替换成可执行的数据即可，算子名直接从导入的函数中读取。
#     1、直接计算：        e.g., lambda factor_dict:abs(factor_dict['bp1'])
#     2、沿着第一维循环计算：  e.g., lambda factor_dict,date_num:[abs(factor_dict['bp1'][i]) for i in range(date_num)]
#     '''
#     code = str(expr)
#     # 利用对应关系转换成可执行的代码func_map_dict.keys()
#     code_split = re.split('([\(,\)\s])', code) 
#     #code_transform_list = ['factor_dict[\'%s\']' %j if j in factor_names else j for j in code_split]
#     code_transform_list = ['factor_dict[\'%s\'][i]' %j if j in factor_names else j for j in code_split]  #第二种读取方式
#     code_transform = ''.join(code_transform_list)
#     #return eval('lambda factor_dict:' + code_transform)   
#     return eval(rf'lambda factor_dict,date_num:[{code_transform} for i in range(date_num)]')   
def get_factor_names():
    factor_names = load_pickle('/home/datamake118/datamake118_base/nas4/Stock60sData/base_list.pkl').tolist()
    return factor_names + ['Vol','Amount']

def formula_to_value(formula, FS_dict):
    return {date: FS_dict[date].get_factor_value(formula) for date in FS_dict.keys()}

def get_FS_dict(date_num=None):
    FS_dict = {}
    date_list_all = get_all_trade_days()
    date_list_all = [x for x in date_list_all if x>=GpConfig.args.start_date and x<=GpConfig.args.end_date]
    if date_num is None:
        date_list = date_list_all
    else:
        date_list = sorted(random.sample(date_list_all, date_num))
    for date in date_list:
        FS_dict[date] = Gp_Factor_Structure(date, None)
    return FS_dict

def get_ic_hedgeret_result(formula, FS_dict):
    ic_list, hedgeret_list, styleexp_list = [], [], []
    for date in FS_dict.keys():
        ic, hedgeret, styleexp = FS_dict[date].get_evaluation_result(formula)
        ic_list.append(ic)
        hedgeret_list.append(hedgeret)
        styleexp_list.append(styleexp)
    ic_score = np.array(ic_list).mean()
    hedgeret_score = np.array([x[-1]-x[0] for x in hedgeret_list]).mean()
    hedgeret_score = hedgeret_score * (2*np.sign(ic_score)-1)
    longret_score = np.array([x[{0:0,1:-1}[int(ic_score>0)]] for x in hedgeret_list]).mean()
    
    # 假设模型能够学到每一个非线性分组，那么style_location是基于收益argsort后的style位置
    style_location = sum(styleexp_list) / len(styleexp_list)
    hedgeret_argsort = torch.argsort(torch.vstack(
        [x.unsqueeze(0) for x in hedgeret_list]).mean(dim=0), descending=False
    )
    style_location = style_location[:,hedgeret_argsort]
    return ic_score, hedgeret_score, longret_score, style_location

def calc_distance(ind1, ind2):
    '''
    输入：两个二维tensor
    输出：两个tensor之间的几何距离
    '''
    return torch.norm(ind1-ind2, p=2)

'''
数据结构体
'''
class Gp_Factor_Structure():
    def __init__(self, date:str, stk_num:int) -> None:
        self.date = date
        self.generator =  Factor_data(date)
        self.Backtester = Backtest_Framework(10, 1, date, date, label_list=[GpConfig.args.label], preload_factor=False)
        if stk_num is not None:
            stock_all = self.generator.stock_list
            stock_use = random.sample(stock_all, stk_num)
            self.generator = Factor_data(date, stock_list = stock_use)

    def get_factor_value(self, fac_exp):
        factor_generator = self.generator
        return self.generator.exec_fac_exp_daily_single(fac_exp)

    def get_evaluation_result(self, fac_exp):
        self.Backtester.eval_fac_sec_hs_all_fast(
            fac_exp,
            {fac_exp:fac_exp},
            load_factor=False,
            workers=0,
            show_tqdm=False,
        )
        time_idx = list(range(GpConfig.start_ind, GpConfig.end_ind))
        return [
            self.Backtester.ic_result[0,0,0,time_idx].nanmean().item(),
            self.Backtester.hedgeret_result[0,0,0,time_idx,:].nanmean(dim=0),
            self.Backtester.styleexp_result[0,:,0,time_idx,:].nanmean(dim=1),
        ]

class TempData:
    '''全局访问，用来存储一些中间数据'''
    factor_values_hist={} 

class PopDataset(torch.utils.data.Dataset):  
    '''
    多进程计算适应度
    '''
    def __init__(self, pop, evaluate_func) -> None:
        self.pop = pop
        self.evaluate_func = evaluate_func

    def __getitem__(self, index):
        ind = self.pop[index]
        fitness = self.evaluate_func(ind)
        return (index,fitness)

    def __len__(self):
        return len(self.pop)

class LoadDataset(torch.utils.data.Dataset):
    '''
    读取数据部分，从内存映射中切片
    '''
    def __init__(self,args,date_list,load_data,stk_num) -> None:
        self.args = args
        self.date_list = date_list
        self.load_data = load_data
        self.stk_num=stk_num

    def __getitem__(self, index):
        date = self.date_list[index]
        factor_td = self.load_data(self.args,date,self.stk_num)
        return date, factor_td

    def __len__(self):
        return len(self.date_list)
    
class GPDataset(torch.utils.data.Dataset):
    def __init__(self,round_list,factor_sample_dict,function) -> None:
        self.round_list = round_list
        self.factor_sample_dict = factor_sample_dict
        self.function=function

    def __getitem__(self, index):
        r = self.round_list[index]
        return self.function(r, self.factor_sample_dict)

    def __len__(self):
        return len(self.round_list)

# def load_factor_group(args,factor_name,date,stock_ind=None):
#     '''
#     不同类型的基础秒频因子,从所有字段的内存映射中直接读取
#     （已弃用）
#     '''
#     date_ind = GlobalConfig.trade_date_loc_dict[date]
#     start_ind=date_ind*GpConfig.sec_num_all+GpConfig.start_ind
#     end_ind=date_ind*GpConfig.sec_num_all+GpConfig.end_ind
    
#     ind_slice=slice(start_ind,end_ind)
#     #return Base_data_dict[factor_name][ind_slice,:] if stock_ind is None else Base_data_dict[factor_name][ind_slice,stock_ind] #同时切片和索引速度很慢
#     return Base_data_dict[factor_name][ind_slice] if stock_ind is None else Base_data_dict[factor_name][ind_slice][:,stock_ind] #先切片，再索引更快

## 加载因子
# def load_data_daily(args,date,stk_num):
#     '''读取每日数据'''
#     #pre_date=get_shift_trade_date(date)
#     factor_generator = Factor_data(date)
#     if stk_num is not None:
#         # zz1000_td_valid = GlobalConfig.zz1000_index.loc[date].dropna().index.tolist()
#         # stockid_used = random.sample(zz1000_td_valid, stk_num)  #每天重新打乱抽取
#         # stockid_used.sort()
#         # stock_ind=[GlobalConfig.trade_code_loc_dict['%06d'%int(code)] for code in stockid_used]
#         stock_all = factor_generator.stock_list
#         stock_use = random.sample(stock_all, stk_num)
#         factor_generator = Factor_data(date, stock_list = stock_use)
#         #注意，code都为六位字符串形式
    
#     # factor_generator = Factor_data(date, stock_list = stockid_used)

#     factor_td_dict = {}
#     for fac in factor_generator.factor_list:
#         factor_td_dict[fac] = factor_generator[fac]
    
#     for label in ['1d', '3d']:
#         factor_td_dict[label] = factor_generator.load_ret(label)
#         # print(factor_td_dict[label])

#     return factor_td_dict

# def get_factor_dict(args):
#     '''
#     large_sample_frac: 大样本天数
#     stk_nums: 每天选取样本股票数量
#     small_sample: 抽取小样本天数，用于加速计算
#     '''
#     large_sample_dates = random.sample(GpConfig.date_list, args.large_sample_num)
#     small_sample_dates=  large_sample_dates[:args.small_sample_num]
    
#     factor_dict = {factor_name:[] for factor_name in factor_names_all}
#     factor_sample_dict = {factor_name:[] for factor_name in factor_names_all}

#     dataloader=DataLoader(LoadDataset(args,large_sample_dates,load_data_daily,args.stock_sample_num),
#                           collate_fn=lambda x:x[0], batch_size=1,
#                           num_workers=args.num_workers,shuffle=False,drop_last=False)
#     for batch in tqdm(dataloader,total=len(large_sample_dates)):
#         date, factor_td = batch
#         for factor_name in factor_names_all:
#             factor_dict[factor_name].append(factor_td[factor_name])
#             if date in small_sample_dates:
#                 factor_sample_dict[factor_name].append(factor_td[factor_name])
    
#     factor_dict={key:torch.stack(value) for key,value in factor_dict.items()}
#     factor_sample_dict={key:torch.stack(value) for key,value in factor_sample_dict.items()}
#     return factor_dict, factor_sample_dict

def get_factor_sample(Round, FS_dict):
    global TempData
    formula_df = pd.read_csv(os.path.join(GpConfig.summary_path, 'GP_Formula_Round_%s.csv'%Round))
    print(formula_df)
    if len(formula_df) == 0:
        return
    else:
        gen_list = formula_df.index.tolist()
        factor_gens_used = gen_list     #读取所有的相关性矩阵用于个体生成适应度计算，防止重复！
        for g in factor_gens_used:
            formula = list(formula_df['formula'])[g]
            if formula not in TempData.factor_values_hist.keys(): TempData.factor_values_hist[formula] = {}
            for date in FS_dict.keys():
                TempData.factor_values_hist[formula][date] = FS_dict[date].get_factor_value(formula)
        print(len(TempData.factor_values_hist))
    return

def update_gp_factor_hist(FS_dict):
    """ 导出所有随机数下的gp因子 """
    global TempData
    file_list = os.listdir(GpConfig.summary_path)
    round_list = [file[17:-4] for file in file_list]
        
    # dataloader=DataLoader(GPDataset(round_list,factor_sample_dict,get_factor_sample),collate_fn=lambda x:x[0],batch_size=1,num_workers=GpConfig.args.num_workers,shuffle=False,drop_last=False)
    # for batch in tqdm(dataloader,total=len(round_list)):
    #     pass
    # 非并行形式，无需使用 DataLoader
    for round_item in tqdm(round_list, total=len(round_list)):
        get_factor_sample(round_item, FS_dict)

    print(len(TempData.factor_values_hist))
    return

def sample_to_corr(factor): 
    '''对数据抽样，转成二维数据'''
    
    
    return factor.reshape(-1,factor.shape[-1])[::12]

def calc_formula_length(formula):  
    '''识别公式长度: num('(')+num(',')  '''
    code_list=re.split('([\(,\)\s])', formula)
    return len([x for x in code_list if x in ['(',',']])

# def fill_inf(arr):
#     arr = arr.copy()
#     arr[np.isinf(arr)] = 0
#     return arr

# def fill_all(arr):
#     arr = arr.copy()
#     arr[np.isinf(arr)|np.isnan(arr)] = 0
#     return arr

# def fill_inf_new(arr):
#     arr = arr.copy()
#     arr[np.isinf(arr)] = np.nan
#     return arr

#%%
if __name__ == '__main__':
    # 测试读写速度
    fac_exp = "ts_delay(ts_wma(OrderAmtSum_B, 30), 12)"
    date_list_all = get_all_trade_days()
    date_list = [x for x in date_list_all if x >= GpConfig.args.start_date and x <= GpConfig.args.end_date]
    ###
    date_list = date_list[:20]
    FS_dict = {}
    for date in date_list:
        FS_dict[date] = Gp_Factor_Structure(date, None)
    get_ic_hedgeret_result(fac_exp, FS_dict)
# %%
