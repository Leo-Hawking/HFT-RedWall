
import os 
import sys
work_path=r'/home/datamake118/equity_min_strategy'
sys.path.append(work_path)
import MinBaseFactorFramework as ff
from argparse import ArgumentParser
from MinBaseFactorFramework.global_config import GlobalConfig
from MinBaseFactorFramework.fac_exp_exec import *
from MinBaseFactorFramework.util_function import *
from MinBaseFactorFramework.torch_operators import *
from MinBaseFactorFramework.save_support import SDataset
from MinBaseFactorFramework.backtest import *
from MinBaseFactorFramework.torch_operators import *
import re


def parse_args():     #对gp参数进行管理
    parser = ArgumentParser()
    #抽样及训练相关
    parser.add_argument('--start_date', default='20220104')
    parser.add_argument('--end_date', default='20221231')
    parser.add_argument('--start_time',default=93000)
    parser.add_argument('--end_time',default=100000)
    parser.add_argument('--large_sample_num',default=60)
    parser.add_argument('--small_sample_num',default=20)
    parser.add_argument('--stock_sample_num',default=None)
    parser.add_argument('--freq', default= '60s')
    parser.add_argument('--factor_input_ratio',default= 0.8)
    parser.add_argument('--num_workers',default= 20)
    
    #过检阈值相关
    parser.add_argument('--ic_thres', default=0.03)
    parser.add_argument('--ls_thres', default=0.003)
    parser.add_argument('--ls_weight', default = 10)
    parser.add_argument('--score_thres', default=0.05)
    parser.add_argument('--decorr', default=False)
    parser.add_argument('--corr_thres1', default=0.6)
    parser.add_argument('--corr_thres2', default=0.7)
    parser.add_argument('--nan_sum', default=5)  #最后输出允许全为空的截面数目
    parser.add_argument('--parsimony_coef', default=0.0001)
    
    #Gplearn参数
    parser.add_argument('--gens', default=4)
    parser.add_argument('--populations', default = 20)
    parser.add_argument('--delay_list', default=[1, 3, 5, 15])  #
    parser.add_argument('--quan_list', default=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]) 
    parser.add_argument('--exp_list', default=[-3, -2, -1, -0.5, 0.5, 2, 3])
    parser.add_argument('--min_depth', default=3)
    parser.add_argument('--max_depth', default=12) 
    parser.add_argument('--p_crossover', default=0.9) 
    parser.add_argument('--tournament_size', default=50) 
    parser.add_argument('--p_mutation', default=0.1) 
    parser.add_argument('--init_method', default='half and half') 
    parser.add_argument('--init_depth', default=[2,6]) 
    
    #计算适应度相关
    parser.add_argument('--label', type=str, default='1d') #使用的label
    parser.add_argument('--group_num', default=50) #使用的label
    
    # Handle kaggle platform
    args, unknown = parser.parse_known_args(args=[])
    return args

def get_file_name(args):
    return '-'.join(filter(None, [  # Remove empty string by filtering
        f'start{args.start_date}',
        f'end{args.end_date}',
        f'start_t{args.start_time}',
        f'end_t{args.end_time}',
        f'ic{args.ic_thres}',
        f'ls{args.ls_thres}',
        f'gens{args.gens}',
        f'pop{args.populations}',
        f'decorr{args.decorr}',
        f'label{args.label}',
    ])).replace(' ', '')


class GpConfig:
    args=parse_args()
    
    gp_result_path=rf'{work_path}/gp_result'
    summary_path = gp_result_path+r'/LeaderBoard_%s'%get_file_name(args)
    os.makedirs(summary_path,exist_ok=True)
    
    date_list_all=get_all_trade_days()
    sec_list_all=list(GlobalConfig.trade_time_loc_dict.keys())
    sec_num_all=len(sec_list_all) #每天总共的时间戳个数
     #抽样起始日期、
     
    start_ind=GlobalConfig.trade_time_loc_dict[int(args.start_time*1000)]
    end_ind=GlobalConfig.trade_time_loc_dict[int(args.end_time*1000)]
    intraday_length=end_ind-start_ind
     
     
# def get_factor_names():
#     factor_names = load_pickle('/home/datamake118/datamake118_base/nas4/Stock60sData/base_list.pkl').tolist()
#     return factor_names + ['Vol','Amount']


GpConfig.date_list=[date for date in GpConfig.date_list_all if date>=GpConfig.args.start_date and date<=GpConfig.args.end_date]
# factor_names = get_factor_names()

# factor_names_all = factor_names + [GpConfig.args.label]

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

# def formula_to_value(formula, factor_dict, date_num):
#     func = my_compile(formula)
#     result = func(factor_dict, date_num)
#     # print(formula)
#     # print(result[0])
#     return torch.stack(result)
