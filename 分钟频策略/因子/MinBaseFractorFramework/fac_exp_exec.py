import pandas as pd
import numpy as np
import os
import warnings
import math
from tqdm import *
import random
import pickle
import torch
from typing import List
# from .base_data_config import Base_data_dict
from .global_config import GlobalConfig
from .torch_operators import *
from .save_support import min_params, min_combine_factor
from .util_function import *
from torch.utils.data import Dataset, DataLoader
import gc
import sys
import traceback
import h5py
import statsmodels.api as sm

# 设置环境变量，避免KMP重复加载库
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# 加载pickle文件
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 从因子表达式中提取使用的基础因子名称
def get_base_data_name_used_in_fac_exp(fac_exp: str, data_dict):
    name_list = [name for name in data_dict.factor_list if name in fac_exp]
    return name_list

# 使用MAD去极值，限制数据的异常值
def mad_winsorize(x: np.array, multiplier: int = 5):
    x_M = np.nanmedian(x)  # 计算中位数
    x_MAD = np.nanmedian(np.abs(x - x_M))  # 计算中位绝对偏差
    upper = x_M + multiplier * x_MAD  # 上限
    lower = x_M - multiplier * x_MAD  # 下限
    x[x > upper] = upper  # 大于上限的值设为上限
    x[x < lower] = lower  # 小于下限的值设为下限
    return x

# 因子预处理：去极值、填充缺失值、标准化
def clean_factor(factor_in: pd.DataFrame, fillna: bool = False):
    factor = factor_in.astype('float')  # 转换为浮动类型
    res = factor.copy()  # 复制数据
    res[:] = np.nan  # 初始化为NaN
    # 对每一天的因子数据进行去极值处理
    for date in factor.index:
        factor_td = factor.loc[date].values
        res.loc[date] = mad_winsorize(factor_td, multiplier=5)  # 对当天因子值去极值
    # 前向填充缺失值，最多填充10个时间点
    res = res.fillna(method='ffill', limit=10)
    # 进行标准化处理（去均值，除以标准差）
    res = res.sub(res.mean(1), 0).div(res.std(1), 0)
    if fillna:
        res.fillna(0, inplace=True)  # 如果需要，填充所有缺失值为0
    return res

# 通过等权法合成多个因子
def syn_factor_ew(factor_in_list, fillna: bool = False):
    res = clean_factor(factor_in_list[0], fillna=fillna)  # 处理第一个因子
    if len(factor_in_list) > 1:
        for factor_in in factor_in_list[1:]:
            factor = clean_factor(factor_in, fillna=fillna)
            res += factor  # 累加因子
    # 再次进行标准化处理
    res = res.sub(res.mean(1), 0).div(res.std(1), 0)
    return res

# 加载风格因子数据
def load_style_factor():
    '''
    print('正在读取风格因子')
    style_path = GlobalConfig.style_path
    style_file_dict = {}
    for style_factor in ['fac_alpha', 'fac_atvr', 'fac_beta', 'fac_cmra', 'fac_daily_std', 'fac_hist_sigma', 'fac_ind_mom', 'fac_lncap', \
        'fac_midcap', 'fac_rs', 'fac_stoa', 'fac_stom', 'fac_stoq', 'fac_strev']:
        style_file = pd.read_feather(os.path.join(style_path, style_factor+'.fea')).set_index('index')
        style_file.columns = style_file.columns.astype(int)
        style_file_dict[style_factor] = style_file

    fac_size = syn_factor_ew([style_file_dict['fac_lncap']])
    fac_mid_size = syn_factor_ew([style_file_dict['fac_midcap']])
    fac_volatility = syn_factor_ew([style_file_dict['fac_beta'], style_file_dict['fac_hist_sigma'], style_file_dict['fac_daily_std'], style_file_dict['fac_cmra']])
    fac_liquidity = syn_factor_ew([style_file_dict['fac_stom'], style_file_dict['fac_stoq'], style_file_dict['fac_stoa'], style_file_dict['fac_atvr']])
    fac_momentum = syn_factor_ew([style_file_dict['fac_strev'], style_file_dict['fac_ind_mom'], style_file_dict['fac_rs'], style_file_dict['fac_alpha']])
    return {
        'size': fac_size,
        'mid_size': fac_mid_size,
        'volatility': fac_volatility,
        'liquidity': fac_liquidity,
        'momentum': fac_momentum
    }
    '''
    # 从pickle文件中加载风格因子字典
    style_dict = pickle.load(open(rf"{GlobalConfig.support_data_path}/style_factor_dict.pkl", 'rb'))
    return style_dict

############################################
# 数据类定义
############################################

# 用于存储风格因子的全局字典
class GlobalStyleDict:
    style_dict = {}  # 静态字典，保存风格因子

# 风格因子数据获取类
class Ret_data:
    def __init__(self, date, _dtype='float32'):
        self.date = date
        self._dtype = _dtype
        # self.period = period
        # self.stockid = self.get_ret_code()

    # 获取指定日期、周期的收益率代码（股票代码）
    def get_ret_code(self, period):
        Base_data_path = GlobalConfig.base_data_root_path2
        file_path = rf"{Base_data_path}/return_tag/{period}_exret/{self.date}.h5"
        with h5py.File(file_path,'r') as f:
            stockid = f['code'][:]
        return stockid.tolist()

    # 获取指定周期的所有收益率数据
    def get_ret_data_all(self, period):
        def _preprocess(style_fac: pd.DataFrame, stock_list: list, yd_date: str):
            # 数据预处理：按百分位排序，划分为10组（group 0-9），然后生成虚拟变量
            style_factor = style_fac.loc[yd_date].reindex(stock_list).rename('factor').to_frame()
            style_factor['group'] = style_factor['factor'].rank(pct=True, method='first', na_option='keep')
            style_factor['group'] = (style_factor['group'] * 10).fillna(0).astype(int)
            group_matrix = pd.get_dummies(style_factor['group'], prefix='group')
            style_factor = pd.concat([style_factor[['factor']], group_matrix.iloc[:, :-1]], axis=1)
            return style_factor

        def regres(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # 计算回归残差
            x_mean = x.nanmean(dim=-1, keepdim=True)
            y_mean = y.nanmean(dim=-1, keepdim=True)
            x_demean = x - x_mean
            y_demean = y - y_mean
            std = torch.nansum(torch.pow(x_demean, 2), dim=-1)
            res = (torch.nansum(x_demean * y_demean, dim=-1, keepdim=True).T / std).T
            return y - res * x
        
        if period in ['ret_size', 'ret_mid_size', 'ret_volatility', 'ret_liquidity', 'ret_momentum']:
            #读取风格收益率
            if len(GlobalStyleDict.style_dict) == 0:
                GlobalStyleDict.style_dict = load_style_factor()
            ret_data = self.get_ret_data_all('1d')
            stockid = self.get_ret_code('1d')
            all_date_list = get_all_trade_days()
            yd_date = all_date_list[all_date_list.index(self.date)-1]
            style_factor = GlobalStyleDict.style_dict[period.split('_')[1]]
            style_factor = _preprocess(style_factor, stockid, yd_date).astype(float)#.numpy()

            time = ret_data.index[0]
            resid_dict = {time:pd.Series(np.linalg.lstsq(style_factor.fillna(0).values, ret_data.loc[time].fillna(0).values)[0]) for time in ret_data.index}
            resid = pd.DataFrame({time: resid_dict[time] for time in ret_data.index}).T
            resid = ret_data - style_factor.fillna(0).values.dot(resid.T.values).T

            return resid
        
        else:
            Base_data_path = GlobalConfig.base_data_root_path2
            file_path = rf"{Base_data_path}/return_tag/{period}_exret/{self.date}.h5"
            with h5py.File(file_path,'r') as f:
                ret_data = f['ret'][:,:]
            stockid = self.get_ret_code(period)
            return pd.DataFrame(ret_data, index=GlobalConfig._sec_list, columns=stockid)

    def get_ret_data(self, idx, period):
        Base_data_path = GlobalConfig.base_data_root_path2
        file_path = rf"{Base_data_path}/return_tag/{period}_exret/{self.date}.h5"
        with h5py.File(file_path,'r') as f:
            ret_data = f['ret'][idx,:]
        stockid = self.get_ret_code(period)
        return pd.Series(ret_data, index=stockid)

    def __getitem__(self, key):
        sec, period = key
        idx = GlobalConfig._sec_list.index(sec)
        return self.get_ret_data(idx)

class Factor_data:
    def __init__(self, date, stock_list=[], _dtype='float32', preload_factor=False):
        self.comp_num = 60  # 基础数据空值长度，用于还原min_periods
        self.date = date  # 当前日期
        date_list_all = get_all_trade_days()  # 获取所有的交易日
        self.pre_date = date_list_all[date_list_all.index(date)-1]  # 获取前一个交易日的日期
        self._dtype = _dtype  # 数据类型，默认为'float32'
        self.daily_support = self.read_daily_support()  # 读取每日支持数据
        self.factor_list = self.get_factor_name()  # 获取因子名称列表
        self.factor_list.extend(['Vol','Amount'])  # 添加'Vol'和'Amount'到因子列表
        self.preload_factor = preload_factor  # 是否预加载因子
        self.preload_factor = {}  # 存储预加载因子的字典
        if len(stock_list) == 0:
            self.stock_list = self.daily_support.index.tolist()  # 使用每日支持数据中的股票列表
        else:
            self.stock_list = stock_list  # 使用传入的股票列表

    def read_daily_support(self):
        # 读取每日支持数据
        data = pd.read_csv(os.path.join(GlobalConfig.daily_support_path, self.date+'_daily_support.csv'),index_col=0)
        data.index = data.index.astype(int)  # 将索引转换为整数
        data = data.rename({'vol':'vol1'},axis=1)  # 重命名列名，'vol'改为'vol1'
        return data

    def get_factor_name(self):
        # 获取因子的名称列表
        factor_list = []
        factor_list.extend(self.daily_support.columns.tolist())  # 添加每日支持数据中的列名作为因子名
        Base_data_path = rf'{GlobalConfig.base_data_root_path2}'  # 获取基础数据的路径
        base_list_file = os.listdir(rf'{Base_data_path}/{self.date}')  # 获取基础数据目录下的文件
        base_list_file = [x for x in base_list_file if 'base_list' in x]  # 过滤出包含'base_list'的文件
        base_list = []
        for file in base_list_file:
            base_list.extend(torch.load(rf'{Base_data_path}/{self.date}/{file}'))  # 加载基础数据
        factor_list.extend(base_list)  # 将基础数据加入因子列表
        return factor_list

    def __getitem__(self, item):
        """
        根据给定的因子名称`item`，获取相应的因子数据。
        1. 如果是'Vol'或'Amount'，则根据对应逻辑计算并返回。
        2. 如果`item`已预加载，则直接返回预加载的数据。
        3. 如果`item`是daily_support中的列，则返回该列的数据。
        4. 否则，读取h5文件中的数据并返回。
        """
        # 1. 是否属于进阶字段：'Vol' 或 'Amount'
        if item in ['Vol','Amount']:
            if item == 'Vol':
                data = self['TradeVolSum_B'] + self['TradeVolSum_S']  # 'Vol'为买入和卖出交易量之和
            elif item == 'Amount':
                data = self['TradeAmtSum_B'] + self['TradeAmtSum_S']  # 'Amount'为买入和卖出成交金额之和
            return data
        
        # 2. 基础字段读取
        if self.preload_factor:
            if item in self.preload_factor.keys():  # 如果因子已经预加载，则直接返回
                return self.preload_factor[item]
        
        # 3. 从daily_support中读取字段数据
        if item in self.daily_support.columns:
            return torch.from_numpy(self.daily_support[item].values.astype(self._dtype)).to(device=GlobalConfig.use_device)

        else:  # 4. 从h5文件中读取数据
            Base_data_path = rf'{GlobalConfig.base_data_root_path2}/{self.date}'
            stockid = torch.load(rf'{Base_data_path}/stockid.pt')
            stockid = [int(x) for x in stockid]  # 将stockid转为整数列表
            with h5py.File(rf"{Base_data_path}/{item}.h5", 'r') as f:
                data = f['data'][:,:].astype(self._dtype)  # 从h5文件中读取数据
            reindexed_data = pd.DataFrame(data, columns=stockid).reindex(columns=self.stock_list).values  # 重索引数据
            comp_data = np.full((self.comp_num, len(self.stock_list)), np.nan, dtype=self._dtype)  # 填充空值
            reindexed_data = np.vstack([comp_data, reindexed_data])  # 将空值数据添加到顶部
            if self.preload_factor:
                self.preload_factor[item] = torch.from_numpy(reindexed_data).to(device=GlobalConfig.use_device)  # 缓存到预加载字典
            return torch.from_numpy(reindexed_data).to(device=GlobalConfig.use_device)

    def get_base_data_name_used_in_fac_exp(self, fac_exp: str):
        # 获取因子表达式中使用的基础数据的名称列表
        name_list = [name for name in self.factor_list if name in fac_exp]
        return name_list

    def exec_fac_exp_daily_single(self, fac_exp: str):
        """
        执行单个因子表达式（fac_exp）并返回结果。
        1. 解析因子表达式，获取所有所需的基础数据。
        2. 执行因子表达式并返回结果。
        """
        use_device = GlobalConfig.use_device
        try:
            target_data_shape = (GlobalConfig.time_num, len(self.stock_list))  # 因子结果的目标形状
            valid_step = GlobalConfig.time_num  # 有效时间步数
            fac_res = torch.empty(size=target_data_shape, device='cpu')  # 存储结果的tensor
            base_data_name_list = self.get_base_data_name_used_in_fac_exp(fac_exp)  # 获取因子表达式中使用的基础数据
            local_var_dict = locals()  # 获取局部变量字典
            for base_data_name in base_data_name_list:
                comp_data = self[base_data_name].to(device=use_device)  # 加载基础数据并移动到设备
                local_var_dict[rf"{base_data_name}"] = comp_data  # 将数据添加到局部变量字典

            # 执行因子表达式
            if ';' in fac_exp:
                exp_list = fac_exp.split(';')  # 分割多个表达式
                for exp in exp_list[:-1]:
                    exec(exp.strip())  # 执行前面的表达式
                exec(rf'temp = ' + exp_list[-1].strip())  # 执行最后一个表达式并将结果赋值给temp
            else:
                exec(rf'temp = ' + fac_exp.strip())  # 执行单个因子表达式
            
            exec(rf'fac_res[:] = temp[self.comp_num:]')  # 赋值因子结果
            gc.collect()  # 垃圾回收
            assert fac_res.shape == target_data_shape  # 确保结果形状正确
            return fac_res
        
        except Exception as e:
            # 异常处理，输出错误信息和异常位置
            print("An error occurred:", e)
            traceback_info = traceback.extract_tb(sys.exc_info()[2])
            filename, line_number, function_name, text = traceback_info[-1]
            print("Wrong factor:", fac_exp)
            print("Exception occurred in file:", filename)
            print("On line number:", line_number)
            print("In function:", function_name)
            print("Exception line:", text)
            del traceback_info
            gc.collect()

    def exec_fac_exp_daily(self, fac_exp_dict={}, if_custom_function=False, custom_function=None, if_daily=False):
        """
        按日获取因子最终输出。如果是自定义函数，则调用自定义函数执行计算。
        """
        if not if_custom_function:
            factor_list = []
            for fac_name, fac_exp in fac_exp_dict.items():
                factor_list.append(self.exec_fac_exp_daily_single(fac_exp).T.unsqueeze(dim=2))  # 执行每个因子表达式
            return torch.cat(factor_list, dim=2), list(fac_exp_dict.keys())  # 拼接因子数据并返回
        else:
            '''
            custom_function的输入：(Factor_data)
            输出：factor_data
            '''
            output, factor_list = custom_function(self)  # 调用自定义函数
            if if_daily:
                output = self.daily_to_minute(output)  # 转换为分钟级数据
            return output, factor_list

    def load_factor(self, factor_name):
        #1.定位因子包名称
        factor_pack_name = factor_name.split('_')[0]

        #2.读取因子数据
        file_path = rf"{GlobalConfig.factor_output_path}/{factor_pack_name}/{self.date}"
        with open(rf"{file_path}/factor_list.pkl","rb") as f:
            factor_list = pickle.load(f)
        factor_idx = factor_list.index(factor_name)
        stockid = torch.load(rf'{file_path}/stockid.pt')
        if type(stockid) != list:
            stockid = stockid.tolist()
        with h5py.File(rf"{file_path}/min_data.h5", 'r') as f:
            factor_data = f['data'][:,:,factor_idx].astype(self._dtype)
        factor_data = pd.DataFrame(factor_data,index=stockid,columns=GlobalConfig._sec_list).reindex(
            index=self.stock_list).T.values
        return torch.from_numpy(factor_data).to(device=GlobalConfig.use_device)

    def load_factor_group(self, factor_pack_name):
        """
        加载因子包的因子数据，支持读取不同类型的因子（如旧因子）。
        """
        if 'OLD_FACTOR_DAILY' in factor_pack_name:
            # 读取日频旧因子
            start_signal = int(factor_pack_name.split(':')[1].split('-')[0])
            end_signal = int(factor_pack_name.split(':')[1].split('-')[1])

            old_factor_path = r'/home/intern1/hft_database/nas3/daily_factor'
            with h5py.File(os.path.join(old_factor_path, self.pre_date + '.h5'), 'r') as f:
                factor_data = f['data'][:]
                factor_data = factor_data[:, start_signal:end_signal]  # 截取指定信号范围
                stockid = f['index'][:]
            factor_data = pd.DataFrame(factor_data, index=stockid).reindex(index=self.stock_list)
            # 扩展数据维度以适应分钟频率
            factor_data = torch.from_numpy(factor_data.values).float().unsqueeze(dim=1).repeat(1, GlobalConfig.time_num, 1).to(device=GlobalConfig.use_device)
        
        elif 'OLD_FACTOR' in factor_pack_name:
            # 读取分钟频旧因子
            start_signal = int(factor_pack_name.split(':')[1].split('-')[0])
            end_signal = int(factor_pack_name.split(':')[1].split('-')[1])

            old_factor_path = r'/home/datamake94/data_nb8/min_factor_agg'
            stockid = torch.load(old_factor_path + '//' + self.date + '//' + 'stockid.pt').tolist()
            min_factor_tensor_A = np.memmap(os.path.join(old_factor_path, self.date, 'min_factor_data_A.Mmap'), dtype=np.float32, mode='r', shape=(
                238, len(stockid), 800))[:GlobalConfig.time_num, :, :]
            min_factor_tensor_B = np.memmap(os.path.join(old_factor_path, self.date, 'min_factor_data_B.Mmap'), dtype=np.float32, mode='r', shape=(
                238, len(stockid), 667))[:GlobalConfig.time_num, :, :]
            min_factor_tensor = np.concatenate([min_factor_tensor_A, min_factor_tensor_B], axis=2)[:, :, start_signal:end_signal]

            # 将数据索引到self.stock_list，缺失的股票填充为nan
            factor_data = np.full((GlobalConfig.time_num, len(self.stock_list), min_factor_tensor.shape[2]), np.nan, dtype=self._dtype)
            for idx, stock in enumerate(self.stock_list):
                for idx2, stock2 in enumerate(stockid):
                    if stock == stock2:
                        factor_data[:, idx, :] = min_factor_tensor[:, idx2, :]
                        break
            factor_data = torch.from_numpy(factor_data).transpose(0, 1).to(device=GlobalConfig.use_device)
        
        else:
            # 读取标准因子包
            file_path = rf"{GlobalConfig.factor_output_path}/{factor_pack_name}/{self.date}"
            with open(rf"{file_path}/factor_list.pkl", "rb") as f:
                factor_list = pickle.load(f)
            stockid = torch.load(rf'{file_path}/stockid.pt')  # 加载股票ID
            if type(stockid) != list:
                stockid = stockid.tolist()
            factor_data_list = []
            for code in self.stock_list:
                if code in stockid:
                    idx = stockid.index(code)  # 获取股票的索引
                    with h5py.File(rf"{file_path}/min_data.h5", 'r') as f:
                        factor_data = f['data'][:, idx, :].astype(self._dtype)
                else:
                    factor_data = np.full((GlobalConfig.time_num, len(factor_list)), np.nan, dtype=self._dtype)
                factor_data_list.append(torch.from_numpy(factor_data).unsqueeze(dim=0).to(device=GlobalConfig.use_device))
            
            factor_data = torch.cat(factor_data_list, dim=0)
        return factor_data

    def load_ret(self, label):
        """
        加载收益数据，根据给定的标签。
        """
        ret_dict = Ret_data(self.date)  # 创建收益数据对象
        ret_data = ret_dict.get_ret_data_all(label)  # 获取所有的收益数据
        ret_data = ret_data.reindex(columns=self.stock_list).T  # 根据股票列表重排数据
        return torch.from_numpy(ret_data.values).to(device=GlobalConfig.use_device)

    def load_style(self, label):
        """
        加载风格因子数据，进行标准化处理。
        """
        if len(GlobalStyleDict.style_dict) == 0:
            GlobalStyleDict.style_dict = load_style_factor()  # 加载风格因子字典
        style_factor = GlobalStyleDict.style_dict[label].loc[self.pre_date].reindex(self.stock_list).rename(label).to_frame()
        style_factor = (style_factor - style_factor.mean()) / style_factor.std()  # 标准化处理
        style_factor = torch.from_numpy(style_factor.values).float().repeat(1, GlobalConfig.time_num).to(device=GlobalConfig.use_device)
        return style_factor

    def daily_to_minute(self, df) -> torch.tensor:
        """
        将日频数据转换为分钟频数据。
        """
        df = df.reindex(index=self.stock_list)  # 根据股票列表重新索引
        tensor = torch.from_numpy(df.values).float().unsqueeze(dim=1).repeat(1, GlobalConfig.time_num, 1)  # 扩展维度以适应分钟级数据
        return tensor