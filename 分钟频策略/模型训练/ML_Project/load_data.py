#%%
import pandas as pd
import numpy as np
import os
import time
import pickle as pkl
import h5py
from torch.utils.data import sampler
from torch.utils.data import DataLoader
import torch
from torch import nn
import bottleneck as bn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Function
import random
import threading
import sys
# sys.path.append(r'/home/datamake94/高频策略/分钟级截面模型/ML_Project')
from prefetch_generator import BackgroundGenerator
from .training_config import calc_time_series, params, Dataset

#%%
#####################################################
#函数区
#####################################################
def Winsorize(dt,axis=0,n=3):
    mean = dt.mean(axis=axis,skipna=True)
    std = dt.std(axis=axis,skipna=True)
    dt_up = mean + n*std
    dt_down = mean - n*std
    return dt.clip(dt_down, dt_up, axis=abs(axis-1))   # 超出上下限的值，赋值为上下限

def Neutralize(df,axis):
    return (df - df.mean(axis=axis,skipna=True)) / df.std(axis=axis,skipna=True)

def Normalize(df,axis):
    return (df - df.min(axis=axis,skipna=True)) / (df.max(axis=axis,skipna=True) - df.min(axis=axis,skipna=True))

def get_datelist_all():
    with open(r"/home/datamake94/data_nb0/support_data/trade_days_dict.pkl",'rb') as f:
        trade_days_dict = pkl.load(f)
    trade_days_list = sorted([x.strftime('%Y%m%d') for x in trade_days_dict['trade_days']])
    return trade_days_list

def get_date_list(date1,date2):
    date_list_all = params.date_list_all
    date_list = date_list_all[date_list_all.index(date1):date_list_all.index(date2)]
    return date_list

def get_year_month(date_list):
    date_list = [x[:6] for x in date_list]
    output = list(set(date_list))
    output.sort()
    return output

def get_first_date(year_month,date_list):
    return [x for x in date_list if x[:6] >= year_month][0]

def get_pre_date(date, period=1):
    date_list_all=get_datelist_all()
    output=date_list_all[date_list_all.index(date)-period:date_list_all.index(date)]
    return output[0] if len(output)==1 else output

def get_train_date_list(date):
    date_list_all = get_datelist_all()
    #此处可以改
    period_date_list = date_list_all[date_list_all.index(date)-480-params.period_num-1:date_list_all.index(date)-params.period_num-1]
    period_date_list = [x for x in period_date_list if x>='20200430']
    ###
    period_date_list = [x for x in period_date_list if (x>='20240220') | (x<='20240126')]
    date_length = len(period_date_list)
    # random.shuffle(period_date_list)
    validation_date_list = period_date_list[-int(date_length*0.1):]
    train_date_list = period_date_list[:period_date_list.index(validation_date_list[0])-params.period_num-1]
    return period_date_list,train_date_list,validation_date_list

params.date_list_all=get_datelist_all()

def read_daily_support(date):
    daily_support_path = r'/home/datamake94/data_nb0/sec_daily_support'
    data = pd.read_csv(os.path.join(daily_support_path, date+'_daily_support.csv'),index_col=0)
    data.index = data.index.astype(int)
    data = data.rename({'vol':'vol1'},axis=1)
    return data

def df_to_tensor(df):
    return torch.from_numpy(df.to_numpy()).float()

def get_default_stock_list(date):
    stock_list=[]
    count=0
    for group in params.factor_path_dict.keys():
        while True:
            try:
                stock_list=set(stock_list)|set(torch.load(r'%s/%s/stockid.pt'%(params.factor_path_dict[group],date)).numpy())
                break
            except:
                print('%s: %s 遇到EOFError错误，等待5秒后重新读取...'%(group,date))
                count+=1
                if count>=12:
                    raise 
                time.sleep(5)
    return sorted(list(stock_list))

def get_daily_valid_stock(date,minute):
    stock_list = pkl.load(open(os.path.join(params.min_valid_path,date,str(minute)+'.pkl'),'rb'))
    return stock_list

def tensor_fill_mean(tensor):
    '''跟实盘对齐调整后的处理方式'''
    tensor[torch.isinf(tensor)]=torch.nan  #将inf转为空
    mask = torch.isnan(tensor)
    # 计算每列的中值
    column_mean = torch.nanmean(tensor.clamp(-3,3), dim=0, keepdim=True)
    # 填充中值到缺失值的位置
    filled_tensor = torch.where(mask, column_mean, tensor)
    return filled_tensor

def get_default_factor_list():
    factor_list=[]
    for group in params.factor_path_dict.keys():
        if group == 'OLD_FACTOR':
            # factor_names=pd.read_pickle(params.factor_path_dict[group]+'/20230103/factor_list.pkl')
            factor_names = list([str(i) for i in range(1467)])
        else:
            try:
                factor_names=pd.read_pickle(params.factor_path_dict[group]+'/20230103/factor_list.pkl')
            except:
                factor_names=torch.load(params.factor_path_dict[group]+'/20230103/factor_list.pt')
        factor_list.extend([group+'_'+x for x in factor_names])
    return factor_list

def load_factor_group(factor_group,date,sec,stock_list):
    count=0
    while True:
        # try:
        factor1_path = os.path.join(params.factor_path_dict[factor_group],date,'sec_data.h5')
        stockid = torch.load(os.path.join(params.factor_path_dict[factor_group],date,'stockid.pt')).tolist()
        factor_data = torch.from_numpy(pd.DataFrame(load_h5(factor1_path,0,params.sec_list_factor.index(sec)),index=stockid).reindex(index=stock_list).to_numpy()).float()
#             factor_data = torch.from_numpy(pd.DataFrame(torch.load(factor1_path).numpy(),index=stockid).reindex(index=stock_list).to_numpy()).float()
        break
        # except:
        #     print('%s: %s, %s 遇到EOFError错误，等待5秒后重新读取...'%(factor_group,date,sec))
        #     count+=1
        #     if count>=12:
        #         raise 
        #     time.sleep(5)
    return factor_data

def load_factor_group(factor_group,date,minute,stock_list):
    if factor_group == 'OLD_FACTOR':
        stockid = torch.load(params.factor_path_dict[factor_group] + '//' + date + '//' + 'stockid.pt').tolist()
        min_factor_tensor_A = np.memmap(os.path.join(params.factor_path_dict[factor_group], date, 'min_factor_data_A.Mmap'), dtype=np.float32, mode='r', shape=(238, len(stockid), 800))[params.sec_list_factor_old.index(int(minute)),:,:]
        min_factor_tensor_B = np.memmap(os.path.join(params.factor_path_dict[factor_group], date, 'min_factor_data_B.Mmap'), dtype=np.float32, mode='r', shape=(238, len(stockid), 667))[params.sec_list_factor_old.index(int(minute)),:,:]
        assert min_factor_tensor_A.shape[0] == min_factor_tensor_B.shape[0]
        min_factor_tensor = np.hstack([min_factor_tensor_A,min_factor_tensor_B])
        factor_data = torch.from_numpy(pd.DataFrame(min_factor_tensor,
                index=stockid).reindex(stock_list).values).float()
    # '''以h5py形式读取，需要注意分块的维度应适合读取'''
    # factor1_path = os.path.join(params.factor_path_dict[factor_group],date)
    # stock_list = torch.load(os.path.join(factor1_path, 'stockid.pt')).numpy().tolist()
    # factor_list = pd.read_pickle(os.path.join(factor1_path, 'factor_list.pkl'))
    # stock_ind = stock_list.index(stock)
    
    # factor_data = load_h5(rf'{factor1_path}/sec_data.h5',0,stock_ind)
    # factor_data=torch.from_numpy(pd.DataFrame(factor_data,index=params.sec_list_factor).reindex(sec_list).to_numpy()).float()
    return factor_data

def load_factor_static(date,stock_list):
    factor_data = torch.from_numpy(pd.read_feather(params.static_factor_path + '/%s.fea'%date).set_index('stockid').reindex(index=stock_list).values).float()
    return factor_data

def load_all_data(date,stock_list, minute , model_training=False, min_se=None, max_se=None, import_se=False, valid_ind=None):
    factor_data_list = []
    for factor_group in params.factor_path_dict.keys():  #各组因子
        factor_data_list.append(load_factor_group(factor_group,date,minute,stock_list))
        
    factor_data_list.append(load_factor_static(date,stock_list))
    factor_data = torch.hstack(factor_data_list)
    
    if model_training:
        if params.agg_method=='doublenorm':
            min_factor_tensor_list = []
            min_factor_tensor1 = (factor_data - min_se) / (max_se - min_se) if import_se else (factor_data - Dataset.min_se) / (Dataset.max_se - Dataset.min_se)
            min_factor_tensor1 = min_factor_tensor1[:,Dataset.valid_ind]
            min_factor_tensor1 = torch.clip(min_factor_tensor1, 1.5, -0.5)
            min_factor_tensor1[torch.isnan(min_factor_tensor1) | torch.isinf(min_factor_tensor1)] = 0.5
            min_factor_tensor_list.append(min_factor_tensor1)
            df = pd.DataFrame(factor_data[:,Dataset.valid_ind].numpy())
            df = Winsorize(df,axis=0)
            df = Neutralize(df,axis=0)
            # df.iloc[:,:] = drop_nan_all(df.values,axis=0,method = 'mean',data_type = 'finance')
            df[(pd.isnull(df)) | (np.isinf(df.values))] = 0
            min_factor_tensor2 = torch.from_numpy(df.values)
            min_factor_tensor_list.append(min_factor_tensor2)
            factor_data = torch.hstack(min_factor_tensor_list)
        else:
            factor_data = (factor_data - min_se) / (max_se - min_se) if import_se else (factor_data - Dataset.min_se) / (Dataset.max_se - Dataset.min_se)
            factor_data[factor_data>1.5] = 1.5
            factor_data[factor_data<-0.5] = -0.5

            if valid_ind is None:
                factor_data = factor_data[:,Dataset.valid_ind]
            else:
                factor_data = factor_data[:,valid_ind]
        
        # if sec<100000:  #采用新方案
        #     factor_data = (tensor_fill_mean(factor_data)-0.5).clamp(-1,1)
        #     factor_data[torch.isnan(factor_data)]=0
        # else:
        #     factor_data = tensor_fill_mean(factor_data).clamp(-1,2)
        #     factor_data[torch.isnan(factor_data)]=0.5

        # 预处理放到模型中进行    
            factor_data = tensor_fill_mean(factor_data)
            factor_data[torch.isnan(factor_data)]=0.5
    return factor_data  #返回的已经是经过筛选的，所以数量不一定等于所有因子的数目

def load_all_data_final(date,stock_list, minute, name, model_training=False, min_se=None, max_se=None, import_se=False, valid_ind=None, tmpfs_fast=False):
    '''训练时使用，考虑是否直接存内存映射中读取已有数据'''
    fac_num=params.factor_num#len(get_default_factor_list(if_static=True))   
    if tmpfs_fast:
        tmpfs_path = os.path.join(params.tmpfs_path,name)
        all_file = os.listdir(tmpfs_path)
        if date + '_' + str(minute) + '.Mmap' in all_file:
            factor_tensor = np.memmap(os.path.join(tmpfs_path, date + '_' + str(minute) + '.Mmap'), dtype=np.float32, mode='r', shape=(len(stock_list), fac_num))
            factor_tensor = torch.from_numpy(factor_tensor).float()
        else:
            factor_tensor = load_all_data(date, stock_list, minute, model_training=model_training,\
                min_se=min_se, max_se=max_se, import_se=import_se, valid_ind=valid_ind)
            factor_numpy = factor_tensor.numpy()
            mmap = np.memmap(os.path.join(tmpfs_path, date + '_' + str(minute) + '.Mmap'), dtype=np.float32, mode='w+', shape=factor_numpy.shape)
            mmap[:] = factor_numpy[:]
            mmap._mmap.close()
    else:
        factor_tensor = load_all_data(date,stock_list, minute, model_training=True, valid_ind=valid_ind)
    
    assert fac_num==factor_tensor.shape[1],rf'fac_num is {fac_num}, but actual_num is {factor_tensor.shape[1]}'
    return factor_tensor

def load_ret_data(date,stock_list,minute,period,agg=False,neu=False,tmpfs_fast=False,qcut=0):
    '''加载收益率数据'''
    if agg:
        ret_list = []
        for per in ['1d','3d','30']:
            ret_list.append(load_ret_data(date,stock_list,minute,per,agg=False,neu=True,tmpfs_fast=tmpfs_fast,qcut=qcut))
        ret_tensor = ret_list[0] * 0.5 + ret_list[1] * 0.25 + ret_list[2] * 0.25
    else:
        ret_path = params.future_ret_path_dict[period] + '//' + date
        ret_tensor = torch.load(ret_path + '//' + str(minute) + '.pt')
        stockid = torch.load(ret_path + '//' + 'stockid.pt').tolist()
        ret_tensor = torch.from_numpy(pd.DataFrame(ret_tensor,index=stockid).reindex(
            index=stock_list).values).float()
    
        ret_tensor[torch.isnan(ret_tensor)]=0
        ret_tensor = ret_tensor*100
        if qcut != 0:
            qmin = ret_tensor.quantile(qcut)
            qmax = ret_tensor.quantile(1-qcut)
            ret_tensor = ret_tensor.clip(qmin,qmax)
        if neu:
            ret_tensor = (ret_tensor-ret_tensor.mean(dim=0,keepdim=True)) / ret_tensor.std(dim=0,keepdim=True)
    return ret_tensor


# %%
############################################
#数据读取区
############################################
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl 

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class UMPDataset(torch.utils.data.Dataset):
    def __init__(self, args, target_date ,date_list, name, module='valid', get_noise=False, if_sample=False) -> None:
        self.args=args
        self.target_date=target_date
        self.date_list = date_list
        self.date_minute_list = [(date,minute) for date in date_list for minute in params.sec_list_period_dict[args.interval]]  #每个单元为一个时间戳
        self.module = module
        self.name = name
        
        if if_sample:
            random.seed(args.seed)
            self.date_minute_list=random.sample(self.date_minute_list,int(args.sample*len(self.date_minute_list)))  #验证集上抽样验证
        self.get_noise = get_noise

    def __getitem__(self, index):
        date,minute = self.date_minute_list[index]
        factor_stock_list=sorted(get_default_stock_list(date))
        valid_stock_list=get_daily_valid_stock(date,minute)
        stock_list = sorted(list(set(factor_stock_list) & set(valid_stock_list)))

        factor_data = load_all_data_final(date,stock_list, minute, self.name, model_training=True, tmpfs_fast=params.tmpfs_fast)
        if self.module == 'valid':
            ret_data = load_ret_data(date,stock_list,minute,self.args.period,agg=False,neu=False)
        else:
            ret_data = load_ret_data(date,stock_list,minute,self.args.period,agg=self.args.agg,neu=True)
        if self.get_noise:
            noise = torch.normal(0,0.02,factor_data.shape)
            factor_data = factor_data + noise

        return (factor_data.float(), ret_data.float())

    def __len__(self):
        return len(self.date_minute_list)

class UMPDataModule(pl.LightningDataModule):
    def __init__(self, args, date, train_date_list, valid_date_list, name):
        super().__init__()
        self.args = args
        self.tr = UMPDataset(args, date, train_date_list, name, module='train', get_noise=args.add_noise, if_sample=True)
        self.val = UMPDataset(args, date, valid_date_list, name, module='valid', get_noise=False, if_sample=False)
        self.test = self.val

    def train_dataloader(self):
        return DataLoader(self.tr,batch_size=self.args.batch_size,collate_fn=lambda x:x,
                          num_workers=self.args.workers, shuffle=True,
                          persistent_workers=self.args.persistent_workers,
                          drop_last=False,
                          pin_memory=True)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset,batch_size=1,collate_fn=lambda x:x, shuffle=False,
                          num_workers=self.args.workers,persistent_workers=self.args.persistent_workers,drop_last=False,
                          pin_memory=False)

    def val_dataloader(self):
        return self._val_dataloader(self.val)
    
    def test_dataloader(self):
        return self._val_dataloader(self.test)
# %%
def update_training_minmax(date_list):
    min_list,max_list = [],[]
    for date in date_list:
        with open(params._min_max_path + '//' + date + '.pkl','rb') as f:
            minmax_dict = pkl.load(f)
        min_list.append(minmax_dict['min'])
        max_list.append(minmax_dict['max'])
    min_tensor = torch.vstack(min_list)
    min_tensor[torch.isnan(min_tensor)] = 99999999999999999
    min_tensor = min_tensor.min(dim=0).values
    max_tensor = torch.vstack(max_list)
    max_tensor[torch.isnan(max_tensor)] = -99999999999999999
    max_tensor = max_tensor.max(dim=0).values
    
    min_static_list,max_static_list = [],[]
    for date in date_list:
        with open(params.factor_path_dict['OLD_FACTOR'] + '//static_factor_path2//static_min_max_dict//' + date + '.pkl','rb') as f:
            minmax_dict = pkl.load(f)
        min_static_list.append(torch.from_numpy(minmax_dict['min'].values))
        max_static_list.append(torch.from_numpy(minmax_dict['max'].values))
    min_static_tensor = torch.vstack(min_static_list)
    min_static_tensor[torch.isnan(min_static_tensor)] = 99999999999999999
    min_static_tensor = min_static_tensor.min(dim=0).values
    max_static_tensor = torch.vstack(max_static_list)
    max_static_tensor[torch.isnan(max_static_tensor)] = -99999999999999999
    max_static_tensor = max_static_tensor.max(dim=0).values

    # Dataset.min_tensor = torch.hstack([min_tensor,min_static_tensor])
    # Dataset.max_tensor = torch.hstack([max_tensor,max_static_tensor])

    # if params.if_select_factor:
    #     Dataset.min_tensor = torch.hstack([min_tensor[params.select_factor], min_static_tensor])
    #     Dataset.max_tensor = torch.hstack([max_tensor[params.select_factor], max_static_tensor])
    # else:
    Dataset.min_se = torch.hstack([min_tensor, min_static_tensor])
    Dataset.max_se = torch.hstack([max_tensor, max_static_tensor])
    return