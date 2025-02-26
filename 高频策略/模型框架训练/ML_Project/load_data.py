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

params.date_list_all=get_datelist_all()

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

def tensor_fill_mean(tensor):
    '''跟实盘对齐调整后的处理方式'''
    tensor[torch.isinf(tensor)]=torch.nan  #将inf转为空
    mask = torch.isnan(tensor)
    # 计算每列的中值
    column_mean = torch.nanmean(tensor.clamp(-3,3), dim=0, keepdim=True)
    # 填充中值到缺失值的位置
    filled_tensor = torch.where(mask, column_mean, tensor)
    return filled_tensor

def get_default_factor_list(if_static=True):
    factor_list=[]
    for group in params.factor_path_dict.keys():
        try:
            factor_names=pd.read_pickle(params.factor_path_dict[group]+'/factor_list.pkl')
        except:
            factor_names=torch.load(params.factor_path_dict[group]+'/factor_list.pt')
        factor_list.extend([group+'_'+x for x in factor_names])
    return factor_list+['static_'+str(x) for x in range(1,72)] if if_static else factor_list

def load_h5(path,axis=-1,ind=None):
    with h5py.File(path, "r") as f:
        h5_data=f['data']
        dims=len(h5_data.shape)
        if dims==3:
            if axis==-1:  #T*N*K
                return h5_data[:,:,:]     
            if axis==0: 
                return h5_data[ind,:,:]
            if axis==1:
                return h5_data[:,ind,:]
            if axis==2:
                return h5_data[:,:,ind]
        if dims==2:  
            if axis==-1:  #T*N
                return h5_data[:,:]     
            if axis==0:
                return h5_data[ind,:]
            if axis==1:
                return h5_data[:,ind]    

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

def load_factor_static(date,stock_list):
    factor_data = torch.from_numpy(pd.read_feather(params.static_factor_path + '/%s.fea'%date).set_index('stockid').reindex(index=stock_list).values).float()
    return factor_data

def load_factor_ml(date,sec,stock_list):
    path=params.ml_factor_path + '/%s.h5'%date
    with h5py.File(path, "r") as f:
        second_list=list(f['sec_list'][:])
        stockid=f['stockid'][:]
        factor_all=pd.Series(f['data'][second_list.index(sec),:],index=stockid)
    factor_data=torch.from_numpy(factor_all.reindex(index=stock_list).fillna(0).values).float()
    return factor_data.unsqueeze(1)

def load_stock_weight(date,stock_list):
    # path=r'/home/intern1/data0/weight_matrix.h5'
    # with h5py.File(path, "r") as f:
    #     date_list=list(f['date_list'][:])
    #     stockid=list(f['stockid'][:])
    #     factor_all=pd.Series(f['data'][date_list.index(date.encode('utf-8')),:],index=stockid)
    # factor_data=torch.from_numpy(factor_all.reindex(index=stock_list).fillna(0).values).float()
    weight = weight_table.vol_weight.loc[date].reindex(index=stock_list)
    weight = weight.rank(pct=True).fillna(0.5)
    weight = 1-torch.from_numpy(weight.values).float()  #对低波股票更多权重
    return weight.unsqueeze(1)

def load_all_data(date,stock_list, sec, model_training=False, min_se=None, max_se=None, import_se=False, valid_ind=None):
    factor_data_list = []
    for factor_group in params.factor_path_dict.keys():  #各组因子
        factor_data_list.append(load_factor_group(factor_group,date,sec,stock_list))
        
    factor_data_list.append(load_factor_static(date,stock_list))
    factor_data = torch.hstack(factor_data_list)
    
    if model_training:
        factor_data = (factor_data - min_se) / (max_se - min_se) if import_se else (factor_data - Dataset.min_se) / (Dataset.max_se - Dataset.min_se)
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

def load_all_data_final(date,stock_list, sec, name, valid_ind=None, tmpfs_fast=False):
    '''训练时使用，考虑是否直接存内存映射中读取已有数据'''
    fac_num=params.factor_num#len(get_default_factor_list(if_static=True))   
    if tmpfs_fast:
        tmpfs_path = os.path.join(params.tmpfs_path,name)
        all_file = os.listdir(tmpfs_path)
        if date + '_' + str(sec) + '.Mmap' in all_file:
            factor_tensor = np.memmap(os.path.join(tmpfs_path, date + '_' + str(sec) + '.Mmap'), dtype=np.float32, mode='r', shape=(len(stock_list), fac_num))
            factor_tensor = torch.from_numpy(factor_tensor).float()
        else:
            factor_tensor = load_all_data(date,stock_list, sec, model_training=True, valid_ind=valid_ind)
            factor_numpy = factor_tensor.numpy()
            mmap = np.memmap(os.path.join(tmpfs_path, date + '_' + str(sec) + '.Mmap'), dtype=np.float32, mode='w+', shape=factor_numpy.shape)
            mmap[:] = factor_numpy[:]
            mmap._mmap.close()
    else:
        factor_tensor = load_all_data(date,stock_list, sec, model_training=True, valid_ind=valid_ind)
    
    assert fac_num==factor_tensor.shape[1],rf'fac_num is {fac_num}, but actual_num is {factor_data.shape[1]}'
    return factor_tensor

def select_factor_index(date_list, threshold, subthreshold, method, future_ret, period):
    '''去除相关性'''
    ic_list=[]
    ls_list=[]
    corr=0
    factor_list=get_default_factor_list(if_static=False)
    
    ic_ls_factor_list=pd.read_pickle(params.ic_ls_path + '/factor_list.pkl')
    sec_factor_num=len(factor_list)
    ind=params.ic_ls_future_ret_list.index(future_ret)   #ic_ls_future_ret_list 此处要严格按照数据储存的顺序来
    
    for date in date_list:
        ic_list.append(torch.load(params.ic_ls_path + '//%s_ic_%s.pt'%(date,period)).numpy())
        ls_list.append(torch.load(params.ic_ls_path + '//%s_ls_%s.pt'%(date,period)).numpy())
        corr+=torch.load(params.ic_ls_path + '//%s_corr_%s.pt'%(date,period)).numpy()

    ic_all=pd.DataFrame(bn.nanmean(np.array(ic_list),axis=0),columns=ic_ls_factor_list)[factor_list].values
    ls_all=pd.DataFrame(bn.nanmean(np.array(ls_list),axis=0),columns=ic_ls_factor_list)[factor_list].values
    corr_all=pd.DataFrame(corr/corr[0,0],index=ic_ls_factor_list,columns=ic_ls_factor_list).loc[factor_list,factor_list].values
    
    ic=ic_all[ind]
    ls=ls_all[ind]
    
    valid_ind=Dataset.valid_ind[:sec_factor_num].numpy()&(~(np.isnan(ic)|np.isnan(ls)))
    select_index=np.arange(sec_factor_num)[valid_ind]  #记录初步筛除后剩下的矩阵在所有因子中的索引
    
    ic=ic[valid_ind]
    ls=ls[valid_ind]
    corr=np.abs(corr_all[valid_ind][:,valid_ind])
    
    # 将相关性矩阵的元素大于等于阈值的设为1，小于阈值的设为0
    adjacency_matrix = (corr >= threshold).astype(int)

    # 自定义联通子图搜索函数
    def find_connected_components(adj_matrix, corr, criterion, connectivity_threshold=subthreshold):
        n = adj_matrix.shape[0]
        visited = [False] * n
        connected_components = []

        # 深度优先搜索
        def dfs(v, component):
            visited[v] = True
            for i in range(n):
                if adj_matrix[v, i] == 1 and not visited[i]:
                    # 检查节点i与当前联通子图中所有节点的相关性是否都大于connectivity_threshold
                    if all(corr[i, member] > connectivity_threshold for member in component):
                        component.append(i)
                        # visited[v] = True
                        dfs(i, component)

        for v in range(n):
            if not visited[v]:
                # visited[v] = True
                component = [v]
                dfs(v, component)
                connected_components.append(component)

        # 筛选每个联通子图中ic前30%的因子
        filtered_components = []
        for component in connected_components:
            # 按照ic值排序，并取前30%，至少保留一个因子
            cri_values = [(criterion[member], member) for member in component]
            cri_values.sort(reverse=True)  # 假设ic值越高越好
            if method >= 1:
                cutoff = method
            else:
                cutoff = max(1, int(np.round(len(cri_values) * method)))

            filtered_components.append([member for cri, member in cri_values[:cutoff]])

        return filtered_components

    connected_components = find_connected_components(adjacency_matrix, corr,ls)
    #print("联通子图数量：",len(connected_components))
    
    selected_factor_all = []
    for component in connected_components:
        selected_factor_all += component
    all_selected_factor_index = selected_factor_all
    #print("筛选后因子数量",len(all_selected_factor_index))
    final_index=select_index[sorted(all_selected_factor_index)]
    
    return ic_all,ls_all,corr_all,final_index

def update_training_minmax(date_list,target_date,future_ret=None,period=None, method='minmax', threshold=0.2, if_decorr=False):
    #获取训练周期内所有的因子最大值，最小值
    min_list,max_list = [],[]
    nan_ratio_list=[]
    for date in date_list:
        min_tensor,max_tensor = torch.load(params._min_max_path + '//%s_min.pt'%date),torch.load(params._min_max_path + '//%s_max.pt'%date)
        min_tensor[torch.isinf(min_tensor)] = torch.nan
        max_tensor[torch.isinf(max_tensor)] = torch.nan
        min_list.append(min_tensor)
        max_list.append(max_tensor)
        nan_ratio_list.append(torch.load(params._min_max_path + '//%s_nan_ratio.pt'%date))
        
    min_tensor,max_tensor,ratio_tensor = torch.vstack(min_list),torch.vstack(max_list),torch.vstack(nan_ratio_list)
    min_max_factor_list=pd.read_pickle(params._min_max_path + '/factor_list.pkl')
    
    if method=='minmax':
        min_value,max_value = np.nanmin(min_tensor.numpy(),axis=0),np.nanmax(max_tensor.numpy(),axis=0)
    elif method=='mean':
        min_value,max_value = np.nanmean(min_tensor.numpy(),axis=0),np.nanmean(max_tensor.numpy(),axis=0)
        
    all_list=get_default_factor_list(if_static=True)
    Dataset.min_se = df_to_tensor(pd.Series(min_value,index=min_max_factor_list)[all_list])  #直接按照位置索引取出，严格防止顺序改变
    Dataset.max_se = df_to_tensor(pd.Series(max_value,index=min_max_factor_list)[all_list])
    Dataset.nan_ratio = df_to_tensor(pd.Series(ratio_tensor.mean(dim=0).numpy(),index=min_max_factor_list)[all_list])

    Dataset.valid_ind=(Dataset.nan_ratio<threshold)&(Dataset.min_se!=Dataset.max_se)

    # 部分因子不可用
    error_list=[]
    invalid_factor_list = \
            ['ysw_orderbook1_PVWAP_IMB_5_mean', 'ysw_orderbook1_vn_5_corr', 'ysw_orderbook1_pvol_ratio_5_mean', \
            'ysw_orderbook1_PVWAP_IMB_15_mean', 'ysw_orderbook1_vn_15_corr', 'ysw_orderbook1_pvol_ratio_15_mean', \
            'ysw_orderbook1_PVWAP_IMB_30_mean', 'ysw_orderbook1_vn_30_corr', 'ysw_orderbook1_pvol_ratio_30_mean', \
            'ysw_orderbook1_PVWAP_IMB_60_mean', 'ysw_orderbook1_vn_60_corr', 'ysw_orderbook1_pvol_ratio_60_mean', \
            'yy_orderbook3_sell_price_intercept', 'yy_orderbook3_buy_price_intercept', 'yy_orderbook3_all_price_intercept']
    for i in range(len(all_list)):
        if all_list[i] in invalid_factor_list:
        # if 'PVWAP_IMB' in all_list[i] or 'vn' in all_list[i] or 'pvol_ratio' in all_list[i] :  
            error_list.append(i)
    Dataset.valid_ind[error_list]=False
    
    factor_list_=get_default_factor_list(if_static=False)
    print('原始因子数量：',len(factor_list_))
    
    if if_decorr: #去除相关性
        ic_all,ls_all,corr_all,final_index=select_factor_index(date_list, threshold = 0.75, subthreshold = 0.5, method = 0.25,future_ret=future_ret,period=period)  #会传入Dataset.valid_ind以保证有效剔除,前面剔除的也会被剔除
        Dataset.valid_ind=torch.tensor([False if x<len(factor_list_) and x not in final_index else True for x in range(len(all_list))]) & Dataset.valid_ind
    
    params.factor_num=Dataset.valid_ind.sum().item()
    return

############################################
#数据读取区
############################################
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl 

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class UMPDataset(torch.utils.data.Dataset):
    def __init__(self, args, target_date ,date_list, module='valid', get_noise=False, if_sample=False) -> None:
        self.args=args
        self.target_date=target_date
        self.date_list = date_list
        self.date_sec_list = [(date,sec) for date in date_list for sec in params.sec_list_period_dict[args.future_ret][args.period]]  #每个单元为一个时间戳
        self.module = module
        
        if if_sample:
            random.seed(args.seed)
            self.date_sec_list=random.sample(self.date_sec_list,params.random_num*len(date_list))  #验证集上抽样验证
        self.get_noise = get_noise
    
    def __getitem__(self, index):
        date,sec = self.date_sec_list[index]
        factor_stock_list=sorted(get_default_stock_list(date))  #所有因子固定的个股
        valid_stock_list=pd.read_pickle(params.sec_valid_path+'/%s/%s.pkl'%(date,sec))  #剔除涨跌停个股了
        #graph_stock_list=Dataset.industry_matrix.columns   #pd.read_pickle(rf"{params.industry_graph_path}/Stockid/{date}.pkl")  #昨日有行业记录的个股
        stock_list=sorted(list(set(factor_stock_list)&set(valid_stock_list)))  
        
        stock_weight=None#load_stock_weight(date,stock_list)
        factor_data = load_all_data_final(date, stock_list, sec, params.name, tmpfs_fast=self.args.tmpfs_fast)
        ret_stock_list=torch.load(params.future_ret_path_dict[self.args.future_ret]+'/%s/stockid.pt'%date)
        ret_5=pd.Series(torch.load(params.future_ret_path_dict[self.args.future_ret]+'/%s/%s.pt'%(date,sec)).view(-1).numpy(),index=ret_stock_list).reindex(stock_list)
        ret_5=ret_5.fillna(ret_5.mean())
        ret_5_clip=ret_5.clip(ret_5.quantile(0.01),ret_5.quantile(0.99))
        
        pre_date = get_pre_date(date)
        industry_slice = Dataset.industry_table.loc[pre_date]
        industry_dummy = torch.from_numpy(pd.get_dummies(industry_slice).reindex(stock_list).fillna(False).values).float()
 
        if self.args.processLabel:
            ret_5_clip=ret_5_clip.apply(lambda x: x if x!=0 else np.random.randn()/1e4)  #处理label
            
        if self.get_noise:
            noise = torch.normal(0,0.02,factor_data.shape)
            factor_data = factor_data + noise

        ###
        #将收益率按照大小分10组，头尾两组全抽样，中间8组随机抽样
        # if self.module == 'train':
        #     ret_5 = df_to_tensor(ret_5).unsqueeze(dim=1)*100
        #     ret_5_clip = df_to_tensor(ret_5_clip).unsqueeze(dim=1)*100
        #     ret_argsort = ret_5.argsort(dim=0,descending=False).squeeze().tolist()
        #     cut_num = len(stock_list) // 10
        #     argsort_idx = ret_argsort[:cut_num] + ret_argsort[-cut_num:] + random.sample(ret_argsort[cut_num:-cut_num], int(8*cut_num*0.1))
        #     random.shuffle(argsort_idx)
        #     stock_list = [stock_list[i] for i in argsort_idx]
        #     factor_data = factor_data[argsort_idx]
        #     industry_dummy = industry_dummy[argsort_idx]
        #     ret_5 = ret_5[argsort_idx]
        #     ret_5_clip = ret_5_clip[argsort_idx]

        if True:
            ret_5 = df_to_tensor(ret_5).unsqueeze(dim=1)*100
            ret_5_clip = df_to_tensor(ret_5_clip).unsqueeze(dim=1)*100
        
        return ([factor_data,industry_dummy], ret_5, ret_5_clip,stock_weight)

    def __len__(self):
        return len(self.date_sec_list)
    
class UMPDataModule(pl.LightningDataModule):
    def __init__(self, args, date, train_date_list, valid_date_list):
        super().__init__()
        self.args = args
        self.tr = UMPDataset(args, date, train_date_list, module='train', get_noise=args.add_noise, if_sample=True)
        self.val = UMPDataset(args, date, valid_date_list, module='valid', get_noise=False, if_sample=True)
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