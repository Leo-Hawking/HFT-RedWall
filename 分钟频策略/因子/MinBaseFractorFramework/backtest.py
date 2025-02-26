import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from .global_config import GlobalConfig
from . import torch_operators as op
from .fac_exp_exec import *
from .util_function import *
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from matplotlib.backends.backend_pdf import PdfPages  # 用于将图像输出到PDF文件
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

def eval_fac_sec_hs(Backtest, date_list, factor_pack_name, factor_prefix_dict={}, factor_value={}, load_factor=True, if_score=False):
    '''
    对每个日期执行回测，返回IC、RankIC、HedgeRet的四维Tensor（每个时间点、标签、因子数、股票数）
    
    Parameters:
        Backtest: 回测对象，包含标签列表等
        date_list: 日期列表，回测的时间范围
        factor_pack_name: 因子包的名称
        factor_prefix_dict: 因子表达式的前缀字典（可选）
        factor_value: 提前计算好的因子数据（可选）
        load_factor: 是否加载因子数据
        if_score: 是否计算打分因子数据

    Returns:
        ic_result: 每个日期、标签、因子的IC值
        ric_result: 每个日期、标签、因子的RankIC值
        hedgeret_result: 每个日期、标签、因子的对冲收益
    '''
    output_dict = {'IC':[], 'RankIC':[], 'HedgeRet':[], 'p_adf':[]}  # 存储结果的字典
    factor_num = len(Backtest.factor_name_list)  # 获取因子数量

    # 横截面回测，逐个时间点进行处理
    ic_result_list, ric_result_list, hedgeret_result_list = [], [], []

    for date in date_list:
        ### 读取数据
        if len(factor_value) != 0 and date in factor_value.keys():
            fac_data, stock_list, ret_list, _ = factor_value[date]  # 如果已有因子数据，直接读取
        else:
            ret_list = []  # 初始化收益列表
            sec_daily_support = Factor_data(date).read_daily_support()  # 读取日数据
            stock_list = sec_daily_support.index.tolist()  # 获取股票列表

            factor_generator = Factor_data(date, stock_list=stock_list, _dtype='float32')  # 创建因子生成器
            if if_score:
                # 读取已保存的因子数据
                score_path = r'/home/datamake118/database_00/min_score_period30'
                fac_data = pd.read_feather(os.path.join(score_path, factor_pack_name, date + '.fea')).set_index('index')
                fac_data.loc[1005:] = np.nan  # 对数据进行预处理
                fac_data.index = [int(x * 100000) for x in fac_data.index]
                fac_data.columns = fac_data.columns.astype(int)
                fac_data = fac_data.reindex(index=GlobalConfig._sec_list, columns=stock_list)
                fac_data = torch.from_numpy(fac_data.values).float().T.unsqueeze(dim=2)  # 转换为tensor
            else:
                if not load_factor:
                    fac_data = factor_generator.exec_fac_exp_daily(factor_prefix_dict)  # 执行因子表达式
                else:
                    fac_data = factor_generator.load_factor_group(factor_pack_name)  # 加载因子组

            for label in Backtest.label_list:
                ret_list.append(factor_generator.load_ret(label).unsqueeze(dim=2))  # 加载回报数据
        ret_data = torch.cat(ret_list, dim=2)  # 合并所有标签的回报数据

        # 数据预处理
        fac_data = fac_data.transpose(0, 1)  # 转置因子数据
        ret_data = ret_data.transpose(0, 1)  # 转置回报数据
        fac_data = tensor_fill_mean(fac_data, dim=1)  # 填充因子数据的缺失值
        ret_data[torch.isnan(ret_data)] = 0  # 将回报数据中的NaN值设为0

        # 定义计算IC、RankIC和HedgeRet的函数
        def _calc_ic(_fac_data, label):
            return cs_corr(_fac_data, ret_data[:, :, Backtest.label_list.index(label)])  # 计算IC值

        def _calc_rankic(_fac_data, label):
            return cs_corr(cs_rank(_fac_data), cs_rank(ret_data[:, :, Backtest.label_list.index(label)]))  # 计算RankIC值

        def _calc_hedgeret(_fac_data, label):
            group_mean = cs_group_mean(_fac_data, ret_data[:, :, Backtest.label_list.index(label)], None, Backtest.num_groups)  # 计算对冲收益
            return group_mean

        # 计算每个标签的IC、RankIC和HedgeRet
        ic_list, ric_list, hedgeret_list = [], [], []
        for label in Backtest.label_list:
            ic_list.append(torch.stack([_calc_ic(fac_data[:, :, i], label) for i in range(factor_num)]))  # 计算所有因子的IC
            ric_list.append(torch.stack([_calc_rankic(fac_data[:, :, i], label) for i in range(factor_num)]))  # 计算所有因子的RankIC
            hedgeret_list.append(torch.stack([_calc_hedgeret(fac_data[:, :, i], label) for i in range(factor_num)]))  # 计算所有因子的对冲收益

        # 将每个标签的结果堆叠起来
        ic_result = torch.stack(ic_list)
        ric_result = torch.stack(ric_list)
        hedgeret_result = torch.stack(hedgeret_list)

        # 将每个日期的结果添加到列表中
        ic_result_list.append(ic_result.unsqueeze(dim=0))
        ric_result_list.append(ric_result.unsqueeze(dim=0))
        hedgeret_result_list.append(hedgeret_result.unsqueeze(dim=0))

    # 汇总所有日期的IC、RankIC、HedgeRet结果
    ic_result = torch.vstack(ic_result_list).nanmean(dim=0, keepdim=True)  # 对所有日期的IC求均值
    ric_result = torch.vstack(ric_result_list).nanmean(dim=0, keepdim=True)  # 对所有日期的RankIC求均值
    hedgeret_result = torch.vstack(hedgeret_result_list).nanmean(dim=0, keepdim=True)  # 对所有日期的HedgeRet求均值

    return ic_result, ric_result, hedgeret_result  # 返回计算结果

def eval_style_factor(Backtest, date_list, factor_pack_name, factor_prefix_dict={}, factor_value={}, load_factor=True, if_score=False):
    '''
    对每天数据进行回测，计算并返回IC、RankIC、HedgeRet的四维Tensor（分别按时间、标签、因子、股票维度组织）
    
    Parameters:
        Backtest: 回测对象，包含标签列表、因子数目等信息
        date_list: 日期列表，指定回测的时间范围
        factor_pack_name: 因子包名称，指定要使用的因子包
        factor_prefix_dict: 因子表达式的前缀字典（可选）
        factor_value: 提前计算好的因子数据（可选），如果已计算，则可以直接传入
        load_factor: 是否加载因子数据
        if_score: 是否使用打分因子数据

    Returns:
        hedgeret_result: 计算出来的HedgeRet（对冲收益）结果，四维Tensor
    '''
    output_dict = {'IC':[], 'RankIC':[], 'HedgeRet':[], 'p_adf':[]}  # 初始化一个字典用于存储计算结果
    factor_num = len(Backtest.factor_name_list)  # 获取因子数量
    style_name_list = ['size', 'mid_size', 'liquidity', 'volatility', 'momentum']  # 设定风格因子名称

    # 横截面回测：逐个日期进行回测，计算IC、RankIC和HedgeRet
    ic_result_list, ric_result_list, hedgeret_result_list = [], [], []  # 存储每个日期的结果

    for date in date_list:
        ### 读取数据
        if len(factor_value) != 0 and date in factor_value.keys():
            fac_data, stock_list, _, style_list = factor_value[date]  # 如果有预先计算好的因子数据，直接使用
        else:
            style_list = []  # 初始化风格因子列表
            sec_daily_support = Factor_data(date).read_daily_support()  # 读取当天的日数据
            stock_list = sec_daily_support.index.tolist()  # 获取当天的股票列表

            factor_generator = Factor_data(date, stock_list=stock_list, _dtype='float32')  # 创建因子生成器
            if if_score:
                # 如果是打分模式，加载打分因子数据
                score_path = r'/home/datamake118/database_00/min_score_period30'
                fac_data = pd.read_feather(os.path.join(score_path, factor_pack_name, date + '.fea')).set_index('index')
                fac_data.index = [int(x * 100000) for x in fac_data.index]  # 转换索引
                fac_data.columns = fac_data.columns.astype(int)  # 确保列是整数类型
                fac_data = fac_data.reindex(index=GlobalConfig._sec_list, columns=stock_list)  # 重排索引
                fac_data = torch.from_numpy(fac_data.values).float().T.unsqueeze(dim=2)  # 转换为Tensor并调整形状
            else:
                # 如果不是打分模式，根据需求加载因子数据
                if not load_factor:
                    fac_data = factor_generator.exec_fac_exp_daily(factor_prefix_dict)  # 执行因子表达式
                else:
                    fac_data = factor_generator.load_factor_group(factor_pack_name)  # 加载因子组数据

            # 加载风格因子数据（如size、liquidity等）
            for label in style_name_list:
                style_list.append(factor_generator.load_style(label).unsqueeze(dim=2))  # 为每个风格因子加载数据
        style_data = torch.cat(style_list, dim=2)  # 将所有风格因子数据沿着第三维度拼接

        # 数据预处理：将因子数据和风格因子数据进行转置，并填补缺失值
        fac_data = fac_data.transpose(0, 1)  # 转置因子数据
        style_data = style_data.transpose(0, 1)  # 转置风格因子数据
        fac_data = tensor_fill_mean(fac_data, dim=1)  # 填充因子数据中的缺失值
        style_data[torch.isnan(style_data)] = 0  # 将风格因子数据中的NaN值填充为0

        # 定义计算HedgeRet的函数
        def _calc_hedgeret(_fac_data, label):
            # 计算按风格因子对冲收益，使用分组计算
            group_mean = cs_group_mean(_fac_data, style_data[:, :, style_name_list.index(label)], None, Backtest.num_groups)  # 对头尾进行分组计算
            return group_mean  # 返回对冲收益

        hedgeret_list = []  # 存储每个风格因子的对冲收益
        for label in style_name_list:
            # 计算每个风格因子的对冲收益，并堆叠结果
            hedgeret_list.append(torch.stack([_calc_hedgeret(fac_data[:, :, i], label) for i in range(factor_num)]))
        hedgeret_result = torch.stack(hedgeret_list)  # 堆叠所有风格因子的结果
        hedgeret_result_list.append(hedgeret_result.unsqueeze(dim=0))  # 将每个日期的结果加到结果列表中
    
    # 汇总所有日期的HedgeRet结果
    hedgeret_result = torch.vstack(hedgeret_result_list).nanmean(dim=0, keepdim=True)  # 对所有日期的结果取均值
    return hedgeret_result  # 返回最终的HedgeRet结果
        
class Backtest_Framework:
    def __init__(self, num_groups, period, start_date, end_date, label_list=[15,60], preload_factor=False):
        '''
        初始化回测框架

        参数：
            num_groups: 用于分组的数量（如分为多少个组）
            period: 回测的时间周期（例如：按月、季度等进行回测）
            start_date: 回测的开始日期
            end_date: 回测的结束日期
            label_list: 回测中使用的标签列表（默认使用[15, 60]表示不同的回测标签）
            preload_factor: 是否预加载因子数据
        '''
        self.num_groups = num_groups  # 设定分组的数量
        self.period = period  # 设定回测周期
        self.start_date = start_date  # 设定回测开始日期
        self.end_date = end_date  # 设定回测结束日期
        self.date_list = get_trade_days(start_date, end_date)  # 获取交易日期列表
        self.date_list_period = [self.date_list[i] for i in range(0, len(self.date_list), self.period)]  # 根据周期拆分日期列表
        self.label_list = label_list  # 设定回测标签（例如：15日、60日）
        
        # 初始化回测结果的临时保存变量
        self.factor_name_list = None  # 用于存储因子名称列表
        self.ic_result = None  # 存储IC值结果
        self.hedgeret_result = None  # 存储对冲收益（hedge return）结果

        self.preload_factor = preload_factor  # 是否预加载因子数据
        self.fg_dict = {}  # 存储因子数据的字典
        if self.preload_factor:
            # 如果预加载因子数据，则为每个日期加载因子数据
            for date in self.date_list:
                self.fg_dict[date] = Factor_data(date, preload_factor=True)

    def eval_fac_sec_hs_all(self, factor_pack_name, factor_prefix_dict={}, load_factor=True, workers=0, if_score=False, show_tqdm=True):
        '''
        评估因子数据，计算因子的IC、RankIC、HedgeRet等回测结果

        参数：
            factor_pack_name: 因子包名称
            factor_prefix_dict: 因子前缀字典，用于指定需要评估的因子
            load_factor: 是否加载已存在的因子数据
            workers: 用于数据加载的工作进程数
            if_score: 是否使用评分模型
            show_tqdm: 是否显示进度条
        '''
        # 根据因子包名称决定因子列表的获取方式
        if load_factor:
            if 'OLD_FACTOR' not in factor_pack_name:
                # 如果不是旧因子，加载因子列表
                file_path = rf"{GlobalConfig.factor_output_path}/{factor_pack_name}/{self.date_list[0]}"
                factor_name_list = load_pickle(rf"{file_path}/factor_list.pkl")
            else:
                # 如果是旧因子，根据信号范围生成因子名称
                start_signal = int(factor_pack_name.split(':')[1].split('-')[0])
                end_signal = int(factor_pack_name.split(':')[1].split('-')[1])
                factor_name_list = ['OLD_FACTOR' + '_' +str(i) for i in range(start_signal, end_signal)]
        else:
            # 如果不加载因子，则使用提供的因子前缀字典的键作为因子名称列表
            factor_name_list = list(factor_prefix_dict.keys())
        
        # 设置因子名称列表
        self.factor_name_list = factor_name_list

        # 按照周期划分日期列表
        period_date_split = [self.date_list[i:i + self.period] for i in range(0, len(self.date_list), self.period)]
        
        # 初始化存储结果的列表
        ic_list, ric_list, hedgeret_list, styleexp_list = [], [], [], []

        # 定义数据集类，用于批量加载回测数据
        class SDataset2(torch.utils.data.Dataset):
            def __init__(self, Backtest, period_date_split, function, function2) -> None:
                '''
                数据集构造函数
                '''
                self.Backtest = Backtest
                self.period_date_split = period_date_split  # 日期分割
                self.function = function  # 因子评估函数
                self.function2 = function2  # 风格因子评估函数

            def __getitem__(self, index):
                '''
                通过索引获取一段数据
                '''
                date_list = self.period_date_split[index]  # 获取当前周期的日期
                result = self.function(self.Backtest, date_list, factor_pack_name, factor_prefix_dict, {}, load_factor, if_score)  # 获取因子评估结果
                result2 = self.function2(self.Backtest, date_list, factor_pack_name, factor_prefix_dict, {}, load_factor, if_score)  # 获取风格因子评估结果
                return result[0], result[1], result[2], result2  # 返回因子结果与风格因子结果

            def __len__(self):
                '''
                返回数据集的长度（即周期数量）
                '''
                return len(self.period_date_split)

        # 使用PyTorch的数据加载器进行批量加载
        dl = DataLoaderX(SDataset2(self, period_date_split, eval_fac_sec_hs, eval_style_factor), collate_fn=lambda x: x[0], batch_size=1, num_workers=workers, shuffle=False, drop_last=False)

        # 根据是否显示进度条来执行回测
        if show_tqdm:
            # 显示进度条
            for ic, ric, hedgeret, styleexp in tqdm(dl, desc='正在进行分钟级因子回测', total=len(dl)):
                ic_list.append(ic)
                ric_list.append(ric)
                hedgeret_list.append(hedgeret)
                styleexp_list.append(styleexp)
        else:
            # 不显示进度条
            for ic, ric, hedgeret, styleexp in dl:
                ic_list.append(ic)
                ric_list.append(ric)
                hedgeret_list.append(hedgeret)
                styleexp_list.append(styleexp)

        # 汇总所有日期的IC、RankIC、HedgeRet等回测结果
        ic_result = torch.vstack(ic_list).cpu()  # 汇总IC结果
        ric_result = torch.vstack(ric_list).cpu()  # 汇总RankIC结果
        hedgeret_result = torch.vstack(hedgeret_list).cpu()  # 汇总HedgeRet结果
        styleexp_result = torch.vstack(styleexp_list).cpu()  # 汇总风格因子结果

        # 保存回测结果
        self.ic_result = ic_result
        self.ric_result = ric_result
        self.hedgeret_result = hedgeret_result
        self.styleexp_result = styleexp_result
        return

    def visualization_all(self, factor_pack_name, save_pdf=True, show_plot=True):
        '''
        可视化所有因子的回测结果

        参数：
            factor_pack_name: 因子包名称
            save_pdf: 是否保存为PDF
            show_plot: 是否显示图形
        '''
        for factor_name in self.factor_name_list:
            # 可视化每个因子的回测结果
            self.visualization_single(factor_name, save_pdf=save_pdf, show_plot=show_plot)

        # 如果需要保存回测结果为PDF文件
        if save_pdf:
            save_path = rf"{GlobalConfig.sample_backtest_result_path}"  # 设定保存路径
            torch.save(self.ic_result, rf"{save_path}/{factor_pack_name}_ic_result.pkl")  # 保存IC结果
            torch.save(self.ric_result, rf"{save_path}/{factor_pack_name}_ric_result.pkl")  # 保存RankIC结果
            torch.save(self.hedgeret_result, rf"{save_path}/{factor_pack_name}_hedgeret_result.pkl")  # 保存HedgeRet结果
            torch.save(self.styleexp_result, rf"{save_path}/{factor_pack_name}_styleexp_result.pkl")  # 保存风格因子结果
    
    def visualization_single(self, factor_name, save_pdf=False, show_plot=False):
        '''
        可视化，对每个因子的表现进行回测。
        '''
        if save_pdf:
            pdf = PdfPages(rf"{GlobalConfig.sample_backtest_result_path}/{factor_name}.pdf")

        if self.factor_name_list is None or self.ic_result is None:
            raise ValueError("请先运行eval_fac_sec_ts_all函数以获取测试结果")

        idx = self.factor_name_list.index(factor_name)

        ic_values = self.ic_result[:,:,idx,:].cpu()#.numpy()
        ric_values = self.ric_result[:,:,idx,:].cpu()#.numpy()
        hedgeret_values = self.hedgeret_result[:,:,idx,:,:].cpu()#.numpy()

        # #1.描述性统计表格
        direction = (ic_values.nanmean(dim=2).nanmean(dim=0) > 0).int().tolist()
        describe_table = pd.DataFrame({
            'IC_mean': ic_values.nanmean(dim=2).nanmean(dim=0).numpy(),
            'IC_std': ic_values.nanmean(dim=2).std(dim=0).numpy(),
            'ls_ret': hedgeret_values.nanmean(dim=2).nanmean(dim=0)[list(range(len(self.label_list))), [{1:-1,0:0}[direction[i]] for i in range(len(self.label_list))]] - \
                    hedgeret_values.nanmean(dim=2).nanmean(dim=0)[list(range(len(self.label_list))), [{1:0,0:-1}[direction[i]] for i in range(len(self.label_list))]],
            'pure_long': hedgeret_values.nanmean(dim=2).nanmean(dim=0)[list(range(len(self.label_list))), [{1:-1,0:0}[direction[i]] for i in range(len(self.label_list))]],
        }, index=self.label_list)
        describe_table = describe_table.applymap(lambda x: f'{x:.4f}')
        #绘制表格，给表格添加点颜色
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.table(cellText=describe_table.values, colLabels=describe_table.columns, rowLabels=describe_table.index, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(describe_table.columns), cellColours=[['#f2f2f2']*len(describe_table.columns)]*len(describe_table.index), colWidths=[0.1]*len(describe_table.columns))
        ax.set_title(f'Factor {factor_name} Performance', fontsize=16)
        if save_pdf: pdf.savefig()
        if show_plot: plt.show()


        #2.绘制分日IC RIC均值，只绘制1d IC和RIC
        d1_idx = self.label_list.index('1d')
        daily_ic_values = ic_values[:,d1_idx,:].nanmean(dim=1).numpy()
        daily_ric_values = ric_values[:,d1_idx,:].nanmean(dim=1).numpy()
        daily_df = pd.DataFrame({f'IC {daily_ic_values.mean():.4f}':daily_ic_values,f'RankIC {daily_ric_values.mean():.4f}':daily_ric_values},index=self.date_list_period)
        plt.figure(figsize=(12, 6))
        plt.plot(daily_df, linewidth=2.0)
        plt.title(f'IC and RankIC Over Time - {factor_name}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('IC / RankIC', fontsize=14)
        plt.xticks(ticks=range(0, len(self.date_list_period), len(self.date_list_period)//20),rotation=45)
        plt.legend(daily_df.columns, loc='upper left', fontsize=12)
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.legend(daily_df.columns, loc='upper left', fontsize=12)
        # plt.tight_layout()
        if save_pdf: pdf.savefig()
        if show_plot: plt.show()

        #3.绘制分时段IC RIC均值图
        period_len = 30
        period_ic_df = pd.DataFrame(index=self.date_list_period,columns=range(1, GlobalConfig.time_num//period_len+2))
        for i in range(1, GlobalConfig.time_num//period_len+1):
            period_ic_df[i] = ic_values[:,d1_idx,(i-1)*period_len:i*period_len].nanmean(dim=1).numpy()
        period_ic_df[GlobalConfig.time_num//period_len+1] = ic_values[:,d1_idx,(GlobalConfig.time_num//period_len)*period_len:].nanmean(dim=1).numpy()
        period_ic_df = period_ic_df.rename(columns={1:f'0930 {period_ic_df[1].mean():.4f}',2:f'1000 {period_ic_df[2].mean():.4f}',3:f'1030 {period_ic_df[3].mean():.4f}',\
            4:f'1100 {period_ic_df[4].mean():.4f}',5:f'1300 {period_ic_df[5].mean():.4f}',6:f'1330 {period_ic_df[6].mean():.4f}',7:f'1400 {period_ic_df[7].mean():.4f}',\
            8:f'1430 {period_ic_df[8].mean():.4f}'})
        period_ic_df = period_ic_df.fillna(0).cumsum(axis=0)

        plt.figure(figsize=(12, 6))
        plt.plot(period_ic_df, linewidth=2.0)
        plt.title(f'IC culmulated Over Time - {factor_name}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('IC', fontsize=14)
        plt.xticks(ticks=range(0, len(self.date_list_period), len(self.date_list_period)//20),rotation=45)
        plt.legend(period_ic_df.columns, loc='upper left', fontsize=12)
        if save_pdf: pdf.savefig()
        if show_plot: plt.show()

        #4.绘制分时段分组收益图
        period_len = 30
        period_ls_df = pd.DataFrame(index=range(1, GlobalConfig.time_num//period_len+2), columns=range(self.num_groups))
        for i in range(1, GlobalConfig.time_num//period_len+1):
            for j in range(self.num_groups):
                period_ls_df.loc[i,j] = hedgeret_values[:,d1_idx,(i-1)*period_len:i*period_len,j].nanmean(dim=1).numpy().mean()
        i = GlobalConfig.time_num//period_len+1
        for j in range(self.num_groups):
            period_ls_df.loc[i,j] = hedgeret_values[:,d1_idx,(i-1)*period_len:i*period_len,j].nanmean(dim=1).numpy().mean()
        period_ls_df = period_ls_df.rename(index={1:f'0930',2:f'1000',3:f'1030',4:f'1100',5:f'1300',6:f'1330',7:f'1400',8:f'1430'})
        period_ls_df.loc['mean'] = period_ls_df.mean(axis=0)
        period_ls_df.columns = [f'group {x}: {period_ls_df[x].mean():.4f}' for x in period_ls_df.columns]
        period_ls_df.plot(kind='bar',figsize=(12,6),colormap='plasma',grid=True,title='group return mean by time group',legend=True)
        if save_pdf: pdf.savefig()
        if show_plot: plt.show()

        #5.分组收益曲线累计图
        period_len = 30
        period_ret_df = pd.DataFrame(index=self.date_list_period, columns=range(self.num_groups))
        for i in range(self.num_groups):
            period_ret_df[i] = hedgeret_values[:,d1_idx,:,i].nanmean(dim=1).numpy()
        period_ret_df = period_ret_df.cumsum(axis=0)
        period_ret_df.columns = [f'group {x}: {period_ret_df[x].mean():.4f}' for x in period_ret_df.columns]
        period_ret_df.plot(figsize=(12,6),grid=True,title='group return cumsum by time group',legend=True)
        if save_pdf: pdf.savefig()
        if show_plot: plt.show()

        #6.绘制style因子的表现
        period_len = 30
        period_style_df = pd.DataFrame(self.styleexp_result[:,:,idx,:,:].nanmean(dim=2).nanmean(dim=0).numpy(),index=[
            'size','mid_size','liquidity','volatility','momentum'],columns=range(1, self.num_groups+1)).T
        period_style_df.plot(figsize=(12,6),grid=True,title='style factor return by group')
        if save_pdf: pdf.savefig()
        if show_plot: plt.show()

        if save_pdf: pdf.close()

        if show_plot: plt.close()
    
        return