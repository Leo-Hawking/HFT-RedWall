import operator
import traceback
import sys
import gc
from my_deap import creator, base, tools, gp #遗传规划函数
from basic_functions import *
from gp_functions import *
from gp_config import *    #基本配置
from calc_method import *
import pickle as pkl
import re 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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

# # def formula_to_value(formula, factor_dict):
# #     func = my_compile(formula)
# #     return func(factor_dict)

# def formula_to_value(formula, factor_dict, date_num):
#     func = my_compile(formula)
#     result = func(factor_dict, date_num)
#     print(result[0])
#     return result

''' 遗传算法 '''
class GPLearn():
    def __init__(self,args,
                 function_set,func_map_dict,
                 random_state,
                 **kwargs):

        self.args=args
        self.function_set=function_set
        self.func_map_dict=func_map_dict
        
        np.random.seed(random_state)
        random.seed(random_state)
        self.random_state=random_state

        self.pset = None
        self.tool = None
        self.logbook = None
        self.best_inds = []
        self.best_ind = None

    def fit(self, metric, FS_dict):

        # 1. 生成 primitive Set
        self.pset = self._generate_pset(self.function_set, self.func_map_dict)
        self.tool = base.Toolbox()

        # 2. Fitness
        self._generate_fitness_individual()

        # 3. 定义个体（Individual）、总体（Population）生成方式
        self._register_init_methods(self.tool, self.pset)

        # 4. 定义进化（Evaluate）、选择（Select）方式
        self._register_evaluate_select_methods(self.tool,
                                               metric,
                                               FS_dict)

        # 5. 定义交叉（Mate）、变异（Mutate）的方式
        self._register_mate_mutate_methods(self.tool, self.pset)

        # 6.  迭代训练
        self._fit(self.tool,FS_dict)

    def _generate_pset(self, function_set, func_map_dict):
        """ 生成primitive set"""
        # 输入因子集合
        factor_names = get_factor_names()
        factor_input_nums= int(len(factor_names)*self.args.factor_input_ratio)
        pset = gp.My_PrimitiveSet('main', factor_input_nums, input_type='basic')
        
        factor_nums_list = random.sample(range(len(factor_names)), factor_input_nums)
        for i,nums in enumerate(factor_nums_list):
            exec( "pset.renameArguments(ARG%i='%s')"%(i, factor_names[nums]))

        # 添加算子集合
        for symb in function_set:
            pset.addPrimitive(func_map_dict[symb][0], len(func_map_dict[symb][1]), args=func_map_dict[symb][1], name=symb)

        # 常数集合
        pset.addConstTerminal('rand_d', self.args.delay_list, 'const_delay', True)  
        pset.addConstTerminal('rand_e', self.args.exp_list, 'const_exp', True)
        pset.addConstTerminal('rand_q', self.args.quan_list, 'const_quan', True)
        return pset

    def _generate_fitness_individual(self):
        """
            为Creator 生成Fitness类和Individual类
        """
        # 创建fitness类、individual类
        creator.create('FitnessMax', base.Fitness, weights = (1.0, ))
        creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMax)

    def _register_init_methods(self, tool, pset):
        """ 定义个体生成方法，种群生成方法 """
        init_method_dict = {
            'half and half': gp.genHalfAndHalf
        }
        tool.register('expr', init_method_dict[self.args.init_method], pset=pset, min_=self.args.init_depth[0], max_=self.args.init_depth[1])
        tool.register('individual', tools.initIterate, creator.Individual, tool.expr)
        tool.register('population', tools.initRepeat, list, tool.individual)

    def _register_evaluate_select_methods(self, tool, metric, FS_dict):
        """ 定义evaluate、select """
        tool.register('evaluate', metric, self.args, FS_dict)   #对个体进行fitness计算
        tool.register('selectTournament', tools.selTournament, tournsize=self.args.tournament_size)  # 选择最优N个
        tool.register('selectBest', tools.selBest)  # 选择最优N个，默认按照适应度

    def _register_mate_mutate_methods(self, tool, pset):
        """ 定义 mate、mutate """
        tool.register('mate', gp.cxOnePoint)                                     # 单点交叉 会产生两棵树
        tool.register('expr_mut', gp.genFull, pset=pset, min_ = 0, max_ = 3)     # 生成一个subtree，且长度至少为1
        tool.register('mutUniform', gp.mutUniform, expr=tool.expr_mut, pset=pset)    # subtree mutation
        tool.register('mutInsert', gp.mutInsert, pset=pset)    # subtree mutation
        tool.register('mutNodeReplacement', gp.mutNodeReplacement, pset=pset)    # subtree mutation

        # 限制一下交叉变异后的树深度，最大12
        tool.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))
        tool.decorate('mutUniform', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))
        tool.decorate('mutInsert', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))
        tool.decorate('mutNodeReplacement', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))

    def _fit(self, tool, FS_dict):
        """ 迭代训练，获得best_inds """
        
        popSize=self.args.populations
        ngen=self.args.gens
        cxpb=self.args.p_crossover
        mutpb=self.args.p_mutation
        num_rounds=self.random_state

        # 生成初始种群
        pop = tool.population(n=popSize)

        update_gp_factor_hist(FS_dict=FS_dict) 
        pds = PopDataset(pop, tool.evaluate)
        dataloader=DataLoader(pds,collate_fn=lambda x:x[0],batch_size=1,num_workers=self.args.num_workers, shuffle=False,drop_last=False)
        for batch in tqdm(dataloader,total=len(pds)):
            i=batch[0]
            fit=batch[1]
            pop[i].indicators = fit[0]
            pop[i].style_location = fit[1]
            pop[i].fitness.values = (fit[0][0], )

        num = 0
#         for ind in tqdm(pop, desc='R_%s_g_%s:' % (self.param_dict['random_state'], num)):
#             fit  = tool.evaluate(ind)
#             ind.indicators = fit
#             ind.fitness.values = (fit[0],fit[1],)

        best_inds = []    # 记录每一代的最优个体
        hof = tools.HallOfFame(1)
        hof.update(pop)
        print([tst.fitness.values for tst in pop])
        print([(tst.indicators[0], tst.fitness.values) for tst in hof.items])
        best_ind = hof.items[0]
        gen=0
        
        self.best_ic = best_ind.indicators[1]
        
        #torch.stack([score, abs_ic, hedge_ret, ic, longret]).numpy(), style_location
        score, abs_ic, hedge_ret, ic, longret = best_ind.indicators
        print('gen_%s: '%gen + str(best_ind) + '\n' +'ic:%s,  ret:%s, score:%s, length:%s, longret:%s'%(
            ic, hedge_ret, score,len(best_ind), longret))
        
        logbook = tools.Logbook()
        logbook.header = 'gen', 'eval', 'Avg_Length', 'Avg_Fitness', 'Best_Ind_Length', 'Best_Ind_Fitness',
        
        # if (abs_ic >= self.args.ic_thres and hedge_ret >=self.args.ls_thres and score>=self.args.score_thres):  #三种情况同时满足则接纳该因子
        if (abs_ic >= self.args.ic_thres) or (hedge_ret >= self.args.ls_thres):
            best_inds.append(best_ind)
            self.save_formula(best_ind, gen)
            record = {
                'Avg_Length':       bn.nanmean([len(p) for p in pop]),
                'Avg_Fitness':      bn.nanmean([p.fitness.values[0] for p in pop]),
                'Best_Ind_Length':  len(best_ind),
                'Best_Ind_Fitness': (ic, hedge_ret),
            }
            
            logbook.record(gen=gen, eval=popSize, **record)
            gen += 1

        print('--------开始迭代--------')
        # 迭代次数少于设定值或者最大IC没有到达标准时，继续迭代
        while num<ngen:
            num += 1
            print('第',num,'次迭代')
            offSpring = tool.selectBest(pop, round(len(pop)*3/5))
            new_pop = tool.population(n=round(len(pop)*2/5))
            offSpring = offSpring + new_pop
            offSpring = list(map(tool.clone, offSpring))

            # crossover  依次两两交叉
            for child1,child2 in zip(offSpring[1::2],offSpring[::2]):
                if np.random.random() < cxpb:
                    tool.mate(child1,child2)    # 交叉产生两个新的个体
#                     del child1.fitness.values   # 新的个体需要重新evaluate，所以删掉原来的fitness value
#                     del child2.fitness.values

            # mutate  变异
            for child in offSpring:       # 每个个体都有概率进行变异
                if np.random.random() < (2 / (1 + np.exp(-(gen+1)/2)) -1.2):
                    # mutate = random.choice([tool.mutUniform, tool.mutInsert, tool.mutNodeReplacement])
                    mutate = tool.mutUniform
                    mutate(child)
#                     del child.fitness.values

#             # evaluate  进化
#             invalid_fit = []      # 找到需要重新计算fitness的个体,已经产生了交叉和变异
#             for ind in offSpring:
#                 if not ind.fitness.valid:
#                     invalid_fit.append(ind)

            update_gp_factor_hist(FS_dict=FS_dict) 
            pds = PopDataset(offSpring, tool.evaluate)
            dataloader=DataLoader(pds,collate_fn=lambda x:x[0],batch_size=1,num_workers=self.args.num_workers,shuffle=False,drop_last=False)
            for batch in tqdm(dataloader,total=len(pds)):
                i=batch[0]
                fit=batch[1]
                offSpring[i].indicators = fit[0]
                offSpring[i].style_location = fit[1]
                offSpring[i].fitness.values = (fit[0][0],)

#             for ind in tqdm(invalid_fit, desc='R_%s_g_%s:' % (self.param_dict['random_state'], num)):
#                 fit = tool.evaluate(ind)
#                 ind.indicators = fit
#                 ind.fitness.values = (fit[0], fit[1],)

            # 更新种群
            pop = offSpring
            hof.clear() #ysw强制清空名人堂，只挑选当代最好的
            hof.update(pop)
            # 找到本代的最优个体，并打印
            best_ind = hof.items[0]
            
            score, abs_ic, hedge_ret, ic, longret = best_ind.indicators
            print('gen_%s: '%gen + str(best_ind) + '\n' +'ic:%s,  ret:%s, score:%s, length:%s, longret:%s'%(ic, hedge_ret, score,len(best_ind), longret))

            # if (abs_ic >= self.args.ic_thres and hedge_ret >= self.args.ls_thres and score>=self.args.score_thres):
            if (abs_ic >= self.args.ic_thres) or (hedge_ret >= self.args.ls_thres):
                best_inds.append(best_ind)
                record = {  'Avg_Length': bn.nanmean([len(p) for p in pop]),
                            'Avg_Fitness': bn.nanmean([p.indicators[1] for p in pop]),
                            'Best_Ind_Length': len(best_ind),
                            'Best_Ind_Fitness': (ic, hedge_ret),
                }
                logbook.record(gen=gen, eval=popSize, **record)

                self.logbook = logbook
                self.best_ind = best_ind
                self.best_ic = best_ind.indicators[1] if best_ind.indicators[1] > self.best_ic else self.best_ic
                self.save_formula(best_ind, gen)
                gen += 1

        self.best_inds = best_inds
        print('--------迭代结束-------')
        if len(logbook)>0:
            print(logbook)
        else:
            print('没有找到过检的因子')
        TempData.factor_values_hist={} #完整一轮完成后，清空历史因子数据

    def save_formula(self, ind, gen_num):
        summary_file = os.path.join(GpConfig.summary_path, "GP_Formula_Round_{}.csv".format(self.random_state))
        score, abs_ic, hedge_ret, ic, long_ret = ind.indicators
        style_location = ind.style_location
        
        is_ic = 0
        if abs_ic > self.args.ic_thres:
            is_ic = 1

        summary = pd.DataFrame({'round':[self.random_state],
                                'gen':[gen_num],
                                'formula':[str(ind)],
                                'IC_mean':[ic],
                                'ret':[hedge_ret],
                                'Length':[len(ind)],
                                'longret':[long_ret],
                                'is_ic':[is_ic],
                                'style_location':[style_location],
                                })
        
        if os.path.isfile(summary_file):
            summary.to_csv(summary_file, index=False, header=False, mode='a+')
        else:
            summary.to_csv(summary_file, index=False, header=True, mode='a+')

################################################################################
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

if __name__ == '__main__':
    random_begin=0
    print('编码为%s'%random_begin)
    num=0
    factor_names = get_factor_names()

    factor_names_all = factor_names + [GpConfig.args.label]
        
    while 1:
        random_state_=random_begin+5*num
        np.random.seed(random_state_)
        random.seed(random_state_)
        
        #弃用
        # rand_date_list= sorted(random.sample(GpConfig.date_list, GpConfig.args.sample_date_num))
        # all_sec_list=list(GlobalConfig.trade_time_loc_dict.keys())
        # base_data_slice_list=[]
        # for date in rand_date_list:
        #     ind= GpConfig.date_list_all.index(date)
        #     base_data_slice_list+=[slice(ind*GpConfig.sec_num_all,(ind+1)*GpConfig.sec_num_all)]
        #Base_data_slice_dict={factor_name:factor[output_data_time_loc_in_base_data] for factor_name,factor in tqdm(Base_data_dict.items())}
        
        FS_dict = get_FS_dict(GpConfig.args.large_sample_num)

        print('数据读取完成')
        deap_gp = GPLearn(GpConfig.args, function_set=function_set, func_map_dict=func_map_dict, random_state=random_state_)
        deap_gp.fit(fit_evaluation, FS_dict)
        num+=1