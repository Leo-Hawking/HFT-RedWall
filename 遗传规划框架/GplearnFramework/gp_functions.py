import numpy as np
import bottleneck as bn
from numpy.lib.stride_tricks import sliding_window_view as sliding
import pandas as pd
from gp_config import *

func_map_dict = {
    ##简单算子
    'add': (add, ['basic', 'basic']),
    'sub': (sub, ['basic', 'basic']),
    'mul': (mul, ['basic', 'basic']),
    'div': (div, ['basic', 'basic']),
    'Max': (Max, ['basic', 'basic']),
    'Min': (Min, ['basic', 'basic']),
    'Abs': (Abs, ['basic']),
    'neg': (neg, ['basic']),
    'sqrt':(sqrt, ['basic']),
    'log': (log, ['basic']),
    'pow': (pow, ['basic','const_exp']),
    # 'sign_square': (sign_square, ['basic','const_exp']),
    'inv': (inv, ['basic']),
    'sigmoid':(sigmoid, ['basic']),
    'sign':(sign, ['basic']),
    
    #时序算子
    'ts_diff_abs_sum':(ts_diff_abs_sum, ['basic','const_delay']),
    'ts_delay':(ts_delay, ['basic','const_delay']),
    'ts_delta':(ts_delta, ['basic','const_delay']),
    'ts_corr':(ts_corr, ['basic','basic','const_delay']),
    'ts_rankcorr':(ts_rankcorr, ['basic','basic','const_delay']),
    'ts_cov':(ts_cov, ['basic','basic','const_delay']),
    'ts_decay_linear':(ts_decay_linear, ['basic','const_delay']),
    
    'ts_rank':(ts_rank, ['basic','const_delay']),
    'ts_min':(ts_min, ['basic','const_delay']),
    'ts_max':(ts_max, ['basic','const_delay']),
    'ts_argmin':(ts_argmin, ['basic','const_delay']),
    'ts_argmax':(ts_argmax, ['basic','const_delay']),
    'ts_sum':(ts_sum, ['basic','const_delay']),
    'ts_mean':(ts_mean, ['basic','const_delay']),
    
    'ts_quantile':(ts_quantile, ['basic','const_quan','const_delay']),
    'ts_qua':(ts_qua, ['basic','const_quan','const_delay']),
    'ts_return':(ts_return, ['basic','const_delay']),
    'ts_mean_return':(ts_mean_return,  ['basic','const_delay']),
    
    'ts_product':(ts_product,  ['basic','const_delay']),
    'ts_stddev':(ts_stddev,  ['basic','const_delay']),
    'ts_demean':(ts_demean,  ['basic','const_delay']),
    'ts_zscore':(ts_zscore,  ['basic','const_delay']),
    'ts_skew':(ts_skew,  ['basic','const_delay']),
    'ts_kurt':(ts_kurt,  ['basic','const_delay']),
    'ts_regbeta':(ts_regbeta,  ['basic','basic','const_delay']),
    'ts_regres':(ts_regres,  ['basic','basic','const_delay']),
    'ts_rankregbeta':(ts_rankregbeta,  ['basic','basic','const_delay']),
    'ts_rankregres':(ts_rankregres,  ['basic','basic','const_delay']),
    'ts_wma':(ts_wma,  ['basic','const_delay']),
    
    #截面算子
    'cs_rank':(cs_rank, ['basic']),
}

function_set = set(func_map_dict.keys())