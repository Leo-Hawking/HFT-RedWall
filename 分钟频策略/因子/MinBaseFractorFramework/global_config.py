import pickle
import pandas as pd

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class GlobalConfig:
    # 基础数据和计算设定
    time_num = 237
    base_data_freq = 'minute'
    base_data_dtype='float32'
    factor_data_dtype='float32'  

    # 每个循环进行的时间步
    time_step_each_loop = time_num * 1
    # 自动选择time_step_each_loop数量。如果其设置过大导致显存不足，尝试减小每个循环时间步的数量
    time_step_each_loop_try_list = [237 * x for x in [1]]

    # 最终因子在基础数据的时间步的位置
    output_data_time_setp_loc = []

    # 因子计算使用的设备
    use_device = 'cpu'  # 'cuda:0'
    check_gpu_mem_leak = True

    # 路径
    sample_backtest_result_path = "/home/datamake94/data_nb0/min_factor_test_result"
    global_root_path ="/home/intern1/hft_database/nas4"
    daily_support_path = rf"/home/datamake94/data_nb0/sec_daily_support"
    base_data_root_path = rf'{global_root_path}/Stock60sData'
    base_data_root_path2 = rf"{global_root_path}/Stock60sData"
    factor_output_path = rf"/home/intern1/hft_database/nas7/Stock60sFactor_Framework"

    basic_data_path = r'/home/datamake94/data_nb0'
    stock_daily_data_path1 = rf"{basic_data_path}/ohlc_fea" # 读取开高低收量额

    style_path = r'/home/datamake94/database_00/style_factor/'

    # 位置信息
    output_data_time_60s_loc_in_base_data = []
    
    _config_data_path = rf"{global_root_path}/support_data"
    support_data_path = rf"{global_root_path}/support_data"

    trade_time_loc_dict = load_pickle(rf'{_config_data_path}/trade_time_60s_loc_dict.fea')
    _sec_list = sorted([i for i in trade_time_loc_dict.keys()])
    
    lob_data_path = rf"/home/intern1/hft_database/nas3/sec_lobdata"
    sec_data_path = rf"/home/intern1/hft_database/nas3"
    
    index_path=rf'{global_root_path}/IndexDailyData'
    zz1000_index=pd.read_feather(index_path+'/constituent_ZZ1000.fea').set_index('date')