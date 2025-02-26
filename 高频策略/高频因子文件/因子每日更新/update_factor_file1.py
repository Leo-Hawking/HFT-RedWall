import os
import datetime
import pickle as pkl

if __name__ == '__main__':
    today = datetime.date.today().strftime('%Y%m%d')
    with open(r'/home/datamake118/data0/support_data/trade_days_dict.pkl','rb') as f:
        trade_days_dict = pkl.load(f)
    trade_days_list = sorted([x.strftime('%Y%m%d') for x in trade_days_dict['trade_days']])
    target_date = [x for x in trade_days_list if x<today][-1]

    os.system(f'python3 /home/datamake118/equity_sec_strategy/因子每日更新/daily_calc_hft_factor_all_final.py {target_date} 0 &')
    os.system(f'python3 /home/datamake118/equity_sec_strategy/因子每日更新/daily_calc_hft_factor_sec2_final.py {target_date} 0 &')
    os.system(f'python3 /home/datamake118/equity_sec_strategy/因子每日更新/daily_calc_hft_factor_sec3_final.py {target_date} 0 &')
    os.system(f'python3 /home/datamake118/equity_sec_strategy/因子每日更新/daily_calc_hft_factor_sec4_final.py {target_date} 0 &')