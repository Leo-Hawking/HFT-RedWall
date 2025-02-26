import sys
import os
import warnings
from argparse import ArgumentParser
sys.path.append(r'/home/datamake94/秒级高频策略')
from ML_Project.training_config import params
from ML_Project.model import *
from ML_Project.load_data import *

params.period=93000   #全时段训练，或分时段训练
params.agg_method='minmax'  #统一每日min_max的方式，分minmax或者平均
params.future_ret='5m'    #预测未来区间
params.if_decorr=True   #是否要去除相关性
params.threshold=0.2

torch.multiprocessing.set_sharing_strategy('file_system')
cpu_num = 40
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7' if params.period!=93000 else '0,1,2,3'

torch.set_num_threads(cpu_num)
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")

def parse_args():  #数据预处理+去除相关性后，得到因子数目
    parser = ArgumentParser()
    parser.add_argument('--period', default=params.period)
    parser.add_argument('--agg_method', default=params.agg_method)
    parser.add_argument('--future_ret', default=params.future_ret)
    parser.add_argument('--if_decorr',default=params.if_decorr)
    parser.add_argument('--threshold',default=params.threshold)
    
    parser.add_argument('--processLabel',default=True)
    parser.add_argument('--tmpfs_fast', default=True)
    parser.add_argument('--graph', default=True)
    
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy',default='ddp')
    parser.add_argument('--find_unused_parameters',default=False)
    parser.add_argument('--threads', type=int, default=40)
    parser.add_argument('--persistent_workers', default=False)
    parser.add_argument('--seed', type=int, default=2929) 
    
    #Network structure parameters
    #cross_num=0 if params.period==93000 and params.future_ret=='15m' else 3
    parser.add_argument('--cross_num', type=int, default=3)
    # parser.add_argument('--factor_num', type=int, default=factor_num)
    parser.add_argument('--ml_num', type=int, default=0)
    parser.add_argument('--add_noise', default=False)  #DCN先不加噪声!!
    
    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)   #注意此处改动
    parser.add_argument('--accumulate_grad_batches', type=int, default=64)  
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_gamma', type=float, default=0.95)
    parser.add_argument('--lr_stepsz', type=int, default=2)
    
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'adamw'])
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    
    parser.add_argument('--nn_szs', type=float, default=[256,64])
    parser.add_argument('--num_mhas', default=2)
    parser.add_argument('--num_heads', default=1)
    
    parser.add_argument('--loss', default='wccc', choices=['huber','mse', 'pcc', 'wpcc','wccc','wccc_ind','wpcc_ind'])
    
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--swa', action='store_true', help='whether to perform Stochastic Weight Averaging')
    parser.add_argument('--lr_scheduler', default='step_lr')
    parser.add_argument('--dropout', default=0)
    # Test
    parser.add_argument('--test', action='store_true')

    # Checkpoint
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')

    # Handle kaggle platform
    args, unknown = parser.parse_known_args(args=[])
    return args

def get_model_name(args,date):
    return '-'.join(filter(None, [  # Remove empty string by filtering
        date,
        'yy_test',
        'update1010',
        'DCN+BERT','ob_ret_%s'%args.future_ret,'%s'%args.period,args.agg_method, 'citic_2',
        'NN'+'x'.join(str(sz) for sz in args.nn_szs),
        f'pLabel{args.processLabel}',
        f'graph{args.graph}',
        f'cross{args.cross_num}',
        f'thre{args.threshold}',
        f'decorr{args.if_decorr}',
        # f'factor{args.factor_num}',
        #f'ml{args.ml_num}',
        f'noise{args.add_noise}',
        f'gpus{args.devices}',
        f'maxepch{args.max_epochs}',
        f'btch{args.batch_size}',
        f'accu_btch{args.accumulate_grad_batches}',
        f'drop{args.dropout}',
        f'schd{args.lr_scheduler}',
        f'loss{args.loss}',
#         f'mhas{args.num_mhas}',
        f'lr{args.lr}',
        f'lr_gamma{args.lr_gamma}',
        f'lr_stepsz{args.lr_stepsz}',
        f'wd{args.weight_decay}',
    ])).replace(' ', '')

def train_single(args, date, train_date_list, valid_date_list, test_date_list):
    seed_everything(args.seed)
    
    #基本数据读取
    name = get_model_name(args,date)
    params.name=name
    
    if args.tmpfs_fast:
        tmpfs_path = os.path.join(params.tmpfs_path,name)
        os.makedirs(tmpfs_path,exist_ok=True)
        os.system('find %s -type f -exec rm -f {} \;'%tmpfs_path)  #删除所有文件
    
    logger = TensorBoardLogger(save_dir=params._model_path, name=name)
    trainer = Trainer(
            max_epochs=args.max_epochs,
            devices=args.devices,
            accelerator=args.accelerator,
            strategy=args.strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            num_sanity_val_steps = 1,
            log_every_n_steps=args.log_every_n_steps,
            val_check_interval=0.5,
            logger=logger,
            deterministic=True,
            precision='bf16')
    
    litmodel = UMPLitModule(args,params.factor_num)
    dm = UMPDataModule(args,date,train_date_list,valid_date_list)
    trainer.fit(litmodel, dm)

    #存储模型使用的最大最小tensor
    save_path = params._model_path + f'//{name}/version_0//'
    all_list=get_default_factor_list(if_static=True)
    torch.save(all_list, save_path + 'all_factor_list.pt')
    torch.save(Dataset.min_se,save_path + 'min_tensor.pt')
    torch.save(Dataset.max_se,save_path + 'max_tensor.pt')
    torch.save(Dataset.valid_ind,save_path + 'valid_ind.pt')
    
    if args.tmpfs_fast:
        tmpfs_path = os.path.join(params.tmpfs_path,name)
        os.system('find %s -type f -exec rm -f {} \;'%tmpfs_path)  #删除所有文件
    
if __name__ == '__main__':
    year_month_list=get_year_month(params.date_list_all)
    for month in year_month_list[year_month_list.index('202405'):year_month_list.index('202406')][::2]:
        date=get_first_date(month,params.date_list_all)
        begin_date=get_first_date(year_month_list[year_month_list.index(month)-24],params.date_list_all)
        begin_date=begin_date if begin_date>='20200430' else '20200430'
        end_date=get_first_date(year_month_list[year_month_list.index(month)+2],params.date_list_all)
        print('训练集数据起始日为{}，模型样本外预测期为{}——{}'.format(begin_date,date,end_date))
        
        random.seed(int(date))
        date_list = get_date_list(begin_date,date)
        test_date_list = get_date_list(date, end_date)

        valid_date_list= date_list[-int(len(date_list)*0.1):] #random.sample(date_list,int(len(date_list)*0.1)) 
        train_date_list=[x for x in date_list if x not in valid_date_list]
        update_training_minmax(date_list,date,params.future_ret,params.period,method=params.agg_method,threshold=params.threshold,if_decorr=params.if_decorr)
        args = parse_args()
        print('%s, %s, 有效因子数: %s'%(params.period,params.future_ret,params.factor_num))
        train_single(args,date,train_date_list, valid_date_list,test_date_list)