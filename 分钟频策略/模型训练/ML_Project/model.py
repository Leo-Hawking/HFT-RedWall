import torch
from torch import nn
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from torchmetrics import PearsonCorrCoef
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.append(r'/home/datamake94/efficient-kan/src/efficient_kan')
from kan import KAN
sys.path.append(r'/home/datamake94/高频策略/分钟级截面模型/divid_time/TabNet')
from YY_Tabnet import TabNet, TabNetPretraining

def get_loss_fn(loss):
    def mse(preds, y):
        return nn.MSELoss()(preds, y)
    
    def wmse(preds, y):
        _,argsort = torch.sort(preds,descending=True,dim=0)
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape,device=preds.device)
        weight_new = torch.tensor([0.5**((i-1)/(preds.shape[0]-1)) for i in range(1,preds.shape[0]+1)]
                                  ,device=preds.device).unsqueeze(dim=1)
        weight[argsort,:] = weight_new
        weighted_mse = ((preds - y)**2 * weight).sum(dim = 0) / weight.sum(dim = 0)
        return weighted_mse.mean()
    
    def pcc(preds, y):
        assert preds.dim() == 2, preds.size()
        assert preds.size() == y.size(), (preds.size(), y.size())

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        return -cos(preds - preds.mean(dim=0, keepdim=True),
                    y - y.mean(dim=0, keepdim=True)).mean()
    
    def mse_pcc(preds, y):
        return mse(preds,y) * 0.5 + pcc(preds,y) * 100
    
    def ccc(preds, y):
        assert preds.dim() == 2, preds.size()
        assert preds.size() == y.size(), (preds.size(), y.size())

        std_preds = preds.std(dim=0, keepdim=True)
        std_y = y.std(dim=0, keepdim=True)

        return ((2 * pcc(preds,y) * std_preds * std_y) / ((std_preds**2) + (std_y**2) + (preds.mean(dim=0,keepdim=True) - y.mean(dim=0,keepdim=True))**2)).mean()
    
    def wpcc(preds, y):
        _,argsort = torch.sort(preds.abs(),descending=True,dim=0)
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape,device=preds.device)
        weight_new = torch.tensor([0.5**((i-1)/(preds.shape[0]-1)) for i in range(1,preds.shape[0]+1)]
                                  ,device=preds.device).unsqueeze(dim=1)
        weight[argsort,:] = weight_new
        wcov = (preds * y * weight).sum(dim = 0) / weight.sum(dim = 0) - ((preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        return -(wcov / (pred_std * y_std)).mean()

    def wpcc_adj(preds, y):
        _,argsort = torch.sort(preds,descending=True,dim=0)
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape,device=preds.device)
        weight_new = torch.tensor([0.5**((i-1)/(preds.shape[0]-1)) for i in range(1,preds.shape[0]+1)]
                                  ,device=preds.device).unsqueeze(dim=1)
        weight[argsort,:] = weight_new
        wcov = (preds * y * weight).sum(dim = 0) / weight.sum(dim = 0) - ((preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        return -(wcov / (pred_std * y_std)).mean()
    
    def wpcc_n(preds, y):
        _,argsort = torch.sort(preds.abs(),descending=False,dim=0)
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape,device=preds.device)
        weight_new = torch.tensor([0.5**((i-1)/(preds.shape[0]-1)) for i in range(1,preds.shape[0]+1)]
                                  ,device=preds.device).unsqueeze(dim=1)
        weight[argsort,:] = weight_new
        wcov = (preds * y * weight).sum(dim = 0) / weight.sum(dim = 0) - ((preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        return -(wcov / (pred_std * y_std)).mean()
    
    def wccc(preds, y):
        std_preds = preds.std(dim=0, keepdim=True)
        std_y = y.std(dim=0,keepdim=True)
        return ((2 * wpcc(preds,y) * std_preds * std_y) / ((std_preds**2) + (std_y**2) + (preds.mean(dim=0,keepdim=True) - y.mean(dim=0,keepdim=True))**2)).mean()

    def tag_loss(preds, y):
        sig_preds = torch.sigmoid(preds) * 2 - 1
        return nn.MSELoss()(sig_preds, y)
    
    def output(loss):
        return {
        'mse': mse,
        'wmse':wmse,
        'pcc': pcc,
        'mse_pcc':mse_pcc,
        'ccc':ccc,
        'wpcc':wpcc,
        'wpcc_n':wpcc_n,
        'wccc':wccc,
        'wpcc_adj':wpcc_adj,
    }[loss]

    return output(loss)

############################################
#基础结构区
############################################

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer as _BertLayer, BertSelfAttention as _BertSelfAttention, BertAttention as _BertAttention

class BertLayer(_BertLayer):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]

class BertSelfAttention(_BertSelfAttention):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]

class BertAttention(_BertAttention):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]

class FlattenBatchNorm1d(nn.BatchNorm1d):
    "BatchNorm1d that treats (N, C, L) as (N*C, L)"

    def forward(self, input):
        sz = input.size()
        return super().forward(input.view(-1, sz[-1])).view(*sz)

class SafeEmbedding(nn.Embedding):
    "Handle unseen id"

    def forward(self, input):
        output = torch.empty((*input.size(), self.embedding_dim),
                             device=input.device,
                             dtype=self.weight.dtype)

        seen = input < self.num_embeddings
        unseen = seen.logical_not()

        output[seen] = super().forward(input[seen])
#         output[unseen] = torch.zeros_like(self.weight[0])
        output[unseen] = self.weight.mean(dim=0).detach()
        return output

############################################
#网络区
############################################
class YY_transformer(nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()

        nn_szs = [input_size] + [args.nn_szs]
        self.Layer = nn.Sequential(*self.get_layers(args, nn_szs, output_size))
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.post_init()

    def get_layers(self, args, szs, output_size):
        layers = []
        
        layers.append(nn.Linear(szs[0], szs[1]))
#         layers.append(FlattenBatchNorm1d(szs[1]))
        layers.append(nn.SiLU(inplace=True))
    
        if args.dropout > 0.0:
            layers.append(nn.Dropout(p=args.dropout, inplace=True))
        
        out_sz = szs[1]
        for i in range(args.num_mhas):
            layers.append(BertLayer(BertConfig(
                num_attention_heads=args.num_heads,
                hidden_size=out_sz,
                attention_probs_dropout_prob=0.05,
                hidden_dropout_prob=0.05,
                classifier_dropout = 0.05,
                intermediate_size=out_sz)))
        
#         layers.append(nn.Linear(out_sz, szs[-1]))
#         layers.append(FlattenBatchNorm1d(szs[-1]))
#         layers.append(nn.SiLU(inplace=True))
        
        layers.append(nn.Linear(out_sz, output_size))
        return layers

    def forward(self,feature):
        feature = feature.unsqueeze(dim=1)
        Output = self.Layer(feature).squeeze(1)
        return self.batchnorm(Output)

    def post_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, SafeEmbedding)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)

class YY_TabNet(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()

        input_size = input_size
        self.tabnet = TabNet(input_dim=input_size,output_dim=1,
               n_d = args.nn_szs,
               n_a = args.nn_szs,
               gamma = args.relax,
               n_steps=args.n_steps,
               n_independent = args.indep,
               n_shared = args.share,
               mask_type=args.mask)
        # self.batchnorm = nn.BatchNorm1d(1)

    def forward(self,feature):
        Output,_ = self.tabnet(feature)
        # Output,_ = self.tabnet(feature[:,:params._factor_num])
        # Output = (Output - Output.min()) / (Output.max() - Output.min())
        # # Output = torch.hstack([Output,feature[:,-params._ml_factor_num:]])
        # # Output = self.output_layer(Output)
        return Output
    
    def load_weights_from_unsupervised(self, unsupervised_model):
        update_state_dict = copy.deepcopy(self.state_dict())
        for param, weights in unsupervised_model.state_dict().items():
            if param.startswith("tabnet_pretrain.encoder"):
                # Convert encoder's layers name to match
                new_param = param.replace("tabnet_pretrain",'tabnet.tabnet')
            else:
                new_param = param
            if self.state_dict().get(new_param) is not None:
                # update only common layers
                update_state_dict[new_param] = weights
        self.load_state_dict(update_state_dict)

class YY_Kan(nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()

        self.args = args
        self.kan = KAN([input_size] + args.nn_szs + [output_size],
            grid_size=args.grid_sz,
            spline_order=args.spline_order,
            scale_noise=args.scale_noise,
            scale_base=args.scale_base,
            scale_spline=args.scale_spline,
            grid_eps=args.grid_eps,
            )

    def forward(self, feature):
        return self.kan(feature)

class UMPLitModule(LightningModule):
    def __init__(self, args, input_size, model_name = 'tabnet'):
        super().__init__()
        self.args = args
        output_size = 1
        if model_name == 'tabnet':
            self.model_name = 'tabnet'
            self.model = YY_TabNet(args, input_size)
        elif model_name == 'transformer':
            self.model_name = 'transformer'
            self.model = YY_transformer(args, input_size, output_size)
        elif model_name =='kan':
            self.model_name = 'kan'
            # self.model = KAN([input_size, args.nn_szs, output_size])
            self.model = YY_Kan(args, input_size, output_size)
        # if model_name in ['tabnet','tabnet_rnn'] and args.use_pretrain:
        #     print('正在使用预训练模型')
        #     pretrain_model = get_best_model_pretrain(args,input_size,pretrain_path,pretrain_tag,target_date)
        #     self.model.load_weights_from_unsupervised(pretrain_model.model)
        self.test_pearson = PearsonCorrCoef()
        self.last_loss = None
        gamma_use = 0.4 if self.model_name == 'tabnet' else 1 if self.model_name == 'node' else 0.7
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.configure_optimizers()[0], step_size=2, gamma=gamma_use)

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        def get_excess_return(preds,ret):
            _,sort = preds.sort(dim=0,descending=True)
            trade_num = ret.shape[0] // 20
            return ret[sort[:trade_num,0]].mean() - ret[sort[-trade_num:,0]].mean()
        # features,rets = batch
        
        loss_fn = get_loss_fn(self.args.loss)
        
        loss_list = []
        exret_list = []
        for i in range(len(batch)):
            feature,ret = batch[i]
            if feature.shape[0] == 0:
                continue
            preds = self.forward(feature)
            loss = loss_fn(preds,ret)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss_list.append(loss)
        
        if len(loss_list) == 0:
            #返回一个梯度为0，值为0的tensor
            return None
        # total_loss = sum(loss_list) / len(loss_list)
        ###加权loss，优先学效果较好样本
        # loss_list = sorted(loss_list)[::]
        # weight = [0.5**((i-1)/(len(loss_list)-1)) for i in range(1,len(loss_list)+1)] if len(loss_list) > 1 else [1]
        # weight_sum = sum(weight)
        # total_loss = sum([loss_list[i]*weight[i]/weight_sum for i in range(len(loss_list))])
        total_loss = sum(loss_list) / len(loss_list)

        self.log('train_loss', total_loss)
        return total_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        def get_excess_return(preds,ret):
            _,sort = preds.sort(dim=0,descending=True)
            trade_num = ret.shape[0] // 20
            return ret[sort[:trade_num,0]].mean() - ret[sort[-trade_num:,0]].mean()
        def get_excess_return_long(preds,ret):
            _,sort = preds.sort(dim=0,descending=True)
            trade_num = ret.shape[0] // 20
            return ret[sort[:trade_num,0]].mean()
        def get_excess_return_short(preds,ret):
            _,sort = preds.sort(dim=0,descending=True)
            trade_num = ret.shape[0] // 20
            return -ret[sort[-trade_num:,0]].mean()

        feature,ret = batch[0][0], batch[0][1]
        pcc_loss = get_loss_fn('pcc')

        preds = self.forward(feature)
        pcc = pcc_loss(preds, ret)
        self.log(f'{stage}_pearson', -pcc, prog_bar=True, sync_dist=True)
        
        exret = get_excess_return(preds,ret)
        self.log(f'{stage}_exret', exret, prog_bar=True, sync_dist=True)

        exret_long = get_excess_return_long(preds,ret)
        self.log(f'{stage}_exret_long', exret_long, prog_bar=True, sync_dist=True)

        exret_short = get_excess_return_short(preds,ret)
        self.log(f'{stage}_exret_short', exret_short, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
        }
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        optim_config = {
            'optimizer': optimizer,
        }
        gamma_use = 0.4 if self.model_name == 'tabnet' else 1 if self.model_name == 'node' else 0.7
        optim_config['lr_scheduler'] = {
            'step_lr': torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args.step_sz, gamma=gamma_use),
        }[self.args.lr_scheduler]

        return optim_config

    def configure_callbacks(self):
        if self.args.loss != 'wpcc_n':
            callbacks = [
                LearningRateMonitor(),
                ModelCheckpoint(monitor='val_pearson', mode='max', save_last=True,
                                filename='{epoch}-{val_exret_long:.4f}-{val_exret:.4f}-{val_pearson:.4f}',save_top_k=5),
            ]
        else:
            callbacks = [
                LearningRateMonitor(),
                ModelCheckpoint(monitor='val_exret_short', mode='max', save_last=True,
                                filename='{epoch}-{val_exret_short:.4f}-{val_exret:.4f}',save_top_k=5),
            ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.7,
                                                       device='cpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_pearson',
                                           mode='max', patience=10))
        return callbacks