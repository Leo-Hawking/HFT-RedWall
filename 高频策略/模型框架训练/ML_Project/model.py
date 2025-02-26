import torch
from torch import nn
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from torchmetrics import PearsonCorrCoef
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

def get_loss_fn(loss):
    def huber(preds, y):
        return nn.HuberLoss(delta=1)(preds, y)   #target的标准差应在0.3-0.5左右（delta接近2-3倍标准差）
    
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
    
    def calc_rank(input_,dim=0):
        _, indices = torch.sort(input_, dim=dim)
        _, indices = torch.sort(indices, dim=dim)
        return indices/input_.size(dim)
    
    def pcc(preds, y):
        assert preds.dim() == 2, preds.size()
        assert preds.size() == y.size(), (preds.size(), y.size())

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        return -cos(preds - preds.mean(dim=0, keepdim=True),
                    y - y.mean(dim=0, keepdim=True)).mean()
    
    def rank_pcc(preds,y):
        return pcc(calc_rank(preds),calc_rank(y))
    
    def mse_pcc(preds, y):
        return mse(preds,y) * 0.5 + pcc(preds,y) * 100
    
    def ccc(preds, y):
        assert preds.dim() == 2, preds.size()
        assert preds.size() == y.size(), (preds.size(), y.size())

        std_preds = preds.std(dim=0, keepdim=True)
        std_y = y.std(dim=0, keepdim=True)

        return ((2 * pcc(preds,y) * std_preds * std_y) / ((std_preds**2) + (std_y**2) + (preds.mean(dim=0,keepdim=True).abs() - y.mean(dim=0,keepdim=True).abs())**2)).mean()
    
    def wpcc(preds, y):
        _,argsort = torch.sort(y.abs(),descending=True,dim=0) #######
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape,device=preds.device)
        weight_new = torch.tensor([0.5**((i-1)/(preds.shape[0]-1)) for i in range(1,preds.shape[0]+1)]
                                  ,device=preds.device).unsqueeze(dim=1)
        weight[argsort,:] = weight_new
        wcov = (preds * y * weight).sum(dim = 0) / weight.sum(dim = 0) - ((preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        return -(wcov / (pred_std * y_std)).mean()
    
    def wpcc_ind(preds, y, weight):
        weight1 = preds.abs()
        weight1 = (weight1 - weight1.mean(dim=0)) / weight1.std(dim=0)
        weight2 = (weight - weight.mean(dim=0)) / weight.std(dim=0)
        weight_total = weight1 * 0.4 + weight2 * 0.6
        _,argsort = torch.sort(weight_total,descending=True,dim=0)
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
        return ((2 * wpcc(preds,y) * std_preds * std_y) / ((std_preds**2) + (std_y**2) + (preds.mean(dim=0,keepdim=True)- y.mean(dim=0,keepdim=True))**2)).mean()
    
    def wccc_ind(preds,y,weight):
        std_preds = preds.std(dim=0, keepdim=True)
        std_y = y.std(dim=0,keepdim=True)
        return ((2 * wpcc_ind(preds,y,weight) * std_preds * std_y) / ((std_preds**2) + (std_y**2) + (preds.mean(dim=0,keepdim=True).abs() - y.mean(dim=0,keepdim=True).abs())**2)).mean()
        
    def output(loss):
        return {
        'huber':huber,
        'mse': mse,
        'wmse':wmse,
        'pcc': pcc,
        'rank_pcc':rank_pcc,
        'mse_pcc':mse_pcc,
        'ccc':ccc,
        'wpcc':wpcc,
        'wccc':wccc,
        'wpcc_ind':wpcc_ind,
        'wccc_ind':wccc_ind,
    }[loss]

    return output(loss)

############################################
#网络区
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

########################
#模型区
########################

class PredictModel(nn.Module):
    def __init__(self,args,szs):
        super(PredictModel,self).__init__()
        self.args = args
        self.NN_layer = nn.Sequential(*self.get_NN_layers(args, szs))
    
    def forward(self,feature_all):
        NN_output = self.NN_layer(feature_all)
        return NN_output

    def get_NN_layers(self, args, nn_szs):
        layers = []

        for layer_i, (in_sz, out_sz) in enumerate(zip(nn_szs[:-1], nn_szs[1:])):
            layers.append(nn.Linear(in_sz, out_sz))
            layers.append(nn.SiLU(inplace=True))
            # if args.dropout > 0.0:
            #     layers.append(nn.Dropout(p=args.dropout))

        layers.append(nn.Linear(nn_szs[-1],1))  #用最后一层隐藏层的结果
        return layers

class Cross_Layer(nn.Module):  #DCN_v2
    def __init__(self,args,K):
        super(Cross_Layer,self).__init__()
        self.Cross = nn.Linear(K,K)
    
    def forward(self,input_data):
        input_raw,feature = input_data
        cross = self.Cross(feature)
        output = input_raw * cross + feature
        return (input_raw,output)
    
class Cross_Model(nn.Module):
    def __init__(self,args,K):
        super(Cross_Model,self).__init__()
        self.args = args
        n_layers = args.cross_num
        self.Cross_Layers = nn.Sequential(*self.get_cross_layers(n_layers,K))
    
    def forward(self,feature_all):
        input_raw = feature_all
        _,cross_output = self.Cross_Layers((input_raw,feature_all))
        return cross_output
         
    def get_cross_layers(self,n_layers,K):
        layers = []
        for i in range(n_layers):
            layers.append(Cross_Layer(self.args,K))
        return layers

class Net2(nn.Module):
    def __init__(self, args, factor_num):
        super().__init__()
        self.args=args
        input_size = factor_num+args.ml_num
        nn_szs = [input_size] + args.nn_szs
        self.input_size=input_size
        
        if args.cross_num>0:  #有可能不使用DCN，对部分模型复杂度过高
            self.Cross_model= Cross_Model(args,nn_szs[0])
        self.Layer_nn = nn.Sequential(*self.get_nn_layers(nn_szs[0]*2, nn_szs[1]))
        #self.Layer_bert = nn.Sequential(*self.get_bert_layers(args, nn_szs))
        self.Layer_batchnorm = nn.BatchNorm1d(1)
        
        nn_szs_new=nn_szs[1:]
        #nn_szs_new[0]=nn_szs[1]*2
        self.Deep_model= PredictModel(args,nn_szs_new)   
        #self.post_init()

    def get_nn_layers(self, in_sz, out_sz):
        layers = []
        
        layers.append(nn.Linear(in_sz, out_sz))
        layers.append(nn.SiLU(inplace=True))
        # if self.args.dropout > 0.0:
        #     layers.append(nn.Dropout(p=self.args.dropout))
            
        return layers
    
    def get_bert_layers(self, args, szs):
        layers = []
        out_sz = szs[1]
        for i in range(args.num_mhas):
            layers.append(BertLayer(BertConfig(
                num_attention_heads=args.num_heads,
                hidden_size=out_sz,
                attention_probs_dropout_prob=args.dropout,
                hidden_dropout_prob=args.dropout,
                intermediate_size=out_sz)))
        return layers

    def forward(self, raw_feature, industry_dummy):
        # raw_feature, industry_dummy = feature
        if self.args.period==93000:
            raw_feature=(raw_feature-0.5).clamp(-1,1)
        else:
            raw_feature=raw_feature.clamp(-1,2)
        
        device = raw_feature.device
        graph=torch.mm(industry_dummy,industry_dummy.T)-torch.eye(raw_feature.shape[0],device=device)
        graph_feature= torch.mm(graph/(graph.sum(axis=0)+1e-8),raw_feature)
        
        first_output=self.Cross_model(raw_feature) if self.args.cross_num>0 else raw_feature
        all_out=torch.hstack([first_output, raw_feature-graph_feature]) #减去同行业的
        nn_output=self.Layer_nn(all_out)  
        return self.Layer_batchnorm(self.Deep_model(nn_output))

class UMPLitModule(LightningModule):
    def __init__(self, args, factor_num):
        super().__init__()
        self.args = args
        self.model = Net2(args, factor_num)

    def forward(self, *args):
        return self.model(*args)
    
    def training_step(self, batch, batch_idx):
        
        def get_longshort_return(preds, ret):
            _, sort = preds.sort(dim=0, descending=True)
            return (ret[sort[:int(preds.size(0)/100), 0]].mean()-ret[sort[-int(preds.size(0)/100):, 0]].mean())/2
        
        loss_list = []
        rankic_list = []
        longshort_list =[]
        
        loss_fn = get_loss_fn(self.args.loss)
        loss_log_fn = get_loss_fn('pcc')
        
        for data in batch:
            tsdata,ret,ret_clip,stock_weight=data
            raw_feature, industry_dummy = tsdata
            times={'5m':1,'1m':2,'15s':5}[self.args.future_ret]  #在训练时调整量纲
            
            preds=self.forward(raw_feature, industry_dummy)#[valid_ind]
            loss = loss_fn(preds,ret_clip*times) if self.args.loss not in ['wpcc_ind','wccc_ind'] else loss_fn(preds,ret_clip*times,stock_weight)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss_list.append(loss)
            rankic_list.append(-loss_log_fn(preds, ret))  
            longshort_list.append(get_longshort_return(preds, ret))
        
        if len(loss_list) == 0:
            #返回一个梯度为0，值为0的tensor
            return None
        
        total_loss = sum(loss_list) / len(loss_list)
        self.log('train_loss', total_loss)
        
        self.log('train_rankic', sum(rankic_list) / len(rankic_list), on_epoch=False, on_step=True)
        self.log('train_ls', sum(longshort_list) /len(longshort_list), on_epoch=False, on_step=True)
        
        return total_loss  

    def _evaluate_step(self, batch, batch_idx, stage):  
        
        def get_longshort_return(preds, ret):
            _, sort = preds.sort(dim=0, descending=True)
            return (ret[sort[:int(preds.size(0)/100), 0]].mean()-ret[sort[-int(preds.size(0)/100):, 0]].mean())/2
        
        tsdata,ret,ret_clip,stock_weight=batch[0]
        raw_feature, industry_dummy = tsdata
        preds=self.forward(raw_feature, industry_dummy)#[valid_ind]
        loss_log_fn = get_loss_fn('pcc')
        
        rankic =-loss_log_fn(preds, ret)
        long_short = get_longshort_return(preds, ret)
        self.log(f'{stage}_rankic', rankic, prog_bar=True,sync_dist=True)
        self.log(f'{stage}_ls', long_short, prog_bar=True,sync_dist=True)
        return #[rankic, long_short]

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')
    
    # def validation_epoch_end(self,val_step_outputs):
    #     num_batch = len(val_step_outputs)
    #     self.log('val_rankic', sum([data[0] for data in val_step_outputs]) / num_batch, prog_bar=True,sync_dist=True)
    #     self.log('val_ls', sum([data[1] for data in val_step_outputs]) / num_batch, prog_bar=True,sync_dist=True)
        
    # def test_epoch_end(self,test_step_outputs):
    #     num_batch = len(test_step_outputs)
    #     self.log('test_rankic', sum([data[0] for data in test_step_outputs]) / num_batch, prog_bar=True,sync_dist=True)
    #     self.log('test_ls', sum([data[1] for data in test_step_outputs]) / num_batch, prog_bar=True,sync_dist=True)
        
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
        if self.args.lr_scheduler is not None:
            optim_config['lr_scheduler'] = {
                'step_lr': torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.args.lr_stepsz, gamma=self.args.lr_gamma),
            }[self.args.lr_scheduler]

        return optim_config

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_ls', mode='max', save_last=False, save_top_k=10,
                            filename='{epoch}-{val_rankic:.4f}-{val_ls:.4f}'),
        ]

        return callbacks