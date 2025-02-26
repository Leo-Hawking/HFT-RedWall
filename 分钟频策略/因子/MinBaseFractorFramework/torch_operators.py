import logging
from typing import List
import torch
import torch.nn.functional as F

# todo max和min兼容空值
# todo 校对算子
# todo valid数据和fillna算子
# todo daily数据有空值 要不要加入数据检测
# todo 跨天的那一分钟要不要block掉

EPS = 1e-9
def protect_log(tensor):
    return torch.sign(tensor)*torch.log(1+torch.abs(tensor))

def tensor_fill_median(tensor,dim=0):
    mask = torch.isnan(tensor)
    # 计算每列的中值
    column_median = torch.nanmedian(tensor, dim=dim, keepdim=True).values
    # 填充中值到缺失值的位置
    filled_tensor = torch.where(mask, column_median, tensor)
    return filled_tensor

def tensor_fill_mean(tensor,dim=0):
    mask = torch.isnan(tensor)
    # 计算每列的中值
    column_mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    # 填充中值到缺失值的位置
    filled_tensor = torch.where(mask, column_mean, tensor)
    return filled_tensor

def winsorize(tensor,k=5,dim=0):
    def use_stddev(x: torch.Tensor, dim=0) -> torch.Tensor:
        x_demean = x - x.nanmean(dim=dim, keepdim=True)
        x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=dim))
        return x_std
    # 去极值
    down = tensor.nanmean(dim=dim,keepdim=True) - k * use_stddev(tensor,dim=dim)
    up = tensor.nanmean(dim=dim,keepdim=True) + k * use_stddev(tensor,dim=dim)
    tensor = torch.where(tensor<down,down,tensor)
    tensor = torch.where(tensor>up,up,tensor)
    return tensor

def get_nan_tensor(shape, device, dtype):
    return torch.full(shape, torch.nan, dtype=dtype, device=device)


def get_nan_tensor(shape, device, dtype):
    return torch.full(shape, torch.nan, dtype=dtype, device=device)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)


def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sub(x, y)


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mul(x, y)


def div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(y) > EPS, torch.divide(x, y), torch.nan)


def Max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


def Min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, y)


def Abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def neg(x: torch.Tensor) -> torch.Tensor:
    return torch.negative(x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x))


def log(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.log(torch.abs(x)), torch.nan)


def pow(x: torch.Tensor, e: float) -> torch.Tensor:
    if e <= 1:
        return torch.pow(x.abs(), e)
    else:
        return torch.pow(x, e)


def inv(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.divide(1., x), torch.nan)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.sigmoid(x)


def sign(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)


def sign_square(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.pow(torch.abs(x), 2)


# 返回x是否大于value的bool矩阵
def is_gt(x: torch.Tensor, value: any) -> torch.Tensor:
    return x > value


# 返回x是否小于value的bool矩阵
def is_lt(x: torch.Tensor, value: any) -> torch.Tensor:
    return x < value


# 取2个bool矩阵的交集
def bool_and2(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1 & x2


# 只保留x中cond为True的部分，其余置为空值
def filt_cond(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, x, torch.nan)


# 返回展开的滑动窗口 2D -> 3D
def _unfold(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(x.shape + (d,), device=x.device, dtype=x.dtype)
    res[d - 1:] = x.unfold(0, d, 1)
    return res


# 最后一个维度上求统计量
def _sum(x: torch.Tensor) -> torch.Tensor:
    return x.nansum(dim=-1)


def _mean(x: torch.Tensor) -> torch.Tensor:
    return x.nanmean(dim=-1)


def _stddev(x: torch.Tensor) -> torch.Tensor:
    x_demean = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1))
    return x_std


def _skew(x: torch.Tensor):
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))
    zscore = torch.pow(div(x_demeaned, x_std), 3)
    return _mean(zscore)


def _kurt(x: torch.Tensor):
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_var = torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True)
    x_4 = torch.pow(x_demeaned, 4)
    return div(_mean(x_4), torch.pow(_mean(x_var), 2)) - 3


def _qua(x: torch.Tensor, q: float):
    return div(x.nanquantile(q, dim=-1), x.nanmean(dim=-1))


# 数值归一化后的熵
def _ent(x: torch.Tensor) -> torch.Tensor:
    x_norm = div(x, x.nansum(dim=-1, keepdim=True))
    res = - (x_norm * torch.log(x_norm)).nansum(dim=-1)
    return res


# 数值归一化后的交叉熵
def _cross_ent(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    x_norm = div(x, x.nansum(dim=-1, keepdim=True))
    y_norm = div(y, y.nansum(dim=-1, keepdim=True))
    res = - (x_norm * torch.log(y_norm)).nansum(dim=-1)
    return res


def ts_ent(x: torch.Tensor, d: int) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    return _ent(x_unfold)


def ts_cross_ent(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    y_unfold = _unfold(y, d)
    return _cross_ent(x_unfold, y_unfold)


# 集中度
def _conc(x: torch.Tensor, q: any) -> torch.Tensor:
    if q <= 1:
        return div(_sum(torch.pow(x.abs(), q)), torch.pow(_sum(x.abs()), q))
    else:
        return div(_sum(torch.pow(x, q)), torch.pow(_sum(x), q))


def ts_conc(x: torch.Tensor, d: int, q: any) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    return _conc(x_unfold, q)


# 位移路程比
def ts_diff_abs_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    diff = ts_delta(x, 1)
    return div(ts_sum(diff, d), ts_sum(diff.abs(), d))


def ts_mean_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _mean(x_unfold)


def ts_stddev_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _stddev(x_unfold)


def ts_skew_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _skew(x_unfold)


def ts_kurt_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _stddev(x_unfold)


def ts_qua_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, q: float, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _qua(x_unfold, q)


def ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d:] = x[:-d]
    return res


def ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d:] = x[d:] - x[: -d]
    return res


def ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))

    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res[d - 1:] = numerator / denominator

    res[d - 1:][(x_std < EPS) | (y_std < EPS)] = torch.nan
    return res


def ts_rankcorr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = _nanrank(x.unfold(0, d, 1), pct=False)
    y_unfold = _nanrank(y.unfold(0, d, 1), pct=False)

    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))

    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res[d - 1:] = numerator / denominator

    res[d - 1:][(x_std < EPS) | (y_std < EPS)] = torch.nan
    return res


def ts_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_windows = x.unfold(0, d, 1)
    y_windows = y.unfold(0, d, 1)

    x_mean = x_windows.nanmean(dim=-1, keepdim=True)
    y_mean = y_windows.nanmean(dim=-1, keepdim=True)
    x_demeaned = x_windows - x_mean
    y_demeaned = y_windows - y_mean

    res[d - 1:] = (x_demeaned * y_demeaned).nansum(dim=-1) / d
    return res


def ts_autocorr(x: torch.Tensor, d: int, shift: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[shift:] = ts_corr(x[shift:], x[:-shift], d)
    return res


def ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    w = torch.arange(1, d + 1, dtype=torch.float32, device=x.device)
    w = w / w.sum()
    x_unfolded = x.unfold(dimension=0, size=d, step=1)
    w = w.view(1, 1, -1)
    res = torch.nansum(x_unfolded * w, dim=-1)
    result = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    result[d - 1:] = res
    return result


# 对最后一个维度排序 跳过空值 排名从1开始 支持将非valid_mask的元素置为空值 pct返回比例
def _nanrank(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    if pct:
        valid_counts = valid_mask.sum(dim=-1, keepdim=True)
        ranks = ranks / valid_counts
    ranks[~valid_mask] = torch.nan
    return ranks


def ts_rank(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanrank(x.unfold(0, d, 1), pct=pct)[..., -1]
    return res


# 求最后一个维度的最小值 跳过空值 支持将非valid_mask的元素置为空值
def _nanmin(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).max
    res = xc.min(dim=-1).values
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


# 求最后一个维度的最大值 跳过空值 支持将非valid_mask的元素置为空值
def _nanmax(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).min
    res = xc.max(dim=-1).values
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


def ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanmin(x.unfold(0, d, 1))
    return res


def ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanmax(x.unfold(0, d, 1))
    return res


# 求最后一个维度的最小值位置 跳过空值 支持将非valid_mask的元素置为空值
def _nanargmin(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).max
    res = xc.argmin(dim=-1).to(dtype=x.dtype)
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


# 求最后一个维度的最大值位置 跳过空值 支持将非valid_mask的元素置为空值
def _nanargmax(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).min
    res = xc.argmax(dim=-1).to(dtype=x.dtype)
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


def ts_argmin(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = d - 1 - _nanargmin(x.unfold(0, d, 1))
    if pct:
        res = (res + 1) / d
    return res


def ts_argmax(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = d - 1 - _nanargmax(x.unfold(0, d, 1))
    if pct:
        res = (res + 1) / d
    return res


def ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(dimension=0, size=d, step=1)
    res[d - 1:] = unfolded.nansum(dim=-1, keepdim=False)
    return res


def ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(dimension=0, size=d, step=1)
    res[d - 1:] = unfolded.nanmean(dim=-1)
    return res


def ts_quantile(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = torch.nanquantile(x.unfold(0, d, 1), q, dim=-1)
    return res


def ts_qua(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    return ts_quantile(x, q, d) / ts_mean(x, d)


def ts_return(x: torch.Tensor, d: int) -> torch.Tensor:
    numerator = ts_delta(x, d)
    denominator = ts_delay(x, d)
    res = div(numerator, denominator)
    return res


def ts_mean_return(x: torch.Tensor, d: int) -> torch.Tensor:
    return ts_mean(ts_return(x, 1), d)


def ts_product(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(0, d, 1)
    res[d - 1:] = fillna(unfolded, 1).prod(dim=-1)
    return res


def ts_stddev(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1))
    res[d - 1:] = x_std
    return res


def ts_demean(x: torch.Tensor, d: int):
    return x - ts_mean(x, d)


# def ts_zscore(x: torch.Tensor, d: int):
#     x_demeaned = ts_demean(x, d)
#     x_std = ts_stddev(x, d)
#     return div(x_demeaned, x_std)


def ts_zscore(x: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    x_demeaned = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))
    zscore = div(x_demeaned, x_std)
    return _mean(zscore)


# def ts_skew(x: torch.Tensor, d: int):
#     roll_zscore = torch.pow(ts_zscore(x, d), 3)
#     return ts_mean(roll_zscore, d)


def ts_skew(x: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    return _skew(x_unfold)


# def ts_kurt(x: torch.Tensor, d: int):
#     roll_zscore = torch.pow(ts_zscore(x, d), 4)
#     return ts_mean(roll_zscore, d) - 3


def ts_kurt(x: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    return _kurt(x_unfold)


def ts_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)
    y_mean = y_unfold.nanmean(dim=-1, keepdim=True)
    x_demean = x_unfold - x_mean
    y_demean = y_unfold - y_mean
    std = torch.nansum(torch.pow(x_demean, 2), dim=-1)
    res[d - 1:] = torch.where(std < EPS, torch.nan, torch.nansum(x_demean * y_demean, dim=-1) / std)
    return res


def ts_regres(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    beta = ts_regbeta(x, y, d)
    return y - beta * x


def ts_rankregbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = _nanrank(x.unfold(0, d, 1), pct=False)
    y_unfold = _nanrank(y.unfold(0, d, 1), pct=False)
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)
    y_mean = y_unfold.nanmean(dim=-1, keepdim=True)
    x_demean = x_unfold - x_mean
    y_demean = y_unfold - y_mean
    std = torch.nansum(torch.pow(x_demean, 2), dim=-1)
    res[d - 1:] = torch.where(std < EPS, torch.nan, torch.nansum(x_demean * y_demean, dim=-1) / std)
    return res


def ts_rankregres(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = _nanrank(x.unfold(0, d, 1), pct=False)
    y_unfold = _nanrank(y.unfold(0, d, 1), pct=False)
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)
    y_mean = y_unfold.nanmean(dim=-1, keepdim=True)
    x_demean = x_unfold - x_mean
    y_demean = y_unfold - y_mean
    std = torch.nansum(torch.pow(x_demean, 2), dim=-1)
    beta = torch.where(std < EPS, torch.nan, torch.nansum(x_demean * y_demean, dim=-1) / std)
    res[d - 1:] = y_unfold[..., -1] - beta * x_unfold[..., -1]
    return res


if __name__ == "__main__":
    import itertools
    logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.DEBUG)
    def check_func_info_match(func, input_features, d: None):
        repr = func.repr_format
        repr = repr.replace("%name", func.name)
        for idx, input_feature in enumerate(input_features):
            repr = repr.replace(f"%x{idx + 1}", input_feature)
        repr = repr.replace("%d", str(d))
        print(repr, end=": ")


    def check_func_info_work(func, d: None):
        err_info_ls = ["\n"]
        for dtype, device in itertools.product([torch.float32, torch.float64],
                                              [torch.device("cpu"), torch.device("cuda:0")]):

            if device == torch.device("cuda:0"):
                before_mem = torch.cuda.memory_allocated()
            # Check operator's function.
            test_data = [torch.rand([300, 200], dtype=dtype, device=device)
                         for i in range(func.arity)]
            test_data_size = test_data[0].element_size() * test_data[0].nelement()
            try:
                if "d" in func.params:
                    res = func(*test_data, d)
                else:
                    res = func(*test_data)
                if res.dtype != dtype:
                    err_info_ls.append(f"Type Error: Input: {dtype}, output: {res.dtype}\n")
                if res.device != device:
                    err_info_ls.append(f"Device Error: Input: {device}, output: {res.device}\n")
                if res.shape != test_data[0].shape:
                    err_info_ls.append(f"Shape Error: Input: {test_data[0].shape}, output: {res.shape}\n")
            except Exception as e:
                err_info_ls.append(e)

            if device == torch.device("cuda:0"):
                # Check GPU memory leak
                del test_data
                del res
                torch.cuda.synchronize()  # 等待所有CUDA核心完成当前任务
                torch.cuda.empty_cache()  # 清空未使用的缓存
                import gc
                gc.collect()
                # 记录计算后的显存
                after_mem = torch.cuda.memory_allocated()
                if before_mem != after_mem:
                    err_info_ls.append(f"Memory Error: Detect GPU memory leak. Input data size: {test_data_size}."
                                       f"Leak size: {after_mem-before_mem}\n")
        if len(err_info_ls) <= 1:
            print("Pass")
        else:
            print("".join(err_info_ls))

    # default_feature_list = ["open", "high", "low", "close", "volume"]
    # for key, val in torch_function_registry.items():
    #     check_func_info_match(func=val, input_features=default_feature_list[:val.arity], d=5)
    #     check_func_info_work(func=val, d=5)



