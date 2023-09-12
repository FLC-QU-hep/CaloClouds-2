import torch
import torch.nn as nn
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn.utils.weight_norm as weight_norm

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


class KLDloss(nn.Module):
    
    def __init__(self):
        super(KLDloss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        
    def forward(self, mu, logvar):
            B = logvar.size(0)
            KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/(B)
            return KLD
    


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
    
    
class ConcatSquashEPiC_wn(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, sum_scale=1e-3):
        super().__init__()
        self.act = F.leaky_relu
        self.sum_scale = sum_scale
        self._pool = weight_norm(Linear(int(2*dim_in+dim_ctx), dim_ctx))
        self._layer = weight_norm(Linear(dim_in, dim_out))
        self._hyper_bias = weight_norm(Linear(dim_ctx, dim_out, bias=False))
        self._hyper_gate = weight_norm(Linear(dim_ctx, dim_out))

    def forward(self, ctx, x):
        """
        Args:
            ctx: context vector (B,1,F)
            x: point cloud (B,N,d)
        """

        x_mean = x.mean(1, keepdim=True)                 # B,1,d
        x_sum = x.sum(1, keepdim=True) * self.sum_scale  # B,1,d
        ctx_pool = torch.cat([ctx, x_mean, x_sum], -1)   # B,1,d+d+F
        ctx_pool = self._pool(ctx_pool)  
        ctx_pool = self.act(ctx_pool)                      # B,1,F

        gate = torch.sigmoid(self._hyper_gate(ctx_pool))   # B,1,d
        bias = self._hyper_bias(ctx_pool)                  # B,1,d
        ret = self._layer(x) * gate + bias            # B,1,d
        # ret = self.act(ret)
        return ctx_pool, ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
