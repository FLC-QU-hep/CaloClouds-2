import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.weight_norm as weight_norm



######################################
### PERMUTATION EQUIVARIANT LAYER  ###
######################################

class EPiClayer_cond2(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim, cond_feats=1, sum_scale=1e-3):
        super().__init__()
        self.act = F.leaky_relu
        self.cond_feats = cond_feats
        self.fc_global1 = weight_norm(nn.Linear(hid_dim+hid_dim+latent_dim+cond_feats, hid_dim)) 
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim)) 
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim+latent_dim+cond_feats, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
        self.sum_scale = sum_scale

    def forward(self, x_global, x_local, cond_feats):   # shapes: x_global[b,latent], x_local[b,n,latent_local]  cond_feats[B,C]
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        x_pooled_mean = x_local.mean(1, keepdim=False)   # B,d
        x_pooled_sum = x_local.sum(1, keepdim=False) * self.sum_scale   # B,d
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global, cond_feats], 1)
        x_global1 = self.act(self.fc_global1(x_pooledCATglobal))  # new intermediate step
        x_global = self.act(self.fc_global2(x_global1) + x_global) # with residual connection before AF

        x_global2local = x_global.view(-1,1,latent_global).repeat(1,n_points,1) # first add dimension, than expand it
        cond_feats2local = cond_feats.view(-1,1,self.cond_feats).repeat(1,n_points,1)   # B,N,C
        x_localCATglobal = torch.cat([x_local, x_global2local, cond_feats2local], 2)
        x_local1 = self.act(self.fc_local1(x_localCATglobal))  # with residual connection before AF
        x_local = self.act(self.fc_local2(x_local1) + x_local)

        return x_global, x_local



#### ENCODER

class EPiC_encoder_cond(nn.Module):
    def __init__(self, zdim, input_dim=4, cond_features=1):
        super().__init__()
        self.hid_d = 128 #  args['hid_d']
        self.feats = input_dim
        self.equiv_layers = 3 #args['equiv_layers_discriminator']
        self.latent =  zdim # args['latent']    # used for latent size of equiv concat
        self.sum_scale = 1e-3
        self.cond_feats = cond_features


        self.fc_l1 = weight_norm(nn.Linear(self.feats+self.cond_feats, self.hid_d))
        self.fc_l2 = weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g1 = weight_norm(nn.Linear(self.hid_d+self.hid_d+self.cond_feats, self.hid_d))
        self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(EPiClayer_cond2(self.hid_d, self.hid_d, self.latent, cond_feats=self.cond_feats, sum_scale=self.sum_scale))
        
        self.fc_g3 = weight_norm(nn.Linear(self.hid_d+self.hid_d+self.latent+self.cond_feats, self.hid_d))
        self.fc_g4 = weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g_out_mean = weight_norm(nn.Linear(self.hid_d, self.latent))
        self.fc_g_out_logvar = weight_norm(nn.Linear(self.hid_d, self.latent))
        
    def forward(self, x, cond_feats):  
        '''
        Args:
            x: input point cloud, (B,N,d)
            cond_feats: context (B,C)
        '''
        batch_size, n_points, latent_local = x.size()
        cond_feats2local = cond_feats.view(-1,1,self.cond_feats).repeat(1,n_points,1)   # B,N,C

        # local encoding
        x_local = F.leaky_relu(self.fc_l1(   torch.cat([x, cond_feats2local], -1))    )
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features
        x_mean = x_local.mean(1, keepdim=False)  # mean over points dim.
        x_sum = x_local.sum(1, keepdim=False) * self.sum_scale  # mean over points dim.
        x_global = torch.cat([x_mean, x_sum, cond_feats], 1)   # B,H+H+C
        x_global = F.leaky_relu(self.fc_g1(x_global)) 
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            x_global, x_local = self.nn_list[i](x_global, x_local, cond_feats)   # contains residual connection
        
        x_mean = x_local.mean(1, keepdim=False)  # mean over points dim.
        x_sum = x_local.sum(1, keepdim=False) * self.sum_scale  # sum over points dim.
        x = torch.cat([x_mean, x_sum, x_global, cond_feats], 1)
        
        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)

        m = self.fc_g_out_mean(x)
        v = self.fc_g_out_logvar(x)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v