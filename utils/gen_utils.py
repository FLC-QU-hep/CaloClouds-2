import numpy as np
import torch
from tqdm import tqdm

from utils.plotting import layer_bottom_pos, cell_thickness, Xmax, Xmin, Zmax, Zmin



def get_cog(x,y,z,e):
    return np.sum((x * e), axis=1) / e.sum(axis=1), np.sum((y * e), axis=1) / e.sum(axis=1), np.sum((z * e), axis=1) / e.sum(axis=1)


def get_scale_factor(num_clusters, coef_real, coef_fake, n_splines):  # num_clusters: (bs, 1)
    
    if n_splines is not None:
        spline_real = n_splines['spline_real']
        spline_fake = n_splines['spline_fake']
        scale_factor = spline_fake.predict(spline_real.predict(num_clusters).reshape(-1,1)) .reshape(-1,1) / num_clusters
    else:
        poly_fn_real = np.poly1d(coef_real)
        poly_fn_fake = np.poly1d(coef_fake) 
        scale_factor = poly_fn_fake(poly_fn_real(num_clusters)) / num_clusters

    scale_factor = np.clip(scale_factor, 0., None)  # cannot be negative
    return scale_factor  # (bs, 1)


def get_shower(model, num_points, energy, cond_N, bs=1, kdiffusion=False, config=None):

    e = torch.ones((bs, 1), device=config.device) * energy
    n = torch.ones((bs, 1), device=config.device) * cond_N
    
    if config.norm_cond:
        e = e / 100 * 2 -1   # max incident energy: 100 GeV
        n = n / config.max_points * 2  - 1
    cond_feats = torch.cat([e, n], -1)
        
    with torch.no_grad():
        if kdiffusion:
            fake_shower = model.sample(cond_feats, num_points, config)
        else:
            fake_shower = model.sample(cond_feats, num_points, config.flexibility)
    
    return fake_shower




def gen_showers_batch(model, shower_flow, e_min, e_max, num=2000, bs=32, kdiffusion=False, config=None, max_points=6000, coef_real=None, coef_fake=None, n_scaling = True, n_splines=None):
    
    cond_E = torch.FloatTensor(num, 1).uniform_(e_min, e_max).to(config.device)   #  B,1
    # sort by energyies for better batching and faster inference
    mask = torch.argsort(cond_E.squeeze())
    cond_E = cond_E[mask]


 


    fake_showers_list = []
    
    for evt_id in tqdm(range(0, num, bs), disable=False):
        if (num - evt_id) < bs:
            bs = num - evt_id

        cond_E_batch = cond_E[evt_id : evt_id+bs]

        # sample from shower flow
        samples = shower_flow.condition(cond_E_batch/100).sample(torch.Size([bs, ])).cpu().numpy()

        # name samples
        num_clusters = np.clip((samples[:, 0] * 5000).reshape(bs, 1), 1, max_points)
        energies = np.clip((samples[:, 1] * 2.5 * 1000).reshape(bs, 1), 0.04*1000, None)   # in MeV  (clip to a minimum energy of 40 MeV)
        cog_x = samples[:, 2] * 25
        # cog_y = samples[:, 3] * 15 + 15
        cog_z = samples[:, 4] * 20 + 40
        clusters_per_layer_gen = np.clip(samples[:, 5:35], 0, 1)  #  B,30
        e_per_layer_gen = np.clip(samples[:, 35:], 0, 1)          #  B,30

        if n_scaling:
            scale_factor = get_scale_factor(num_clusters, coef_real, coef_fake, n_splines)   # B,1
            num_clusters = (num_clusters * scale_factor).astype(int)  # B,1
        else:
            num_clusters = (num_clusters).astype(int)  # B,1
        
        # scale relative clusters per layer to actual number of clusters per layer   and same for energy
        clusters_per_layer_gen = (clusters_per_layer_gen / clusters_per_layer_gen.sum(axis=1, keepdims=True) * num_clusters).astype(int) # B,30
        e_per_layer_gen = e_per_layer_gen / e_per_layer_gen.sum(axis=1, keepdims=True) * energies  # B,30            

        # convert clusters_per_layer_gen to a fractions of points in the layer out of sum(points in the layer) of event
        # multuply clusters_per_layer_gen by corrected tottal num of points
        hits_per_layer_all = clusters_per_layer_gen#[evt_id : evt_id+bs] # shape (bs, num_layers) 
        e_per_layer_all = e_per_layer_gen#[evt_id : evt_id+bs] # shape (bs, num_layers)
        max_num_clusters = hits_per_layer_all.sum(axis=1).max()
        cond_N = torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)


        fs = get_shower(model, max_num_clusters, cond_E_batch, cond_N, bs=bs, kdiffusion=kdiffusion, config=config)
        fs = fs.cpu().numpy()
        
        # loop over events
        y_positions = layer_bottom_pos+cell_thickness/2
        for i, hits_per_layer in enumerate(hits_per_layer_all):
    
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

            y_flow = np.repeat(y_positions, hits_per_layer)
            y_flow = np.concatenate([y_flow, np.zeros(n_hits_to_concat)])

            mask = np.concatenate([ np.ones( hits_per_layer.sum() ), np.zeros( n_hits_to_concat ) ])

            fs[i, :, 1][mask == 0] = 10
            idx_dm = np.argsort(fs[i, :, 1])
            fs[i, :, 1][idx_dm] = y_flow


            fs[i, :, :][y_flow==0] = 0  

            fs[fs[:, :, -1]  <= 0] = 0    # setting events with negative energy to zero

            # energy per layer calibration
            for j in range(len(y_positions)):
                mask = fs[i, :, 1] == y_positions[j]
                fs[i, :, -1][mask] = fs[i, :, -1][mask] / fs[i, :, -1][mask].sum() * e_per_layer_all[i, j]

        length = max_points - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)  # B, max_points, 4

        fs = np.moveaxis(fs, -1, -2)   # (bs, num_points, 4) -> (bs, 4, num_points)
        fs[:, 0, :] = (fs[:, 0, :] + 1) / 2
        fs[:, 2, :] = (fs[:, 2, :] + 1) / 2

        fs[:, 0] = fs[:, 0] * (Xmin-Xmax) + Xmax
        fs[:, 2] = fs[:, 2] * (Zmin-Zmax) + Zmax

        # CoG calibration
        cog = get_cog(fs[:, 0], fs[:, 1], fs[:, 2], fs[:, 3])
        fs[:, 0] -= (cog[0] - cog_x)[:,None]
        fs[:, 2] -= (cog[2] - cog_z)[:,None]

        fake_showers_list.append(fs)
        
    fake_showers = np.vstack(fake_showers_list)  # (bs, num_points, 4)
    
    return fake_showers, cond_E.detach().cpu().numpy()   # (bs, 4, num_points), (bs, 1)