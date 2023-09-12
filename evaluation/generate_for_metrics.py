import argparse
import sys
import gc
sys.path.append('../')

import numpy as np
import torch
import time
import sys
import joblib
import pickle
import h5py

from models.vae_flow import *
from models.shower_flow import compile_HybridTanH_model
from configs import Configs
import utils.gen_utils as gen_utils
from utils.plotting import get_projections, MAP, layer_bottom_pos
import utils.metrics as metrics
import utils.plotting as plotting

import models.CaloClouds_2 as mdls
import models.CaloClouds_1 as mdls2

import k_diffusion as K

cfg = Configs()

# print(cfg.__dict__)

###############################################  PARAMS


total_events = 500_000   # total events to process
n_events = 50_000    # in chunks of n_events

### FULL SPECTRUM GENERATION
min_energy= 10
max_energy = 90

pickle_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/metrics/'


### COMMMONG PARAMETERS
n_scaling = True     # default True
prefix = ''   # default ''
key_exceptions=['e_radial', 'e_layers', 'e_layers_distibution', 'occ_layers', 'e_radial_lists']  # not saved in dict

###############################################

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--caloclouds', '-cc', type=str, default='cm', help='caloclouds model to use, choose from: ddpm, edm, cm, or g4')

args = parser.parse_args()

###############################################


### load shower flow model
flow, distribution = compile_HybridTanH_model(num_blocks=10, 
                                        num_inputs=65, ### adding 30 e layers 
                                        num_cond_inputs=1, device=cfg.device)  # num_cond_inputs

checkpoint = torch.load('BEEGFS/6_PointCloudDiffusion/shower_flow/220714_cog_e_layer_ShowerFlow_best.pth')   # trained about 350 epochs
flow.load_state_dict(checkpoint['model'])
flow.eval().to(cfg.device)

print('flow model loaded')


# load diffusion model and weights

# caloclouds baseline
if args.caloclouds == 'ddpm':
    seed = 12345
    batch_size = 64
    # cfg = Configs()
    kdiffusion=False   # EDM vs DDPM diffusion
    cfg.sched_mode = 'quardatic'
    cfg.num_steps = 100
    cfg.residual = True
    cfg.latent_dim = 256
    cfg.dropout_rate = 0.0
    model = mdls2.CaloClouds_1(cfg).to(cfg.device)
    checkpoint = torch.load('BEEGFS/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_quardatic2023_04_06__16_34_39/ckpt_0.000000_837000.pt', map_location=torch.device(cfg.device)) # quadratic
    model.load_state_dict(checkpoint['state_dict'])
    coef_real = np.array([ 2.42091454e-09, -2.72191705e-05,  2.95613817e-01,  4.88328360e+01])   # fixed coeff at 0.1 threshold
    coef_fake = np.array([-2.03879741e-06,  4.93529413e-03,  5.11518795e-01,  3.14176987e+02])
    n_splines = None #joblib.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/n_spline/spline_ddpm.joblib')

# caloclouds EDM
elif args.caloclouds == 'edm':
    seed = 123456
    batch_size = 64
    # cfg = Configs()
    kdiffusion=True   # EDM vs DDPM diffusion
    cfg.num_steps = 13
    cfg.sampler = 'heun'   # default 'heun'
    cfg.s_churn =  0.0     # stochasticity, default 0.0  (if s_churn more than num_steps, it will be clamped to max value)
    cfg.s_noise = 1.0    # default 1.0   # noise added when s_churn > 0
    cfg.sigma_max = 80.0 #  5.3152e+00  # default 80.0
    cfg.sigma_min = 0.002   # default 0.002
    cfg.rho = 7. # default 7.0
    # # # baseline with lat_dim = 0, max_iter 10M, lr=1e-4 fixed, dropout_rate=0.0, ema_power=2/3 (long training)            USING THIS TRAINING
    cfg.dropout_rate = 0.0
    cfg.latent_dim = 0
    cfg.residual = False
    checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_29__23_08_31/ckpt_0.000000_2000000.pt', map_location=torch.device(cfg.device))    # max 5200000
    model = mdls.CaloClouds_2(cfg).to(cfg.device)
    model.load_state_dict(checkpoint['others']['model_ema'])
    coef_real = np.array([ 2.42091454e-09, -2.72191705e-05,  2.95613817e-01,  4.88328360e+01])  # fixed coeff at 0.1 threshold
    coef_fake = np.array([-7.68614180e-07,  2.49613388e-03,  1.00790407e+00,  1.63126644e+02])
    n_splines = None # joblib.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/n_spline/spline_edm.joblib')

# condsistency model
elif args.caloclouds == 'cm':
    seed = 1234567
    batch_size = 16
    # cfg = Configs()
    kdiffusion=True   # EDM vs DDPM diffusion
    cfg.num_steps = 1
    cfg.sigma_max = 80.0 #  5.3152e+00  # default 80.0
    # long baseline with lat_dim = 0, max_iter 1M, lr=1e-4 fixed, num_steps=18, bs=256, simga_max=80, epoch=2M, EMA
    cfg.dropout_rate = 0.0
    cfg.latent_dim = 0
    cfg.residual = False
    checkpoint = torch.load(cfg.logdir + '/' + 'CD_2023_07_07__16_32_09/ckpt_0.000000_1000000.pt', map_location=torch.device(cfg.device))   # max 1200000
    model = mdls.CaloClouds_2(cfg, distillation = True).to(cfg.device)
    model.load_state_dict(checkpoint['others']['model_ema'])
    coef_real = np.array([ 2.42091454e-09, -2.72191705e-05,  2.95613817e-01,  4.88328360e+01])  # fixed coeff at 0.1 threshold
    coef_fake = np.array([-9.02997505e-07,  2.82747963e-03,  1.01417267e+00,  1.64829018e+02])
    n_splines = None # joblib.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/n_spline/spline_cm.joblib')


elif args.caloclouds == 'g4':
    path = 'BEEGFS/data/calo-clouds/hdf5/all_steps/validation/10-90GeV_x36_grid_regular_712k.hdf5'
    all_events = h5py.File(path, 'r')['events']
    all_energy = h5py.File(path, 'r')['energy']

else:
    raise ValueError('caloclouds must be one of: ddpm, edm, cm or g4')

print(args.caloclouds, ' model loaded')

if args.caloclouds != 'g4':
    model.eval()
    torch.manual_seed(seed)
    print(' one random torch number: ', torch.rand(1))


### GENERATE EVENTS

merge_dict = {}
i = 0
for _ in range(int(total_events / n_events)):

    if args.caloclouds == 'g4':
        print('loading Geant4 showers')
        showers, cond_E = all_events[i:i+n_events], all_energy[i:i+n_events]
        print(showers.shape)
        showers[:, -1] = showers[:, -1] * 1000   # GeV to MeV
        i += n_events

    else:
        print('generating showers')
        s_t = time.time()
        showers, cond_E = gen_utils.gen_showers_batch(model, distribution, min_energy, max_energy, n_events, bs=batch_size, kdiffusion=kdiffusion, config=cfg, coef_real=coef_real, coef_fake=coef_fake, n_scaling=n_scaling, n_splines=n_splines)
        t = time.time() - s_t
        print(showers.shape)
        print(t)
        print('time per shower: (s)', t / n_events)

    print('projecting showers')
    events, clouds  = get_projections(showers, MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)

    print('get features and center of gravities')
    dict = plotting.get_features(events)
    dict['incident_energy'] = cond_E.reshape(-1)  # GeV  shape: (n_events,)
    cog_list = plotting.get_cog(clouds)
    dict['cog_x'] = cog_list[0]
    dict['cog_y'] = cog_list[1]
    dict['cog_z'] = cog_list[2]

    print('merging dicts')
    merge_dict = metrics.merge_dicts([merge_dict, dict], key_exceptions=key_exceptions)

    print('current shape of occupancy in merge_dict: ', merge_dict['occ'].shape)

    # save dictonary 
    with open(pickle_path+'merge_dict_{}-{}GeV_{}_{}.pickle'.format(str(min_energy), str(max_energy), str(total_events), args.caloclouds), 'wb') as f:
        pickle.dump(merge_dict, f)
    print('merge_dict saved in pickle file')

    del showers, events, dict
    gc.collect()



