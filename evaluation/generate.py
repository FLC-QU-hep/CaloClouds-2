import sys
sys.path.append('../')

import numpy as np
import torch
import time
import sys
import joblib

from models.vae_flow import *
from models.shower_flow import compile_HybridTanH_model
from configs import Configs
import utils.gen_utils as gen_utils

import models.CaloClouds_2 as mdls
import models.CaloClouds_1 as mdls2

import k_diffusion as K

cfg = Configs()

# print(cfg.__dict__)

###############################################  PARAMS

## SINGLE ENERGY GENERATION
# min_energy_list = [10, 50, 90]
# max_energy_list = [10, 50, 90]
# n_events = 2000
# out_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/singleE/'

### FULL SPECTRUM GENERATION
min_energy_list = [10]
max_energy_list = [90]
n_events = 40_000
out_path = 'BEEGFS/6_PointCloudDiffusion/output/full/'


### COMMMONG PARAMETERS
caloclouds_list = ['ddpm', 'edm', 'cm']   # 'ddpm, 'edm', 'cm'
seed_list = [12345, 123456, 1234567]
# caloclouds_list = ['cm']   # 'ddpm, 'edm', 'cm'
# seed_list = [1234]
n_scaling = True     # default True
batch_size = 16
prefix = ''   # default ''

###############################################

for i in range(len(caloclouds_list)):
    caloclouds = caloclouds_list[i]
    seed = seed_list[i]

    for j in range(len(min_energy_list)):
        min_energy = min_energy_list[j]
        max_energy = max_energy_list[j]

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
        if caloclouds == 'ddpm':
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
        elif caloclouds == 'edm':
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
        elif caloclouds == 'cm':
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

        else:
            raise ValueError('caloclouds must be one of: ddpm, edm, cm')

        model.eval()
        print(caloclouds, ' model loaded')


        ### GENERATE EVENTS
        torch.manual_seed(seed)
        print(' one random torch number: ', torch.rand(1))

        s_t = time.time()
        fake_showers, cond_E = gen_utils.gen_showers_batch(model, distribution, min_energy, max_energy, n_events, bs=batch_size, kdiffusion=kdiffusion, config=cfg, coef_real=coef_real, coef_fake=coef_fake, n_scaling=n_scaling, n_splines=n_splines)
        t = time.time() - s_t
        print(fake_showers.shape)
        print(t)
        print('time per shower: (s)', t / n_events)
        print(fake_showers[0,0,0])


        #### save fake showers
        f = out_path + prefix + '{}-{}GeV_{}_{}_seed{}.npz'.format(str(min_energy), str(max_energy), str(n_events), caloclouds, str(seed))
        np.savez(f, fake_showers=fake_showers, energy=cond_E)

        print('fake showers (energy in MeV) saved in ', f)