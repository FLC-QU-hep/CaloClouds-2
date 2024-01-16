# using a single thread
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    # to run numpy single threaded

import torch
import time

from configs import Configs

import utils.gen_utils as gen_utils

from models.shower_flow import compile_HybridTanH_model
import models.CaloClouds_2 as mdls
import models.CaloClouds_1 as mdls2


########################
### PARAMS #############
########################

caloclouds = 'cm'   # 'ddpm, 'edm', 'cm'

cfg = Configs()
cfg.device = 'cpu'  # 'cuda' or 'cpu'
#use single thread
torch.set_num_threads(1)   # also comment out os.environ['OPENBLAS_NUM_THREADS'] = '1' above for multi threaded

# min and max energy of the generated events
min_e = 10
max_e = 90

num = 2000 # total number of generated events

bs = 1 # batch size   # optimized: bs=64(cm), 64(edm), 64(ddpm) for GPU, bs=1 for CPU (single-threaded)

iterations = 10 # number of iterations for timing

cfg.num_steps = 13
cfg.sampler = 'heun'   # default 'heun'
cfg.s_churn =  0.0     # stochasticity, default 0.0  (if s_churn more than num_steps, it will be clamped to max value)
cfg.s_noise = 1.0    # default 1.0   # noise added when s_churn > 0
cfg.sigma_max = 80.0 #  5.3152e+00  # default 80.0
cfg.sigma_min = 0.002   # default 0.002
cfg.rho = 7. # default 7.0

n_scaling = True


########################
########################
########################


def main(cfg, min_e, max_e, num, bs, iterations):

    num_blocks = 10
    flow, distribution = compile_HybridTanH_model(num_blocks, 
                                            num_inputs=65, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 
                                            num_cond_inputs=1, device=cfg.device)  # num_cond_inputs
    checkpoint = torch.load('BEEGFS/6_PointCloudDiffusion/shower_flow/220714_cog_e_layer_ShowerFlow_best.pth', map_location=torch.device(cfg.device))   # trained about 350 epochs
    
    flow.load_state_dict(checkpoint['model'])
    flow.eval().to(cfg.device)


    if caloclouds == 'ddpm':
        kdiffusion=False   # EDM vs DDPM diffusion
        # caloclouds baseline
        cfg.sched_mode = 'quardatic'
        cfg.num_steps = 100
        cfg.residual = True
        cfg.latent_dim = 256
        model = mdls2.CaloClouds_1(cfg).to(cfg.device)
        checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_quardatic2023_04_06__16_34_39/ckpt_0.000000_837000.pt', map_location=torch.device(cfg.device)) # quadratic
        model.load_state_dict(checkpoint['state_dict'])
        coef_real = np.array([ 2.42091454e-09, -2.72191705e-05,  2.95613817e-01,  4.88328360e+01])   # fixed coeff at 0.1 threshold
        coef_fake = np.array([-2.03879741e-06,  4.93529413e-03,  5.11518795e-01,  3.14176987e+02])

    elif caloclouds == 'edm':
        kdiffusion=True   # EDM vs DDPM diffusion
        # caloclouds EDM
        # # # baseline with lat_dim = 0, max_iter 10M, lr=1e-4 fixed, dropout_rate=0.0, ema_power=2/3 (long training)            USING THIS TRAINING
        cfg.dropout_rate = 0.0
        cfg.latent_dim = 0
        cfg.residual = False
        checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_29__23_08_31/ckpt_0.000000_2000000.pt', map_location=torch.device(cfg.device))    # max 5200000
        model = mdls.CaloClouds_2(cfg).to(cfg.device)
        model.load_state_dict(checkpoint['others']['model_ema'])
        coef_real = np.array([ 2.42091454e-09, -2.72191705e-05,  2.95613817e-01,  4.88328360e+01])  # fixed coeff at 0.1 threshold
        coef_fake = np.array([-7.68614180e-07,  2.49613388e-03,  1.00790407e+00,  1.63126644e+02])

    elif caloclouds == 'cm':
        kdiffusion=True   # EDM vs DDPM diffusion
        # condsistency model
        # long baseline with lat_dim = 0, max_iter 1M, lr=1e-4 fixed, num_steps=18, bs=256, simga_max=80, epoch=2M, EMA
        cfg.dropout_rate = 0.0
        cfg.latent_dim = 0
        cfg.residual = False
        checkpoint = torch.load(cfg.logdir + '/' + 'CD_2023_07_07__16_32_09/ckpt_0.000000_1000000.pt', map_location=torch.device(cfg.device))   # max 1200000
        model = mdls.CaloClouds_2(cfg, distillation = True).to(cfg.device)
        model.load_state_dict(checkpoint['others']['model_ema'])
        coef_real = np.array([ 2.42091454e-09, -2.72191705e-05,  2.95613817e-01,  4.88328360e+01])  # fixed coeff at 0.1 threshold
        coef_fake = np.array([-9.02997505e-07,  2.82747963e-03,  1.01417267e+00,  1.64829018e+02])  # compile_HybridTanH_model   w 10 blocks

    else:
        raise ValueError('caloclouds must be one of: ddpm, edm, cm')

    model.eval()

    torch.manual_seed(123)
    times_per_shower = []
    print('model used: ', caloclouds)
    for _ in range(iterations):

        s_t = time.time()
        fake_showers, cond_E = gen_utils.gen_showers_batch(model, distribution, min_e, max_e, num, bs, kdiffusion=kdiffusion, config=cfg, coef_real=coef_real, coef_fake=coef_fake, n_scaling=n_scaling)
        t = time.time() - s_t
        print(fake_showers.shape)
        print('total time [seconds]: ', t)
        print('time per shower [ms]: ', t / num * 1000)
        times_per_shower.append(t / num)

    print('mean time per shower [ms]: ', np.mean(times_per_shower) * 1000)
    print('std time per shower [ms]: ', np.std(times_per_shower) * 1000)




if __name__ == '__main__':
    main(cfg, min_e, max_e, num, bs, iterations)
