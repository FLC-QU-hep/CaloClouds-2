from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.dataset import PointCloudDataset
from utils.misc import seed_all, get_new_log_dir, CheckpointManager
from models.common import get_linear_scheduler
from models.CaloClouds_1 import CaloClouds_1
from models.CaloClouds_2 import CaloClouds_2
from configs import Configs

import k_diffusion as K

import time

cfg = Configs()
seed_all(seed = cfg.seed)
start_time = time.localtime()

if cfg.log_comet:
    experiment = Experiment(
        project_name=cfg.comet_project, auto_metric_logging=False,
    )
    experiment.log_parameters(cfg.__dict__)
    experiment.set_name(cfg.name+time.strftime('%Y_%m_%d__%H_%M_%S', start_time))

# Logging
log_dir = get_new_log_dir(cfg.logdir, prefix=cfg.name, postfix='_' + cfg.tag if cfg.tag is not None else '', start_time=start_time)
ckpt_mgr = CheckpointManager(log_dir)

# Datasets and loaders
if cfg.dataset == 'x36_grid' or cfg.dataset ==  'clustered':
    train_dset = PointCloudDataset(
        file_path=cfg.dataset_path,
        bs=cfg.train_bs,
        quantized_pos=cfg.quantized_pos
    )
dataloader = DataLoader(
    train_dset,
    batch_size=1,
    num_workers=cfg.workers,
    shuffle=cfg.shuffle
)
val_dset = []

# Model
if cfg.model_name == 'CaloClouds_1':
    model = CaloClouds_1(cfg).to(cfg.device)
    model_ema = CaloClouds_1(cfg).to(cfg.device)
elif cfg.model_name == 'CaloClouds_2':
    model = CaloClouds_2(cfg).to(cfg.device)
    model_ema = CaloClouds_2(cfg).to(cfg.device)

# initiate EMA (exponential moving average) model
model_ema.load_state_dict(model.state_dict())
model_ema.eval().requires_grad_(False)
assert cfg.ema_type == 'inverse'
ema_sched = K.utils.EMAWarmup(power=cfg.ema_power,
                                max_value=cfg.ema_max_value)

# Sigma (time step) distibution --> lognormal distribution, so minimum value is 0
sample_density = K.config.make_sample_density(cfg.__dict__["model"])

# Optimizer and scheduler
if cfg.optimizer == 'Adam':

    if cfg.latent_dim > 0:
        optimizer = torch.optim.Adam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                [
                {'params': model.encoder.parameters()}, 
                {'params': model.diffusion.parameters()},
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
        optimizer_flow = torch.optim.Adam(
            [
            {'params': model.flow.parameters()}, 
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
            [
            {'params': model.diffusion.parameters()},
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
elif cfg.optimizer == 'RAdam':
    if cfg.latent_dim > 0:
        optimizer = torch.optim.RAdam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                [
                {'params': model.encoder.parameters()}, 
                {'params': model.diffusion.parameters()},
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
        optimizer_flow = torch.optim.RAdam(
            [
            {'params': model.flow.parameters()}, 
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.RAdam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
            [
            {'params': model.diffusion.parameters()},
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
else: 
    raise NotImplementedError('Optimizer not implemented')

scheduler = get_linear_scheduler(optimizer, start_epoch=cfg.sched_start_epoch, end_epoch=cfg.sched_end_epoch, start_lr=cfg.lr, end_lr=cfg.end_lr)
if cfg.latent_dim > 0:
    scheduler_flow = get_linear_scheduler(optimizer_flow, start_epoch=cfg.sched_start_epoch, end_epoch=cfg.sched_end_epoch, start_lr=cfg.lr, end_lr=cfg.end_lr)


# Train, validate and test
def train(batch, it):
    # Load data
    x = batch['event'][0].float().to(cfg.device) # B, N, 4
    e = batch['energy'][0].float().to(cfg.device) # B, 1
    n = batch['points'][0].float().to(cfg.device) # B, 1
    # Reset grad and model state
    optimizer.zero_grad()
    if cfg.model_name == 'CaloClouds_1' or cfg.model_name == 'CaloClouds_2':
        if cfg.latent_dim > 0:
            optimizer_flow.zero_grad()
    model.train()

    # Forward
    if cfg.model_name == 'flow':
        loss = model.get_loss(x, kl_weight=cfg.kl_weight, writer=experiment, it=it)
        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

    elif cfg.model_name == 'CaloClouds_1':
        if cfg.norm_cond:
            e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
            n = n / cfg.max_points * 2  - 1
        cond_feats = torch.cat([e,n], -1) # B, 2
        if cfg.log_comet:
            loss, loss_flow = model.get_loss(x, cond_feats, kl_weight=cfg.kl_weight, writer=experiment, it=it, kld_min=cfg.kld_min)
        else:
            loss, loss_flow = model.get_loss(x, cond_feats, kl_weight=cfg.kl_weight, writer=None, it=it, kld_min=cfg.kld_min)

    elif cfg.model_name == 'CaloClouds_2':
        if cfg.norm_cond:
            e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
            n = n / cfg.max_points * 2  - 1
        cond_feats = torch.cat([e,n], -1) # B, 2

        noise = torch.randn_like(x)    # noise for forward diffusion
        sigma = sample_density([x.shape[0]], device=x.device)  # time steps

        if cfg.log_comet:
            loss, loss_flow = model.get_loss(x, noise, sigma, cond_feats, kl_weight=cfg.kl_weight, writer=experiment, it=it, kld_min=cfg.kld_min)
        else:
            loss, loss_flow = model.get_loss(x, noise, sigma, cond_feats, kl_weight=cfg.kl_weight, writer=None, it=it, kld_min=cfg.kld_min)


     # Backward and optimize
        loss.backward()
        if cfg.latent_dim > 0:
            loss_flow.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if cfg.latent_dim > 0:
            optimizer_flow.step()
            scheduler_flow.step()

    # Update EMA model
        ema_decay = ema_sched.get_value()
        K.utils.ema_update(model, model_ema, ema_decay)
        ema_sched.step()

    if it % cfg.log_iter == 0:
        print('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f | EMAdecay %.4f' % (
            it, loss.item(), orig_grad_norm, cfg.kl_weight, ema_decay
        ))
        if cfg.log_comet:
            experiment.log_metric('train/loss', loss, it)
            experiment.log_metric('train/kl_weight', cfg.kl_weight, it)
            experiment.log_metric('train/lr', optimizer.param_groups[0]['lr'], it)
            experiment.log_metric('train/grad_norm', orig_grad_norm, it)
            experiment.log_metric('train/ema_decay', ema_decay, it)
            if cfg.latent_dim > 0:
                experiment.log_metric('train/loss_flow', loss_flow, it)
                experiment.log_metric('train/lr_flow', optimizer_flow.param_groups[0]['lr'], it)

# Main loop
print('Start training...')

stop = False
it = 1
start_time = time.time()
while not stop:
    for batch in dataloader:
        it += 1
        train(batch, it)
        if it % cfg.val_freq == 0 or it == cfg.max_iters:
            opt_states = {
                'model_ema': model_ema.state_dict(), # save the EMA model
                'ema_sched': ema_sched.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
        if it >= cfg.max_iters:
            stop = True
            break
print('training done in %.2f seconds' % (time.time() - start_time))
