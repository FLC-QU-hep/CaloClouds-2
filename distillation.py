from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.dataset import *
from utils.misc import *
from models.vae_flow import *
from models.CaloClouds_2 import CaloClouds_2
from configs import Configs


def main():
    cfg = Configs()
    seed_all(seed = cfg.seed)
    start_time = time.localtime()

    # Comet online logging
    if cfg.log_comet:
        experiment = Experiment(
            project_name=cfg.comet_project, auto_metric_logging=False,
        )
        experiment.log_parameters(cfg.__dict__)
        experiment.set_name(cfg.name+time.strftime('%Y_%m_%d__%H_%M_%S', start_time))
    else:
        experiment = None

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
    elif cfg.dataset == 'gettig_high':
        train_dset = PointCloudDatasetGH(
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

    # Model
    model = CaloClouds_2(cfg, distillation = True).to(cfg.device)
    model_ema_target = CaloClouds_2(cfg, distillation = True).to(cfg.device)
    model_teacher = CaloClouds_2(cfg, distillation = False).to(cfg.device)

    # load model
    checkpoint = torch.load(cfg.logdir + '/' + cfg.model_path)
    if cfg.use_ema_trainer:
        model.load_state_dict(checkpoint['others']['model_ema'])
        model_ema_target.load_state_dict(checkpoint['others']['model_ema'])
        model_teacher.load_state_dict(checkpoint['others']['model_ema'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
        model_ema_target.load_state_dict(checkpoint['state_dict'])
        model_teacher.load_state_dict(checkpoint['state_dict'])
    print('Model loaded from: ', cfg.logdir + '/' + cfg.model_path)

    if cfg.cm_random_init:
        print('randomly initializing diffusion parameters for online and ema (target) model')
        random_model = CaloClouds_2(cfg, distillation = True).to(cfg.device)
        model.diffusion.load_state_dict(random_model.diffusion.state_dict())
        model_ema_target.diffusion.load_state_dict(random_model.diffusion.state_dict())
        del random_model

    # set model status
    model.diffusion.requires_grad_(True)  # student ("online") model which is actually trained
    model.diffusion.train()
    if cfg.latent_dim > 0:
        model.encoder.requires_grad_(False)   # encoder is not trained
        model.encoder.eval()
    model_ema_target.requires_grad_(False) # target model for sampling from consistency model, updated as EMA of student model
    model_ema_target.train() # traget model needs to be in same state as online model, but does not require gradients
    model_teacher.requires_grad_(False) # teacher model used as score function in ODE solver
    model_teacher.eval()

    # Optimizer
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                [
                {'params': model.diffusion.parameters()},
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
    elif cfg.optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                [
                {'params': model.diffusion.parameters()},
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
    else: 
        raise NotImplementedError('Optimizer not implemented')
    print('optimizer used: ', cfg.optimizer, 'only diffusion parameters are optimized')
    print('no learning rate scheduler implemented')


    # Train
    def train(batch, it):
        # Load data
        x = batch['event'][0].float().to(cfg.device) # B, N, 4
        e = batch['energy'][0].float().to(cfg.device) # B, 1
        n = batch['points'][0].float().to(cfg.device) # B, 1
        # Reset grad
        optimizer.zero_grad()
        model.zero_grad()

        # get condition features
        if cfg.norm_cond:
            e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
            n = n / cfg.max_points * 2  - 1
        cond_feats = torch.cat([e,n], -1) # B, 2

        loss = model.get_cd_loss(x, cond_feats, model_teacher, model_ema_target, cfg)

        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.diffusion.parameters(), cfg.max_grad_norm)
        optimizer.step()

        # Update EMA target model  (necessary for CD, not the same as an EMA decay of the online model itself)
        # update only diffusion parameters
        mu = cfg.start_ema
        for targ, src in zip(model_ema_target.diffusion.parameters(), model.diffusion.parameters()):
            targ.detach().mul_(mu).add_(src, alpha=1 - mu)

        ## TODO also add EMA model of online model with lower decay rate, i.e. 0.9999 
        # (might perfrom better than target model or last online model for sampling?)

        # Logging
        if it % cfg.log_iter == 0:
            print('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
                it, loss.item(), orig_grad_norm, cfg.kl_weight
            ))
            if cfg.log_comet:
                experiment.log_metric('train/loss', loss, it)
                experiment.log_metric('train/kl_weight', cfg.kl_weight, it)
                experiment.log_metric('train/lr', optimizer.param_groups[0]['lr'], it)
                experiment.log_metric('train/grad_norm', orig_grad_norm, it)


    ## training loop
    # Main loop
    print('Start training...')

    it = 1
    start_time = time.time()
    st = time.time()
    while(it <= cfg.max_iters):
        for batch in dataloader:
            it += 1
            train(batch, it)
            if it % cfg.log_iter == 0:
                print('Time for %d iterations: %.2f' % (cfg.log_iter, time.time() - st))
                st = time.time()
            if it % cfg.val_freq == 0 or it == cfg.max_iters:
                opt_states = {
                    'model_ema': model_ema_target.state_dict(), # save the EMA model
                    'model_teacher': model_teacher.state_dict(), # save the teacher model
                    'optimizer': optimizer.state_dict(),
                }
                ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)

    print('training done in %.2f seconds' % (time.time() - start_time))




if __name__ == "__main__":
    main()