import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from .encoders.epic_encoder_cond import EPiC_encoder_cond
from utils.misc import *


class CaloClouds_1(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = EPiC_encoder_cond(args.latent_dim, input_dim=args.features, cond_features=args.cond_features)
        self.flow = get_flow_model(args)
        if args.use_epic:
            self.diffusion = DiffusionPoint(
                net = PointwiseNet_epic_res_wn(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual),
                var_sched = VarianceSchedule(
                    num_steps=args.num_steps,
                    beta_1=args.beta_1,
                    beta_T=args.beta_T,
                    mode=args.sched_mode
                ))
        else:
            self.diffusion = DiffusionPoint(
                net = PointwiseNet(point_dim=args.features, context_dim=args.latent_dim+args.cond_features, residual=args.residual),
                var_sched = VarianceSchedule(
                    num_steps=args.num_steps,
                    beta_1=args.beta_1,
                    beta_T=args.beta_T,
                    mode=args.sched_mode
                ))
            
        self.kld = KLDloss()

    def get_loss(self, x, cond_feats, kl_weight, writer=None, it=None, kld_min=0.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            cond_feats: conditioning features (B,C)
        """
        # batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x, cond_feats)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # H[Q(z|X)]
        # entropy = gaussian_entropy(logvar=z_sigma)      # (B, )     encoder loss, but why not KLD incl. z_mu?
        loss_kld = self.kld(mu=z_mu, logvar=z_sigma)

        # P(z), Prior probability, parameterized by the flow: z -> w.
        nll = - self.flow.log_prob(z.detach().clone(), cond_feats)    # detach from computational graph if optimizing encoder+diffuison seperate from flow
        # w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        # log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        # log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)        # flow loss, Logliklihood

        # Negative ELBO of P(X|z)
        z = torch.cat([z, cond_feats], -1)
        neg_elbo = self.diffusion.get_loss(x, z)    # diffusion loss

        # Loss
        # loss_entropy = -entropy.mean()
        loss_prior = nll.mean()
        loss_recons = neg_elbo

        # print('loss_kld before clamp: ', loss_kld.item())
        loss_kld_clamped = torch.clamp(loss_kld, min=kld_min)

        loss = kl_weight*(loss_kld_clamped) + neg_elbo
        # print(loss_kld_clamped.item(), loss_recons.item())

        if writer is not None:
            # writer.log_metric('train/loss_e', loss_entropy, it)
            writer.log_metric('train/loss_kld', loss_kld, it)
            writer.log_metric('train/loss_kld_clamped', loss_kld_clamped, it)
            writer.log_metric('train/loss_prior', loss_prior, it)
            writer.log_metric('train/loss_recons', loss_recons, it)
            writer.log_metric('train/z_mean', z_mu.mean(), it)
            writer.log_metric('train/z_mag', z_mu.abs().max(), it)
            writer.log_metric('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss, loss_prior


    def sample(self, cond_feats, num_points, flexibility):
        batch_size, _ = cond_feats.size()
        z = self.flow.sample(context=cond_feats, num_samples=1).view(batch_size, -1)  # B,F
        z = torch.cat([z, cond_feats], -1)   # B, F+C
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples
    
    def inference(self, num_points, cfg, num_samples=256):
        with torch.no_grad():
            z = torch.randn([num_samples, cfg.latent_dim]).to(cfg.device)
            x = self.sample(z, num_points, flexibility=cfg.flexibility)
        return x

    def sample_fromData(self, x, cond_feats, num_points, flexibility):
        z_mu, z_sigma = self.encoder(x, cond_feats)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        z = torch.cat([z, cond_feats], -1)   # B,F+C
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples



        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        # orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        # optimizer.step()
        # scheduler.step()
    
        # optimizer_flow.zero_grad()
        # loss_prior.backward()
        # orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        # optimizer_flow.step()
        # scheduler_flow.step()



class PointwiseNet_epic_res_wn(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashEPiCres_wn(point_dim, 128, context_dim+3+context_dim+3),     # gated learned sum of point vector and context vector
            ConcatSquashEPiCres_wn(128, 256, context_dim+3+context_dim+3),
            ConcatSquashEPiCres_wn(256, 512, context_dim+3+context_dim+3),
            ConcatSquashEPiCres_wn(512, 256, context_dim+3+context_dim+3),
            ConcatSquashEPiCres_wn(256, 128, context_dim+3+context_dim+3),
            ConcatSquashEPiCres_wn(128, point_dim, context_dim+3+context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        out = x
        ctx_emb_in = ctx_emb.clone()
        for i, layer in enumerate(self.layers):
            # out = layer(ctx=ctx_emb, x=out)
            ctx_emb = torch.cat([ctx_emb_in, ctx_emb], -1)
            ctx_emb, out = layer(ctx=ctx_emb, x=out)
            # ctx_emb = ctx_emb + ctx_emb_in
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out