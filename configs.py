class Configs():
    
    def __init__(self):
        
    # Experiment Name for Comet Logging
        self.name = 'CD_'  # options: [TEST_, kCaloClouds_, CaloClouds_, CD_]
        self.comet_project = 'calo-consistency'   # options: ['k-CaloClouds', 'calo-consistency']
        self.Acomment = 'commment' 
        self.log_comet = False

    # Model arguments
        self.model_name = 'CaloClouds_2'             # choices=['CaloClouds_1', 'CaloClouds_2]
        self.latent_dim = 0     # caloclouds default: 256, caloclouds_2 default: 0
        self.beta_1 = 1e-4    # only for CC_1
        self.beta_T = 0.02     # only for CC_1
        self.sched_mode = 'quardatic'  # options: ['linear', 'quardatic', 'sigmoid]  # only for CC_1
        self.flexibility = 0.0   # only for CC_1
        self.truncate_std = 2.0  # only for CC_1
        self.latent_flow_depth = 14   # only for CC_1
        self.latent_flow_hidden_dim = 256    # only for CC_1
        self.num_samples = 4   # only for CC_1
        self.features = 4
        self.sample_num_points = 2048   # only for CC_1
        self.kl_weight = 1e-3   # default: 0.001 = 1e-3
        self.residual = False            # choices=[True, False]   # !! CC_1: True, CC_2: False
        
        self.cond_features = 2       # number of conditioning features (i.e. energy+points=2)
        self.norm_cond = True    # normalize conditioniong to [-1,1]
        self.kld_min = 1.0       # default: 1.0

        # EPiC arguments
        self.use_epic = False
        self.epic_layers = 5
        self.hid_d = 128        # default: 128
        self.sum_scale = 1e-3
        self.weight_norm = True

        # for n_flows model
        self.flow_model = 'PiecewiseRationalQuadraticCouplingTransform'
        self.flow_transforms = 10
        self.flow_layers = 2
        self.flow_hidden_dims = 128
        self.tails = 'linear'
        self.tail_bound = 10
        


    # Data
        self.dataset = 'x36_grid' # choices=['x36_grid', 'clustered', 'getting_high']
        self.dataset_path = 'BEEGFS/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'
        self.quantized_pos = False

    # Dataloader
        self.workers = 32
        self.train_bs = 128      # CaloClous_1: 256 / CaloClouds_2: 128 / CD: 256
        self.pin_memory = False         # choices=[True, False]
        self.shuffle = True             # choices=[True, False]
        self.max_points = 6_000
        

    # Optimizer and scheduler
        self.optimizer = 'RAdam'         # choices=['Adam', 'RAdam']  # CC_1: Adam, CC_2: RAdam
        self.lr = 1e-4              # CC_2 default: 1e-4 fixed,   
        self.end_lr = 1e-4
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.sched_start_epoch = 100 * 1e3
        self.sched_end_epoch = 400 * 1e3
        self.max_iters = 10 * 1e6

    # Others
        self.device = 'cuda'
        self.logdir = 'BEEGFS/6_PointCloudDiffusion/log'
        self.seed = 42
        self.val_freq =  10_000  #  1e3          # saving intervall for checkpoints

        self.test_freq = 30 * 1e3   
        self.test_size = 400
        self.tag = None
        self.log_iter = 100   # log every n iterations, default: 100

    # EMA scheduler
        self.ema_type = 'inverse'
        self.ema_power = 0.6667   # depends on the number of iterations, 2/3=0.6667 good for 1e6 iterations, 3/4=0.75 good for less
        self.ema_max_value = 0.9999
        
    # EDM diffusion parameters for training
        self.model = {
            "sigma_data" : 0.5,
            "sigma_sample_density" : {
                "type": "lognormal",
                "mean": -1.2,
                "std": 1.2
                }
            }
        self.dropout_mode = 'all'     # options: 'all',  'mid'  location of the droput layers
        self.dropout_rate = 0.0       # EDM: approx. 0.1, Caloclouds default: 0.0
        self.diffusion_loss = 'l2'    # l2 or l1

    # EDM diffusion parameters for sampling    / also used in CM distillation
        self.num_steps = 18      # EDM paper: 18
        self.sampler = 'heun'
        self.sigma_min = 0.002  # EDM paper: 0.002, k-diffusion config: 0.01
        self.sigma_max = 80.0
        self.rho = 7.0    # exponent in EDM boundaries
        self.s_churn = 0.0
        self.s_noise = 1.0


    # Consistency Distillation parameters   for the usage with cd.py
        self.model_path = 'kCaloClouds_2023_07_02__20_30_03/ckpt_0.000000_2000000.pt'
        self.use_ema_trainer = True
        self.start_ema = 0.95
        self.cm_random_init = False    # kinda like consistency training, but still with a teacher score function

    