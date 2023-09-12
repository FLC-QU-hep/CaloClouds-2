
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import utils.metrics as metrics

mpl.rcParams['xtick.labelsize'] = 25    
mpl.rcParams['ytick.labelsize'] = 25
# mpl.rcParams['font.size'] = 28
mpl.rcParams['font.size'] = 35
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['axes.aspect'] = 'equal' 
# mpl.rcParams['figure.autolayout'] = True # for the hit e distr.
# mpl.rcParams['font.weight'] = 1  # Or you can use a numerical value, e.g., 700


class Configs():
    
    def __init__(self):

    # legend font
        self.font = font_manager.FontProperties(
            family='serif',
            size=23
            # size=20
        )
        self.text_size = 20
        
    # radial profile
        self.bins_r = 35
        self.origin = (0, 40)
        # self.origin = (3.754597092*10, -3.611833191*10)
        

    # occupancy
        self.occup_bins = np.linspace(150, 1419, 50)
        self.plot_text_occupancy = False
        self.occ_indent = 20

    # e_sum
        self.e_sum_bins = np.linspace(20.01, 2200, 50)
        self.plot_text_e = False
        self.plot_legend_e = True
        self.e_indent = 20

    # hits
        self.hit_bins = np.logspace(np.log10(0.01000001), np.log10(1000), 70)
        # self.hit_bins = np.logspace(np.log10(0.01), np.log10(100), 70)
        self.ylim_hits = (1, 1*1e7)
        # self.ylim_hits = (10, 8*1e5)

    #CoG
        self.bins_cog = 30  
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        self.cog_ranges = [(-3.99+1.5, 3.99-1.5), (1861, 2001), (37.99, 39.99+1.99)]
        # self.cog_ranges = [(-1.7, 1.2), (1891, 1949), (38.5, 41.99)]
        # self.cog_ranges = [(-3.99, 3.99), (1861, 1999), (36.01, 43.99)]
        # self.cog_ranges = [(33.99, 39.99), (1861, 1999), (-38.9, -32.9)]

    # xyz featas
        self.bins_feats = 50  
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        self.feats_ranges = [(-200, 200), (1811, 2011), (-160, 240)]
        # self.cog_ranges = [(-3.99, 3.99), (1861, 1999), (36.01, 43.99)]
        # self.cog_ranges = [(33.99, 39.99), (1861, 1999), (-38.9, -32.9)]


    # all
        self.threshold = 0.1   # MeV / half a MIP
        #self.color_lines = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        self.color_lines = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
        # self.color_lines = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']

        self.include_artifacts = False   # include_artifacts = True -> keeps points that hit dead material

    # percentile edges for occupancy based metrics
        self.layer_edges = [0, 8, 11, 13, 15, 16, 18, 19, 21, 24, 29]
        self.radial_edges = [0, 6.558, 9.849, 12.96, 17.028, 23.434, 33.609, 40.119, 48.491, 68.808, 300]

cfg = Configs()

Ymin = 1811
Xmin = -200
Xmax = 200
# Xmin = -260
# Xmax = 340

Zmin = -160
Zmax = 240
# Zmin = -300
# Zmax = 300

half_cell_size = 5.0883331298828125/2
cell_thickness = 0.5250244140625

layer_bottom_pos = np.array([   1811.34020996, 1814.46508789, 1823.81005859, 1826.93505859,
                                    1836.2800293 , 1839.4050293 , 1848.75      , 1851.875     ,
                                    1861.2199707 , 1864.3449707 , 1873.68994141, 1876.81494141,
                                    1886.16003418, 1889.28503418, 1898.63000488, 1901.75500488,
                                    1911.09997559, 1914.22497559, 1923.56994629, 1926.69494629,
                                    1938.14001465, 1943.36499023, 1954.81005859, 1960.03503418,
                                    1971.47998047, 1976.70495605, 1988.15002441, 1993.375     ,
                                    2004.81994629, 2010.04504395])

X = np.load('BEEGFS/data/calo-clouds/muon-map/X.npy')
Z = np.load('BEEGFS/data/calo-clouds/muon-map/Z.npy')
Y = np.load('BEEGFS/data/calo-clouds/muon-map/Y.npy')
E = np.load('BEEGFS/data/calo-clouds/muon-map/E.npy')

inbox_idx = np.where((Y > Ymin) & (X < Xmax) & (X > Xmin) & (Z < Zmax) & (Z > Zmin) )[0]
# inbox_idx = np.where(Y > Ymin)[0]


X = X[inbox_idx]
Z = Z[inbox_idx]
Y = Y[inbox_idx]
E = E[inbox_idx]


def create_map(X, Y, Z, dm=3):
    """
        X, Y, Z: np.array 
            ILD coordinates of sencors hited with muons
        dm: int (1, 2, 3, 4, 5) dimension split multiplicity
    """

    offset = half_cell_size*2/(dm)

    layers = []
    for l in tqdm(range(len(layer_bottom_pos))): # loop over layers
        idx = np.where((Y <= (layer_bottom_pos[l] + cell_thickness*1.5)) & (Y >= layer_bottom_pos[l] - cell_thickness/2 ))
        
        xedges = np.array([])
        zedges = np.array([])
        
        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])
        
        xedges = np.append(xedges, unique_X[0] - half_cell_size)
        xedges = np.append(xedges, unique_X[0] + half_cell_size)
        
        for i in range(len(unique_X)-1): # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i+1]) > half_cell_size * 1.9:
                xedges = np.append(xedges, unique_X[i+1] - half_cell_size)
                xedges = np.append(xedges, unique_X[i+1] + half_cell_size)
                
                for of_m in range(dm):
                    xedges = np.append(xedges, unique_X[i+1] - half_cell_size + offset*of_m) # for higher granularity
                
        for z in unique_Z: # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size)
            zedges = np.append(zedges, z + half_cell_size)
            
            for of_m in range(dm):
                zedges = np.append(zedges, z - half_cell_size + offset*of_m) # for higher granularity
                
            
        zedges = np.unique(zedges)
        xedges = np.unique(xedges)
        
        xedges = [xedges[i] for i in range(len(xedges)-1) if abs(xedges[i] - xedges[i+1]) > 1e-3] + [xedges[-1]]
        zedges = [zedges[i] for i in range(len(zedges)-1) if abs(zedges[i] - zedges[i+1]) > 1e-3] + [zedges[-1]]
        
            
        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({'xedges': xedges, 'zedges': zedges, 'grid': H})

    return layers, offset




def get_projections(showers, MAP, layer_bottom_pos, return_cell_point_cloud=False, max_num_hits=6000):
    events = []
    for shower in tqdm(showers):
        layers = []
        
        x_coord, y_coord, z_coord, e_coord = shower

        for l in range(len(MAP)):
            idx = np.where((y_coord <= (layer_bottom_pos[l] + 1)) & (y_coord >= layer_bottom_pos[l] - 0.5 ))
            
            xedges = MAP[l]['xedges']
            zedges = MAP[l]['zedges']
            H_base = MAP[l]['grid']
            
            H, xedges, zedges = np.histogram2d(x_coord[idx], z_coord[idx], bins=(xedges, zedges), weights=e_coord[idx])
            if not cfg.include_artifacts:
                H[H_base==0] = 0
            
            layers.append(H)
        
        events.append(layers)
    
    if not return_cell_point_cloud:
        return events
    
    else:
        cell_point_clouds = []
        for event in tqdm(events):
            point_cloud = []
            for l, layer in enumerate(event):
                
                xedges = MAP[l]['xedges']
                zedges = MAP[l]['zedges']
                
                x_indx, z_indx = np.where(layer > 0)

                cell_energy = layer[layer > 0]
                cell_coordinate_x = xedges[x_indx] + half_cell_size
                cell_coordinate_y = np.repeat(layer_bottom_pos[l] + cell_thickness/2, len(x_indx))
                cell_coordinate_z = zedges[z_indx] + half_cell_size
                
                point_cloud.append(
                    [cell_coordinate_x, cell_coordinate_y, cell_coordinate_z, cell_energy]
                )
                
            point_cloud = np.concatenate(point_cloud, axis=1)
            zeros_to_concat = np.zeros((4, max_num_hits-len(point_cloud[0])))
            point_cloud = np.concatenate((point_cloud, zeros_to_concat), axis=1)
            
            
            cell_point_clouds.append([point_cloud])
            
        cell_point_clouds = np.vstack(cell_point_clouds)
        
        return events, cell_point_clouds




def get_cog(cloud, thr=cfg.threshold):  # expects shape [events, 4, points]
    cloud = cloud.copy()
    # set all pionts with energy below threshold to zero
    mask = cloud[:, -1, :] < thr
    mask = np.repeat(mask[:, np.newaxis, :], 4, axis=1)
    cloud[mask] = 0

    x, y, z, e = cloud[:, 0], cloud[:, 1], cloud[:, 2], cloud[:, 3],

    # mask out events with zero or negative e sum
    mask = e.sum(axis=1) > 0
    x, y, z, e = x[mask], y[mask], z[mask], e[mask]

    x_cog = np.sum((x * e), axis=1) / e.sum(axis=1)
    y_cog = np.sum((y * e), axis=1) / e.sum(axis=1)
    z_cog = np.sum((z * e), axis=1) / e.sum(axis=1)
    return x_cog, y_cog, z_cog


def get_features(events):
    
    incident_point = cfg.origin
    
    occ_list = [] # occupancy
    hits_list = [] # energy per cell
    e_sum_list = [] # energy per shower
    e_radial = [] # radial profile
    e_layers_list = [] # energy per layer
    occ_layers_list = [] # occupancy per layer
    e_radidal_lists = [] # radial profile per layer
    hits_noThreshold_list = [] # energy per cell without threshold
    dict = {}

    for layers in tqdm(events):

        occ = 0
        e_sum = 0
        e_layers = []
        occ_layers = []
        y_pos = []
        e_radial_layers = []
        for l, layer in enumerate(layers):
            # layer = layer*1000 # energy rescale to MeV
            layer = layer.copy()   # for following inplace operations
            layer_noThreshold = layer.copy()
            layer[layer < cfg.threshold] = 0

            hit_mask = layer > 0    # shape i.e. 82,81
            layer_hits = layer[hit_mask]
            layer_sum = layer.sum()

            occ += hit_mask.sum()
            e_sum += layer.sum()

            hits_list.append(layer_hits)
            hits_noThreshold_list.append(layer_noThreshold[layer_noThreshold > 0])
            e_layers.append(layer.sum())

            occ_layers.append(hit_mask.sum())

            # get radial profile #######################
            x_hit_idx, z_hit_idx = np.where(hit_mask)
            x_cell_coord = MAP[l]['xedges'][:-1][x_hit_idx] + half_cell_size
            z_cell_coord = MAP[l]['zedges'][:-1][z_hit_idx] + half_cell_size
            e_cell = layer[x_hit_idx, z_hit_idx]
            dist_to_origin = np.sqrt((x_cell_coord - incident_point[0])**2 + (z_cell_coord - incident_point[1])**2)
            e_radial.append([dist_to_origin, e_cell])
            e_radial_layers.append([dist_to_origin, e_cell])
            ############################################

        e_layers_list.append(e_layers)
        occ_layers_list.append(occ_layers)

        occ_list.append(occ)
        e_sum_list.append(e_sum)

        e_radial_layers = np.concatenate(e_radial_layers, axis=1)
        e_radidal_lists.append(e_radial_layers)

    dict['e_radial'] = np.concatenate(e_radial, axis=1)  # out shape: [2, flattend hits]
    dict['e_sum'] = np.array(e_sum_list)
    dict['hits'] = np.concatenate(hits_list)   # hit energies
    dict['occ'] = np.array(occ_list)
    dict['e_layers_distibution'] = np.array(e_layers_list)  # distibution of energy per layer
    dict['e_layers'] = np.array(e_layers_list).sum(axis=0)/len(events)  # average energy per layer
    dict['occ_layers'] = np.array(occ_layers_list)#.sum(axis=0)/len(events)
    dict['e_radial_lists'] = e_radidal_lists  # nested list: e_rad_lst[ EVENTS ][DIST,E ] [ POINTS ]
    dict['hits_noThreshold'] = np.concatenate(hits_noThreshold_list)  # hit energies without threshold

    # add binned layer and radial energy metrics
    dict['binned_layer_e'] = metrics.binned_layer_energy(dict['e_layers_distibution'], bin_edges=cfg.layer_edges)  # shape: [bin_centeres, events]
    dict['binned_radial_e'] = metrics.binned_radial_energy(dict['e_radial_lists'], bin_edges=cfg.radial_edges)           # shape: [bin_centeres, events]
    
    # return e_radial, occ_list, e_sum_list, hits_list, e_layers_list, occ_layers_list, e_layers_distibution, e_radial_lists, hits_noThreshold_list
    return dict


def plt_radial(e_radial, e_radial_list, labels, cfg=cfg, title=r'\textbf{full spectrum}', events=40_000):
    fig, axs = plt.subplots(2, 1, figsize=(7,9), height_ratios=[3, 1], sharex=True)

    ## for legend ##########################################
    axs[0].hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(e_radial_list)):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    axs[0].set_title(title, fontsize=cfg.font.get_size(), loc='right')
    # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    axs[0].legend(prop=cfg.font, loc='upper right')
    ########################################################

    h = axs[0].hist(e_radial[0], bins=cfg.bins_r, weights=e_radial[1]/events, color='lightgrey', rasterized=True)
    h = axs[0].hist(e_radial[0], bins=cfg.bins_r, weights=e_radial[1]/events, color='dimgrey', histtype='step', lw=2)
    
    for i, e_radial_ in enumerate(e_radial_list):
        h1 = axs[0].hist(e_radial_[0], bins=h[1], weights=e_radial_[1]/events, histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i])

        # ratio plot on the bottom
        lims_min = 0.5
        lims_max = 1.9
        eps = 1e-5
        centers = np.array((h[1][:-1] + h[1][1:])/2)
        ratios = np.clip(np.array((h1[0]+eps)/(h[0]+eps)), lims_min, lims_max)
        mask = (ratios > lims_min) & (ratios < lims_max)  # mask ratios within plotting y range
        # only connect dots with adjecent points
        starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0]
        ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1
        indexes = np.stack((starts,ends)).T
        for idxs in indexes:
            sub_mask = np.zeros(len(mask), dtype=bool)
            sub_mask[idxs[0]:idxs[1]] = True
            axs[1].plot(centers[sub_mask], ratios[sub_mask], linestyle=':', lw=2, marker='o', color=cfg.color_lines[i])
        # remaining points either above or below plotting y range
        mask = (ratios == lims_min)
        axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=cfg.color_lines[i], clip_on=False)
        mask = (ratios == lims_max)
        axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=cfg.color_lines[i], clip_on=False)

    axs[1].set_ylim(lims_min, lims_max)

    # horizontal line at 1
    axs[1].axhline(1, linestyle='-', lw=1, color='k')
        
    axs[0].set_ylim(1.1e-4,5e4)
    
    axs[0].set_yscale('log')
    
    plt.xlabel("radius [mm]")
    axs[0].set_ylabel('mean energy [MeV]')
    axs[1].set_ylabel('ratio to MC')
    
    plt.subplots_adjust(hspace=0.1)
    # plt.tight_layout()

    plt.savefig('radial.pdf', dpi=100, bbox_inches='tight')
    plt.show()
    
def plt_spinal(e_layers, e_layers_list, labels, cfg=cfg, title=r'\textbf{full spectrum}'):
    
    fig, axs = plt.subplots(2, 1, figsize=(7,9), height_ratios=[3, 1], sharex=True)

    ## for legend ##########################################
    axs[0].hist(np.zeros(1)-10, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(e_layers_list)):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    ########################################################

    pos = np.arange(1, len(e_layers)+1)
    bins = np.arange(0.5, len(e_layers)+1.5)

    axs[0].hist(pos, bins=bins, weights=e_layers, color='lightgrey', rasterized=True)
    axs[0].hist(pos, bins=bins, weights=e_layers, color='dimgrey', histtype='step', lw=2)
    
    for i, e_layers_ in enumerate(e_layers_list):
        axs[0].hist(pos, bins=bins, weights=e_layers_, histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i])

        # ratio plot on the bottom
        lims_min = 0.8
        lims_max = 1.05
        eps = 1e-5
        centers = pos
        ratios = np.clip((e_layers_+eps)/(e_layers+eps), lims_min, lims_max)
        mask = (ratios > lims_min) & (ratios < lims_max)  # mask ratios within plotting y range
        # only connect dots with adjecent points
        starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0]
        ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1
        indexes = np.stack((starts,ends)).T
        for idxs in indexes:
            sub_mask = np.zeros(len(mask), dtype=bool)
            sub_mask[idxs[0]:idxs[1]] = True
            axs[1].plot(centers[sub_mask], ratios[sub_mask], linestyle=':', lw=2, marker='o', color=cfg.color_lines[i])
        # remaining points either above or below plotting y range
        mask = (ratios == lims_min)
        axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=cfg.color_lines[i], clip_on=False)
        mask = (ratios == lims_max)
        axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=cfg.color_lines[i], clip_on=False)

    axs[1].set_ylim(lims_min, lims_max)

    # horizontal line at 1
    axs[1].axhline(1, linestyle='-', lw=1, color='k')

    axs[0].set_yscale('log')
    axs[0].set_ylim(1.1e-1, 2e2)
    axs[0].set_xlim(0, 31)
    plt.xlabel('layers')
    axs[0].set_ylabel('mean energy [MeV]')
    axs[1].set_ylabel('ratio to MC')
    
    # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    #plt.legend(prop=cfg.font, loc='best')
    axs[0].set_title(title, fontsize=cfg.font.get_size(), loc='right')
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    plt.savefig('spinal.pdf', dpi=100, bbox_inches='tight')
    plt.show()
    
def plt_occupancy(occ, occ_list, labels, cfg=cfg):
    plt.figure(figsize=(7,7))

    ## for legend ##########################################
    plt.hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(occ_list)):
        plt.plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    ########################################################

    h = plt.hist(occ, bins=cfg.occup_bins, color='lightgrey', rasterized=True)
    h = plt.hist(occ, bins=cfg.occup_bins, color='dimgrey', histtype='step', lw=2)
    
    for i, occ_ in enumerate(occ_list):
        plt.hist(occ_, bins=h[1], histtype='step', linestyle='-', lw=2.5, color=cfg.color_lines[i])

    plt.xlim(cfg.occup_bins.min() - cfg.occ_indent, cfg.occup_bins.max() + cfg.occ_indent)
    plt.xlabel('number of hits')
    plt.ylabel('\# showers')

    # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    #plt.legend(prop=cfg.font, loc='best')
    if cfg.plot_text_occupancy:
        plt.text(315, 540, '10 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(870, 215, '50 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(1230, 170, '90 GeV', fontsize=cfg.font.get_size() + 2)


    plt.tight_layout()
    plt.savefig('occ.pdf', dpi=100)
    plt.show()
    
def plt_hit_e(hits, hits_list, labels, cfg=cfg, title=r'\textbf{full spectrum}'):
    fig, axs = plt.subplots(2, 1, figsize=(7,9), height_ratios=[3, 1], sharex=True)

    ## for legend ##########################################
    axs[0].hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(hits_list)):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    axs[0].set_title(title, fontsize=cfg.font.get_size(), loc='right')
    ########################################################

    h = axs[0].hist(hits, bins=cfg.hit_bins, color='lightgrey', rasterized=True)
    h = axs[0].hist(hits, bins=cfg.hit_bins, histtype='step', color='dimgrey', lw=2)
    
    for i, hits_ in enumerate(hits_list):
        h1 = axs[0].hist(hits_, bins=h[1], histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i])
        # ratio plot on the bottom
        # axs[1].plot((h[1][:-1] + h[1][1:])/2, h1[0]/h[0], linestyle='-', lw=2, marker='o', color=cfg.color_lines[i])
        # ratio plot on the bottom
        lims_min = 0.5
        lims_max = 1.9
        eps = 1e-5
        centers = np.array((h[1][:-1] + h[1][1:])/2)
        ratios = np.clip(np.array((h1[0]+eps)/(h[0]+eps)), lims_min, lims_max)
        mask = (ratios > lims_min) & (ratios < lims_max)  # mask ratios within plotting y range
        # only connect dots with adjecent points
        starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0]
        ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1
        indexes = np.stack((starts,ends)).T
        for idxs in indexes:
            sub_mask = np.zeros(len(mask), dtype=bool)
            sub_mask[idxs[0]:idxs[1]] = True
            axs[1].plot(centers[sub_mask], ratios[sub_mask], linestyle=':', lw=2, marker='o', color=cfg.color_lines[i])
        # remaining points either above or below plotting y range
        mask = (ratios == lims_min)
        axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=cfg.color_lines[i], clip_on=False)
        mask = (ratios == lims_max)
        axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=cfg.color_lines[i], clip_on=False)

    axs[1].set_ylim(lims_min, lims_max)

    # horizontal line at 1
    axs[1].axhline(1, linestyle='-', lw=1, color='k')

    axs[0].axvspan(h[1].min(), cfg.threshold, facecolor='gray', alpha=0.5, hatch= "/", edgecolor='k')
    axs[1].axvspan(h[1].min(), cfg.threshold, facecolor='gray', alpha=0.5, hatch= "/", edgecolor='k')
    # axs[0].set_xlim(h[1].min(), h[1].max()+0)
    axs[0].set_xlim(h[1].min(), 3e2)
    axs[0].set_ylim(cfg.ylim_hits[0], cfg.ylim_hits[1])

    axs[0].set_yscale('log')
    axs[0].set_xscale('log')

    plt.xlabel('visible cell energy [MeV]')
    axs[0].set_ylabel('\# cells')
    axs[1].set_ylabel('ratio to MC')


    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    plt.savefig('hits.pdf', dpi=100, bbox_inches='tight')
    plt.show()
    
def plt_esum(e_sum, e_sum_list, labels, cfg=cfg):
    plt.figure(figsize=(7, 7))
    
    h = plt.hist(np.array(e_sum), bins=cfg.e_sum_bins, color='lightgrey', rasterized=True)
    h = plt.hist(np.array(e_sum), bins=cfg.e_sum_bins,  histtype='step', color='dimgrey', lw=2)

    ## for legend ##########################################
    plt.hist(np.zeros(10), label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(e_sum_list)):
        plt.plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    ########################################################

    
    for i, e_sum_ in enumerate(e_sum_list):
        plt.hist(np.array(e_sum_), bins=h[1], histtype='step', linestyle='-', lw=2.5, color=cfg.color_lines[i])

    plt.xlim(cfg.e_sum_bins.min() - cfg.e_indent, cfg.e_sum_bins.max() + cfg.e_indent)
    plt.xlabel('energy sum [MeV]')
    plt.ylabel('\# showers')
    

    if cfg.plot_text_e:
        plt.text(300, 740, '10 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(1170, 250, '50 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(1930, 160, '90 GeV', fontsize=cfg.font.get_size() + 2)
        plt.ylim(0, 799)

   #if cfg.plot_legend_e:
        # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
        #plt.legend(prop=cfg.font, loc='best')

    plt.tight_layout()
    plt.savefig('e_sum.pdf', dpi=100)
    plt.show()

def plt_cog(cog, cog_list, labels, cfg=cfg, title=r'\textbf{full spectrum}'):
    lables = ["X", "Z", "Y"] # local coordinate system
    # plt.figure(figsize=(21, 9))
    fig, axs = plt.subplots(2, 3, figsize=(25, 9), height_ratios=[3, 1], sharex='col')

    for k, j in enumerate([0, 2, 1]):
        # plt.subplot(1, 3, k+1)

        axs[0, k].set_xlim(cfg.cog_ranges[j])
        
        h = axs[0, k].hist(np.array(cog[j]), bins=cfg.bins_cog, color='lightgrey', range=cfg.cog_ranges[j], rasterized=True)
        h = axs[0, k].hist(np.array(cog[j]), bins=h[1], color='dimgrey', histtype='step', lw=2)
        
        # for legend ##############################################
        if k == k:
        #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
            axs[0, k].hist(np.zeros(1)-10, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
            for i in range(len(cog_list)):
                axs[0, k].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
        ###########################################################

        for i, cog_ in enumerate(cog_list):
            h1 = axs[0, k].hist(np.array(cog_[j]), bins=h[1], histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i], range=cfg.cog_ranges[j])

            # ratio plot on the bottom
            lims_min = [0.5, 0.01, 0.5]
            lims_max = [1.5, 3, 1.5]
            marker_styles = ['o', 's', 'X']
            eps = 1e-5
            centers = np.array((h[1][:-1] + h[1][1:])/2)
            ratios = np.clip(np.array((h1[0]+eps)/(h[0]+eps)), lims_min[k], lims_max[k])
            mask = (ratios > lims_min[k]) & (ratios < lims_max[k])  # mask ratios within plotting y range
            # only connect dots with adjecent points
            starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0]
            ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1
            indexes = np.stack((starts,ends)).T
            for idxs in indexes:
                sub_mask = np.zeros(len(mask), dtype=bool)
                sub_mask[idxs[0]:idxs[1]] = True
                axs[1, k].plot(centers[sub_mask], ratios[sub_mask], linestyle=':', lw=2, marker='o', color=cfg.color_lines[i])
            # remaining points either above or below plotting y range
            mask = (ratios == lims_min[k])
            axs[1, k].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=cfg.color_lines[i], clip_on=False)
            mask = (ratios == lims_max[k])
            axs[1, k].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=cfg.color_lines[i], clip_on=False)

        # horizontal line at 1
        axs[1, k].axhline(1, linestyle='-', lw=1, color='k')

        # for legend ##############################################
        if k == 2:
            # plt.legend(prop=cfg.font, loc=(0.37, 0.76))
            axs[0, k].legend(prop=cfg.font, loc='best')

        # ax = plt.gca()
        axs[0, k].set_title(title, fontsize=cfg.font.get_size(), loc='right')

        ###########################################################

        axs[0, k].set_ylim(0, max(h[0]) + max(h[0])*0.5)
        axs[1, k].set_ylim(lims_min[k], lims_max[k])
        #axs[1, k].set_yscale('log')

        axs[1, k].set_xlabel(f'center of gravity {lables[j]} [mm]')
        axs[0, k].set_ylabel('\# showers')
        axs[1, k].set_ylabel('ratio to MC')
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0, hspace=0.1, wspace=0.25)
    plt.savefig('cog.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def plt_feats(events, events_list: list, labels, cfg=cfg, title=r'\textbf{full spectrum}', scale=None, density=False):
    lables = ["X", "Z", "Y"] # local coordinate system
    plt.figure(figsize=(21, 7))

    for k, j in enumerate([0, 2, 1]):
        plt.subplot(1, 3, k+1)

        plt.xlim(cfg.feats_ranges[j])
        
        h = plt.hist(np.array(events[:,j,:][events[:,3,:] != 0.0].flatten()), bins=cfg.bins_feats, color='lightgrey', range=cfg.feats_ranges[j], rasterized=True, density=density)
        # h = plt.hist(np.array(events[:,j,:][events[:,3,:] != 0.0].flatten()), bins=h[1], color='dimgrey', histtype='step', lw=2, density=density)
        
        # for legend ##############################################
        if k == k:
        #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
            plt.hist(np.ones(10)*(-300), label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2, density=density)
            for i in range(len(events_list)):
                plt.plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
        ###########################################################

        for i, events_ in enumerate(events_list):
            h2 = plt.hist(np.array(events_[:,j,:][events_[:,3,:] != 0.0].flatten()), bins=h[1], histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i], range=cfg.feats_ranges[j], density=density)

        # for legend ##############################################
        if k == 2:
            # plt.legend(prop=cfg.font, loc=(0.37, 0.76))
            plt.legend(prop=cfg.font, loc='best')

        ax = plt.gca()
        plt.title(title, fontsize=cfg.font.get_size(), loc='right')

        ###########################################################

        if density:
            plt.ylim(1e-6, max(h[0]) + max(h[0])*0.5)
        else:
            plt.ylim(1, max(h[0]) + max(h[0])*0.5)

        if scale == 'log':
            plt.yscale('log')

        plt.xlabel(f'feature {lables[j]}')
        plt.ylabel('\# points')

    
    plt.tight_layout()
    # plt.savefig('cog.pdf', dpi=100)
    plt.show()


def plt_occupancy_singleE(occ_list, occ_list_list, labels, cfg=cfg):
    fig, axs = plt.subplots(2, 1, figsize=(9,12), height_ratios=[3, 1], sharex=True)

    ## for legend ##########################################
    axs[0].hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(occ_list_list[0])):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    ########################################################

    for j, (occ, occ_list) in enumerate(zip(occ_list, occ_list_list)):   # loop over energyies

        h = axs[0].hist(occ, bins=cfg.occup_bins, color='lightgrey', rasterized=True)
        h = axs[0].hist(occ, bins=cfg.occup_bins, color='dimgrey', histtype='step', lw=2)
        
        for i, occ_ in enumerate(occ_list):   # loop over models
            h1 = axs[0].hist(occ_, bins=h[1], histtype='step', linestyle='-', lw=2.5, color=cfg.color_lines[i])

            # ratio plot on the bottom
            lims_min = 0.5
            lims_max = 1.7
            eps = 1e-5
            x_nhits_range = [cfg.occup_bins.min() - cfg.occ_indent, 400, 950, cfg.occup_bins.max() + cfg.occ_indent]

            centers = np.array((h[1][:-1] + h[1][1:])/2)
            x_mask = (centers >= x_nhits_range[j]) & (centers < x_nhits_range[j+1])

            centers = centers[x_mask]
            ratios = np.clip(np.array((h1[0]+eps)/(h[0]+eps)), lims_min, lims_max)[x_mask]
            mask = (ratios > lims_min) & (ratios < lims_max)  # mask ratios within plotting y range
            # only connect dots with adjecent points
            starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0]
            ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1
            indexes = np.stack((starts,ends)).T
            for idxs in indexes:
                sub_mask = np.zeros(len(mask), dtype=bool)
                sub_mask[idxs[0]:idxs[1]] = True
                axs[1].plot(centers[sub_mask], ratios[sub_mask], linestyle=':', lw=2, marker='o', color=cfg.color_lines[i])
            # remaining points either above or below plotting y range
            mask = (ratios == lims_min)
            axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=cfg.color_lines[i], clip_on=False)
            mask = (ratios == lims_max)
            axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=cfg.color_lines[i], clip_on=False)

    # horizontal line at 1
    axs[1].axhline(1, linestyle='-', lw=1, color='k')
    axs[1].axvline(x_nhits_range[1], linestyle='-', lw=2, color='k')
    axs[1].axvline(x_nhits_range[2], linestyle='-', lw=2, color='k')

    axs[0].set_xlim(cfg.occup_bins.min() - cfg.occ_indent, cfg.occup_bins.max() + cfg.occ_indent)
    axs[1].set_ylim(lims_min, lims_max)
    plt.xlabel('number of hits')
    axs[0].set_ylabel('\# showers')
    axs[1].set_ylabel('ratio to MC')

    # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    axs[0].legend(prop=cfg.font, loc='best')

    plt.text(350, 4, '10 GeV', fontsize=cfg.font.get_size() + 2)
    plt.text(750, 3.6, '50 GeV', fontsize=cfg.font.get_size() + 2)
    plt.text(1150, 3.2, '90 GeV', fontsize=cfg.font.get_size() + 2)

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig('occ_singleE.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def plt_esum_singleE(e_sum_list, e_sum_list_list, labels, cfg=cfg):
    fig, axs = plt.subplots(2, 1, figsize=(9,12), height_ratios=[3, 1], sharex=True)

    ## for legend ##########################################
    axs[0].hist(np.zeros(10), label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(e_sum_list)):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    ########################################################

    for j, (e_sum, e_sum_list) in enumerate(zip(e_sum_list, e_sum_list_list)):   # loop over energyies
        h = axs[0].hist(np.array(e_sum), bins=cfg.e_sum_bins, color='lightgrey', rasterized=True)
        h = axs[0].hist(np.array(e_sum), bins=cfg.e_sum_bins,  histtype='step', color='dimgrey', lw=2)
        
        for i, e_sum_ in enumerate(e_sum_list):
            h1 = axs[0].hist(np.array(e_sum_), bins=h[1], histtype='step', linestyle='-', lw=2.5, color=cfg.color_lines[i])
            
            # ratio plot on the bottom
            lims_min = 0.5
            lims_max = 2.0
            eps = 1e-5
            x_nhits_range = [cfg.e_sum_bins.min() - cfg.e_indent, 500, 1300, cfg.e_sum_bins.max() + cfg.e_indent]

            centers = np.array((h[1][:-1] + h[1][1:])/2)
            x_mask = (centers >= x_nhits_range[j]) & (centers < x_nhits_range[j+1])

            centers = centers[x_mask]
            ratios = np.clip(np.array((h1[0]+eps)/(h[0]+eps)), lims_min, lims_max)[x_mask]
            mask = (ratios > lims_min) & (ratios < lims_max)  # mask ratios within plotting y range
            # only connect dots with adjecent points
            starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0]
            ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1
            indexes = np.stack((starts,ends)).T
            for idxs in indexes:
                sub_mask = np.zeros(len(mask), dtype=bool)
                sub_mask[idxs[0]:idxs[1]] = True
                axs[1].plot(centers[sub_mask], ratios[sub_mask], linestyle=':', lw=2, marker='o', color=cfg.color_lines[i])
            # remaining points either above or below plotting y range
            mask = (ratios == lims_min)
            axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=cfg.color_lines[i], clip_on=False)
            mask = (ratios == lims_max)
            axs[1].plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=cfg.color_lines[i], clip_on=False)
    
    # horizontal line at 1
    axs[1].axhline(1, linestyle='-', lw=1, color='k')
    axs[1].axvline(x_nhits_range[1], linestyle='-', lw=2, color='k')
    axs[1].axvline(x_nhits_range[2], linestyle='-', lw=2, color='k')

    axs[0].set_xlim(cfg.e_sum_bins.min() - cfg.e_indent, cfg.e_sum_bins.max() + cfg.e_indent)
    axs[1].set_ylim(lims_min, lims_max)
    plt.xlabel('energy sum [MeV]')
    axs[0].set_ylabel('\# showers')
    axs[1].set_ylabel('ratio to MC')

    plt.text(320, 5, '10 GeV', fontsize=cfg.font.get_size() + 2)
    plt.text(880, 4.1, '50 GeV', fontsize=cfg.font.get_size() + 2)
    plt.text(1750, 3.4, '90 GeV', fontsize=cfg.font.get_size() + 2)

   #if cfg.plot_legend_e:
        # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    axs[0].legend(prop=cfg.font, loc='best')

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig('e_sum_singleE.pdf', dpi=100, bbox_inches='tight')
    plt.show()


def get_plots(events, events_list: list, labels: list = ['1', '2', '3'], title=r'\textbf{full spectrum}'):
    
    #e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real, occ_layer_real, e_layers_distibution_real, e_radial_lists_real, hits_noThreshold_list_real = get_features(events)
    dict_real = get_features(events)
    e_radial_real = dict_real['e_radial']
    occ_real = dict_real['occ']
    e_sum_real = dict_real['e_sum']
    hits_real = dict_real['hits_noThreshold']
    e_layers_real = dict_real['e_layers']
    
    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = [], [], [], [], []
    
    for i in range(len(events_list)):
        dict_fake = get_features(events_list[i])
        
        e_radial_list.append(dict_fake['e_radial'])
        occ_list.append(dict_fake['occ'])
        e_sum_list.append(dict_fake['e_sum'])
        hits_list.append(dict_fake['hits_noThreshold'])
        e_layers_list.append(dict_fake['e_layers'])
        
    
    plt_radial(e_radial_real, e_radial_list, labels=labels, title=title)
    plt_spinal(e_layers_real, e_layers_list, labels=labels, title=title)
    plt_hit_e(hits_real, hits_list, labels=labels, title=title)
    plt_occupancy(occ_real, occ_list, labels=labels)
    plt_esum(e_sum_real, e_sum_list, labels=labels)


def get_observables_for_plotting(events, events_list: list):
    
    dict_real = get_features(events)
    e_radial_real = dict_real['e_radial']
    occ_real = dict_real['occ']
    e_sum_real = dict_real['e_sum']
    hits_real = dict_real['hits_noThreshold']
    e_layers_real = dict_real['e_layers']
    
    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = [], [], [], [], []
    
    for i in range(len(events_list)):
        dict_fake = get_features(events_list[i])
        
        e_radial_list.append(dict_fake['e_radial'])
        occ_list.append(dict_fake['occ'])
        e_sum_list.append(dict_fake['e_sum'])
        hits_list.append(dict_fake['hits_noThreshold'])
        e_layers_list.append(dict_fake['e_layers'])
        
    fakes_list = [e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list]

    real_list = [e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real]
    
    return real_list, fakes_list


def get_plots_from_observables(real_list: list, fakes_list: list, labels: list = ['1', '2', '3'], title=r'\textbf{full spectrum}', events=40_000):
    
    if len(real_list) != len(fakes_list):
        e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real, occ_layer_real, e_layers_distibution_real, e_radial_lists_real, hits_noThreshold_list_real = real_list
    else:
        e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real = real_list
    
    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = fakes_list        
    
    plt_radial(e_radial_real, e_radial_list, labels=labels, title=title, events=events)
    plt_spinal(e_layers_real, e_layers_list, labels=labels, title=title)
    if len(real_list) != len(fakes_list):
        plt_hit_e(hits_noThreshold_list_real, hits_list, labels=labels, title=title)
    else:
        plt_hit_e(hits_real, hits_list, labels=labels, title=title)
    plt_occupancy(occ_real, occ_list, labels=labels)
    plt_esum(e_sum_real, e_sum_list, labels=labels)


def get_plots_from_observables_singleE(real_list_list: list, fakes_list_list: list, labels: list = ['1', '2', '3']):
    
    occ_real_list, occ_fake_list_list = [], []
    e_sum_real_list, e_sum_fake_list_list = [], []
    for i in range(len(real_list_list)):  # observables for certain single energy
        e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real = real_list_list[i]
        occ_real_list.append(occ_real)
        e_sum_real_list.append(e_sum_real)

        e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = fakes_list_list[i]
        occ_fake_list_list.append(occ_list)
        e_sum_fake_list_list.append(e_sum_list)
    
    plt_occupancy_singleE(occ_real_list, occ_fake_list_list, labels=labels)
    plt_esum_singleE(e_sum_real_list, e_sum_fake_list_list, labels=labels)



MAP, offset = create_map(X, Y, Z, dm=1)

