import sys
sys.path.append('../')
import time
s_time = time.time()

import h5py
import numpy as np
import sys
import pickle
import h5py

# from configs import Configs
import utils.plotting as plotting
from utils.plotting import get_projections, MAP, layer_bottom_pos

# cfg = Configs()

# print(cfg.__dict__)


###############################################  PARAMS

### SINGLE ENERGY PARAMS
full_spectrum = False
min_energy_list = [10, 50, 90]
max_energy_list = [10, 50, 90]
n_events = 2000
out_path = 'BEEGFS/6_PointCloudDiffusion/output/singleE/'

### FULL SPECTRUM PARAMS
# full_spectrum = True
# min_energy_list = [10]
# max_energy_list = [90]
# n_events = 40_000
# out_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/full/'


## COMMON PARAMETERS
caloclouds_list = ['ddpm', 'edm', 'cm']   # 'ddpm, 'edm', 'cm'
seed_list = [12345, 123456, 1234567]
pickle_path = out_path + 'pickle/'
n = n_events   # for debugging --> not projecting all events

###############################################

for j in range(len(min_energy_list)):
    dict = {}
    min_energy = min_energy_list[j]
    max_energy = max_energy_list[j]

    #### load save data
    if full_spectrum:
        path = 'BEEGFS/data/calo-clouds/hdf5/all_steps/validation/photon-showers_10-90GeV_A90_Zpos4.slcio.hdf5'
    else:
        path = 'BEEGFS/data/calo-clouds/hdf5/all_steps/validation/photon-showers_{}GeV_A90_Zpos4.slcio.hdf5'.format(min_energy)
    real_showers = h5py.File(path, 'r')['events'][:]
    real_showers[:, -1] = real_showers[:, -1] * 1000   # GeV to MeV
    print('real showers shape: ', real_showers.shape)

    i = 0
    f = out_path + '{}-{}GeV_{}_{}_seed{}.npz'.format(str(min_energy), str(max_energy), str(n_events), caloclouds_list[i], str(seed_list[i]))
    fake_showers = np.load(f)['fake_showers']
    print(f, ' loaded')
    print('fake showers shape: ', fake_showers.shape)

    i = 1
    f = out_path + '{}-{}GeV_{}_{}_seed{}.npz'.format(str(min_energy), str(max_energy), str(n_events), caloclouds_list[i], str(seed_list[i]))
    fake_showers_2 = np.load(f)['fake_showers']
    print(f, ' loaded')
    print('fake showers shape: ', fake_showers_2.shape)

    i = 2
    f = out_path + '{}-{}GeV_{}_{}_seed{}.npz'.format(str(min_energy), str(max_energy), str(n_events), caloclouds_list[i], str(seed_list[i]))
    fake_showers_3 = np.load(f)['fake_showers']
    print(f, ' loaded')
    print('fake showers shape: ', fake_showers_3.shape)


    # projection
    print('files loaded (all energy hits in MeV). now projection.')
    events, cloud = get_projections(real_showers[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)
    events_fake, cloud_fake = get_projections(fake_showers[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)
    events_fake_2, cloud_fake_2 = get_projections(fake_showers_2[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)
    events_fake_3, cloud_fake_3 = get_projections(fake_showers_3[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)

    # calculate cog
    c_cog = plotting.get_cog(cloud)
    print('len c_cog: ', len(c_cog[0]), len(c_cog[1]), len(c_cog[2]))
    c_cog_2 = plotting.get_cog(cloud_fake)
    print('len c_cog_2: ', len(c_cog_2[0]), len(c_cog_2[1]), len(c_cog_2[2]))
    c_cog_3 = plotting.get_cog(cloud_fake_2)
    print('len c_cog_3: ', len(c_cog_3[0]), len(c_cog_3[1]), len(c_cog_3[2]))
    c_cog_4 = plotting.get_cog(cloud_fake_3)
    print('len c_cog_4: ', len(c_cog_4[0]), len(c_cog_4[1]), len(c_cog_4[2]))
    c_cog_real = c_cog
    c_cog_fake = [c_cog_2, c_cog_3, c_cog_4]


    # features / observablesc
    print('get observables')
    real_list, fakes_list = plotting.get_observables_for_plotting(events, [events_fake, events_fake_2, events_fake_3])


    # save to dict
    dict['real_list'] = real_list
    dict['fakes_list'] = fakes_list
    dict['c_cog_real'] = c_cog_real
    dict['c_cog_fake'] = c_cog_fake

    with open(pickle_path+'dict_{}-{}GeV.pickle'.format(str(min_energy), str(max_energy)), 'wb') as f:
        pickle.dump(dict, f)
    print('dict saved in pickle file')


print('done. took {} mins'.format((time.time()-s_time)/60))