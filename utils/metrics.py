import numpy as np
from scipy.stats import wasserstein_distance

# metrics needed:
# threshold applied for all, at hit/cell level

# cog x
# cog y
# cog z
# hit energy  (not using all hits)
# occupancy
# energy sum or sampling fraction
# energy sum per layer (10 bins) --> mean
# radial energy (10 bins) --> mean

# all normalized by with real maximum value


def percentile_edges_layer_occupancy(occ_layers):
    occ_layers_list_sum = occ_layers.sum(axis=0)
    bin_edges = []
    total = occ_layers_list_sum.sum()
    running_sum = 0
    fraction_interval_min = 0.0
    fraction_interval_max = 0.1
    for i in range(len(occ_layers_list_sum)):
        running_sum += occ_layers_list_sum[i]
        running_fraction = running_sum / total
        if (running_fraction >= fraction_interval_min):# and (running_fraction <= fraction_interval_max):
            fraction_interval_min += 0.1
            fraction_interval_max += 0.1
            bin_edges.append(i)

    return bin_edges


def percentile_edges_radial_occupancy(occ_radial, min_radius = 0, max_radius = 300): 
    """
        occ_radial, i;e. dict['e_radial'][0]
    """
    radial_edges = []
    total = occ_radial.sum()
    # running_sum = 0
    fraction_interval_min = 0.0
    fraction_interval_max = 0.1
    for radius in np.arange(0,300,1e-3):
        running_sum = occ_radial[occ_radial < radius].sum()
        running_fraction = running_sum / total
        if (running_fraction > fraction_interval_min) and (running_fraction <= fraction_interval_max):
            fraction_interval_min += 0.1
            fraction_interval_max += 0.1
            radial_edges.append(radius)
    radial_edges[0] = min_radius  # set the first bin to be the minimum radius
    radial_edges[-1] = max_radius # set the last bin to be the maximum radius
    return radial_edges

\
def binned_layer_occupcany(occ_layers, bin_edges = [0, 29]):
    binned_occ_layer_list = []
    for i in range(len(bin_edges)-1):
        occ_sub = occ_layers[:,bin_edges[i]:bin_edges[i+1]]
        if bin_edges[i+1] == bin_edges[-1]: # last bin
            occ_sub = occ_layers[:,bin_edges[i]:bin_edges[i+1]+1] # include last bin
        occ_sub = occ_sub.reshape(occ_sub.shape[0], -1)
        binned_occ_layer_list.append(occ_sub.sum(axis=1))
    binned_layer_occ = np.array(binned_occ_layer_list)
    return binned_layer_occ   # shape (bin centeres, events)


def binned_layer_energy(e_layers_distibution, bin_edges = [0, 29]):
    binned_e_layer_list = []
    for i in range(len(bin_edges)-1):
        e_sub = e_layers_distibution[:,bin_edges[i]:bin_edges[i+1]]
        if bin_edges[i+1] == bin_edges[-1]: # last bin
            e_sub = e_layers_distibution[:,bin_edges[i]:bin_edges[i+1]+1] # include last bin
        e_sub = e_sub.reshape(e_sub.shape[0], -1)
        binned_e_layer_list.append(e_sub.sum(axis=1))
    binned_layer_e = np.array(binned_e_layer_list)
    return binned_layer_e   # shape (bin centeres, events)


def binned_radial_occupancy_sum(occ_radial, bin_edges = [0, 300]):   # just for debugging
    """
        occ_radial, i;e. dict['e_radial'][0]

        returns:
            binned_radial_occ_sum: shape (bin centeres,)
    """
    binned_occ_radial = []
    for i in range(len(bin_edges)-1):
        mask =  (occ_radial >= bin_edges[i]) & (occ_radial < bin_edges[i+1])
        radial_sub = occ_radial[mask]
        binned_occ_radial.append(radial_sub.sum())
    binned_radial_occ_sum = np.array(binned_occ_radial)
    return binned_radial_occ_sum   # shape (bin centeres,)  --> should each be about 10% of total occupancy


def binned_radial_occupancy(occ_radial, bin_edges = [0, 300]):   # just for debugging
    """
        occ_radial, i;e. dict['e_radial'][0]
    """
    binned_occ_radial = []
    for i in range(len(bin_edges)-1):
        mask =  (occ_radial >= bin_edges[i]) & (occ_radial < bin_edges[i+1])
        radial_sub = occ_radial[mask]
        binned_occ_radial.append(radial_sub.sum())
    binned_radial_occ = np.array(binned_occ_radial)

    return binned_radial_occ   # shape (bin centeres,)  --> should each be about 10% of total occupancy


def binned_radial_energy(e_radial, bin_edges = [0, 300]):   # just for debugging
    """
        e_radial[0]: radial occupancy
        e_radial[1]: radial energy

        returns: 
            binned_radial_e: shape (bin centeres, events)
    """
    binned_e_radial = []
    for i in range(len(bin_edges)-1):
        layer_radial = []
        for e_rad in e_radial:
            mask = (e_rad[0] >= bin_edges[i]) & (e_rad[0] < bin_edges[i+1])
            radial_sub = e_rad[1][mask]
            layer_radial.append(radial_sub.sum())
        binned_e_radial.append(layer_radial)
    binned_radial_e = np.vstack(binned_e_radial)
    return binned_radial_e 


def merge_dicts(dict_list, key_exceptions=[]):
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in key_exceptions:
                if key in merged_dict:
                    # Assuming you want to concatenate lists or concatenate numpy arrays for duplicate keys.
                    if isinstance(merged_dict[key], list):
                        merged_dict[key].extend(value)
                    elif isinstance(merged_dict[key], np.ndarray):
                        if len(merged_dict[key].shape) == 1:
                            merged_dict[key] = np.concatenate((merged_dict[key], value), axis=0)
                        elif len(merged_dict[key].shape) == 2:
                            merged_dict[key] = np.concatenate((merged_dict[key], value), axis=1)
                    # Add other cases for specific data types if needed.
                else:
                    merged_dict[key] = value
    return merged_dict


def get_event_observables_from_dict(dict):
    cog_x = dict['cog_x'].reshape(-1,1)   # 0
    cog_y = dict['cog_y'].reshape(-1,1)   # 1
    cog_z = dict['cog_z'].reshape(-1,1)   # 2
    occ = dict['occ'].reshape(-1,1)       # 3
    sampling_fraction = (dict['e_sum'] / (dict['incident_energy']*1000)).reshape(-1,1)  # 4
    hits = dict['hits'][0:len(cog_x)].reshape(-1,1)   # only use the first X hits       # 5
    binned_layer_e = dict['binned_layer_e'].T                                           # 6-15
    binned_radial_e = dict['binned_radial_e'].T                                         # 16-25
    return np.hstack([cog_x, cog_y, cog_z, occ, sampling_fraction, hits, binned_layer_e, binned_radial_e])


def calc_wdist(obs_real, obs_model, iterations=5, batch_size=10_000):
    means, stds = [], []
    for i in range(obs_real.shape[1]):
        j = 0
        wdists = []
        for _ in range(iterations):
            wdist = wasserstein_distance(obs_real[j:j+batch_size,i], obs_model[j:j+batch_size,i])
            wdists.append(wdist)
            j += batch_size
        # print(f'feature {i}: {np.mean(wdists)} +- {np.std(wdists)}')
        means.append(np.mean(wdists))
        stds.append(np.std(wdists))
    return np.array(means), np.array(stds)


def combine_scores(means, stds):
    mean = np.mean(means)
    std = np.sqrt(np.sum(stds**2))  # error propagation
    return mean, std