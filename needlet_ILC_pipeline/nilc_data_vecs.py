import sys
sys.path.append('../shared')
import numpy as np
import subprocess
import healpy as hp
from scipy import stats
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc, weight_maps_exist
from load_weight_maps import load_wt_maps
from utils import tsz_spectral_response, GaussianNeedlets, build_NILC_maps, get_scalings


def get_scaled_maps_and_wts(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    CMB_map_unscaled, tSZ_map_unscaled, noise_maps_unscaled: unscaled maps of all the components
    all_wt_maps: (Nscalings, 2, 2, Nsplits, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps
                dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
                      idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
                dim1: idx0 for unscaled CMB, idx1 for scaled CMB
                dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
                Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 000 (all unscaled)

    '''

    N_preserved_comps = 2
    Nsplits = 2

    #array for all weight maps
    Nscalings = len(inp.scaling_factors)
    all_wt_maps = np.zeros((Nscalings, 2, 2, N_preserved_comps, Nsplits, inp.Nscales, len(inp.freqs), 12*inp.nside**2))
    scalings = get_scalings(inp)

    for s, scaling in enumerate(scalings):

        #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
        if s==0:
            CC, T, CMB_map_unscaled, tSZ_map_unscaled, noise_maps_unscaled = generate_freq_maps(inp, sim, scaling=scaling)
        
        #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
        for split in [1,2]:
            if not weight_maps_exist(sim, split, inp, scaling=scaling): #check if not all the weight maps already exist
                
                #remove any existing weight maps for this sim and scaling to prevent pyilc errors
                if scaling is not None:  
                    scaling_str = ''.join(str(e) for e in scaling)                                                  
                    subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/{scaling_str}/sim{sim}_split{split}*', shell=True, env=env)
                else:
                    subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/sim{sim}_split{split}*', shell=True, env=env)
                
                #generate and save files containing frequency maps and then run pyilc
                if split == 1: #generate_freq_maps gets maps for both splits, so only need to generate once
                    generate_freq_maps(inp, sim, scaling=scaling)
                setup_pyilc(sim, split, inp, env, suppress_printing=True, scaling=scaling) #set suppress_printing=False to debug pyilc runs

            #load weight maps
            CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, split, scaling=scaling)
            all_wt_maps[scaling[0], scaling[1], scaling[2], split-1] = np.array([CMB_wt_maps, tSZ_wt_maps])

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights
        
    return CMB_map_unscaled, tSZ_map_unscaled, noise_maps_unscaled, all_wt_maps



def get_scaled_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (Nscalings, 2, 2, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: idx0 for unscaled CMB, idx1 for scaled CMB
        dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        preserved_comps = CMB, ftSZ
        For example, Clpq[1,0,1,0,1] is cross-spectrum of CMB-preserved NILC map and 
        ftSZ-preserved NILC map when ftSZ is scaled according to inp.scaling_factors[1]
        Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)
    '''
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, ftSZ

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)

    #get maps and weight maps
    CMB_map, tSZ_map, noise_maps, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env)

    #get map level propagation of components
    Npix = 12*inp.nside**2
    Nscalings = len(inp.scaling_factors)
    Nsplits = 2
    all_map_level_prop = np.zeros((Nscalings, 2, 2, Nsplits, N_preserved_comps, Npix)) 
    scalings = get_scalings(inp)
    
    for scaling in scalings:

        CMB_amp, tSZ_amp_extra = 1,1
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor

        for split in [1,2]:

            map_0 = CMB_amp*CMB_map + tSZ_amp_extra*g_tsz[0]*tSZ_map + noise_maps[0,split-1]
            map_1 = CMB_amp*CMB_map + tSZ_amp_extra*g_tsz[1]*tSZ_map + noise_maps[1,split-1]
            
            CMB_wt_maps, tSZ_wt_maps = all_wt_maps[scaling[0], scaling[1], scaling[2], split-1]

            CMB_preserved, tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[map_0, map_1])
            all_map_level_prop[scaling[0], scaling[1], scaling[2], split-1, 0] = CMB_preserved
            all_map_level_prop[scaling[0], scaling[1], scaling[2], split-1, 1] = tSZ_preserved

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights

    #define and fill in array of data vectors (dim 0 has size Nscalings for which scaling in inp.scaling_factors is used)
    Clpq_tmp = np.zeros((Nscalings,2,2,N_preserved_comps, N_preserved_comps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((Nscalings,2,2,N_preserved_comps, N_preserved_comps, inp.Nbins)) #binned

    for scaling in scalings:

        CMB_preserved_s1, CMB_preserved_s2 = all_map_level_prop[scaling[0], scaling[1], scaling[2], :, 0]
        tSZ_preserved_s1, tSZ_preserved_s2 = all_map_level_prop[scaling[0], scaling[1], scaling[2], :, 1]
    
        Clpq_tmp[scaling[0], scaling[1], scaling[2], 0,0] = hp.anafast(CMB_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)
        Clpq_tmp[scaling[0], scaling[1], scaling[2], 1,1] = hp.anafast(tSZ_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)
        Clpq_tmp[scaling[0], scaling[1], scaling[2], 0,1] = hp.anafast(CMB_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)
        Clpq_tmp[scaling[0], scaling[1], scaling[2], 1,0] = hp.anafast(tSZ_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)

        ells = np.arange(inp.ellmax+1)
        all_spectra = [ Clpq_tmp[scaling[0], scaling[1], scaling[2], 0,0], 
                        Clpq_tmp[scaling[0], scaling[1], scaling[2], 1,1], 
                        Clpq_tmp[scaling[0], scaling[1], scaling[2], 0,1], 
                        Clpq_tmp[scaling[0], scaling[1], scaling[2], 1,0]]
        
        index_mapping = {0:(0,0), 1:(1,1), 2:(0,1), 3: (1,0)} #maps idx to p,q for all_spectra
        for idx, Cl in enumerate(all_spectra):
            Dl = ells*(ells+1)/2/np.pi*Cl
            res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
            mean_ells = (res[1][:-1]+res[1][1:])/2
            p,q = index_mapping[idx]
            Clpq[scaling[0], scaling[1], scaling[2], p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
        
        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights

    return Clpq


def get_maps_and_wts(sim, inp, env, pars=None):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    CMB_map, tSZ_map, noise_maps: maps of all the components
    all_wt_maps: (Nsplits, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps

    '''

    #array for all weight maps
    Nsplits = 2
    N_preserved_comps = 2
    all_wt_maps = np.zeros((Nsplits, N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))

    #pars string
    if pars is not None:
        pars_str = f'_pars{pars[0]:.3f}_{pars[1]:.3f}_'
    else:
        pars_str = ''

    #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T)
    CC, T, CMB_map, tSZ_map, noise_maps = generate_freq_maps(inp, sim, pars=pars)
       
    #generate and save files containing frequency maps and then run pyilc
    for split in [1,2]:
        if not weight_maps_exist(sim, split, inp, pars=pars): #check if not all the weight maps already exist
            #remove any existing weight maps for this sim and pars to prevent pyilc errors, and then run pyilc
            subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/sim{sim}_split{split}{pars_str}*', shell=True, env=env)
            setup_pyilc(sim, split, inp, env, suppress_printing=True, pars=pars) #set suppress_printing=False to debug pyilc runs
        CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, split, pars=pars) #load weight maps
        all_wt_maps[split-1] = np.array([CMB_wt_maps, tSZ_wt_maps])

    return CMB_map, tSZ_map, noise_maps, all_wt_maps



def get_data_vectors(inp, env, sim=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    sim: int, simulation number (if sim is None, a random simulation number will be used)
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clpq: (N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        preserved_comps = CMB, ftSZ
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, ftSZ

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)

    #get maps and weight maps
    CMB_map, tSZ_map, noise_maps, all_wt_maps = get_maps_and_wts(sim, inp, env, pars=pars)

    #get map level propagation of components
    Npix = 12*inp.nside**2
    Nsplits = 2
    all_map_level_prop = np.zeros((Nsplits, N_preserved_comps, Npix)) 

    for split in [1,2]:
        map_0 = CMB_map + g_tsz[0]*tSZ_map + noise_maps[0,split-1]
        map_1 = CMB_map + g_tsz[1]*tSZ_map + noise_maps[1,split-1]
        CMB_wt_maps, tSZ_wt_maps = all_wt_maps[split-1]
        all_map_level_prop[split-1] = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[map_0, map_1])

    #define and fill in array of data vectors
    Clpq_tmp = np.zeros((N_preserved_comps, N_preserved_comps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((N_preserved_comps, N_preserved_comps, inp.Nbins)) #binned

    CMB_preserved_s1,  CMB_preserved_s2 = all_map_level_prop[:,0]
    tSZ_preserved_s1,  tSZ_preserved_s2 = all_map_level_prop[:,1]

    Clpq_tmp[0,0] = hp.anafast(CMB_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[1,1] = hp.anafast(tSZ_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[0,1] = hp.anafast(CMB_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[1,0] = hp.anafast(tSZ_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)

    ells = np.arange(inp.ellmax+1)
    all_spectra = [Clpq_tmp[0,0], Clpq_tmp[1,1], Clpq_tmp[0,1], Clpq_tmp[1,0]]
    
    index_mapping = {0:(0,0), 1:(1,1), 2:(0,1), 3: (1,0)} #maps idx to p,q for all_spectra
    for idx, Cl in enumerate(all_spectra):
        Dl = ells*(ells+1)/2/np.pi*Cl
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        mean_ells = (res[1][:-1]+res[1][1:])/2
        p,q = index_mapping[idx]
        Clpq[p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)

    return Clpq