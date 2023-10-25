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
    CMB_map_unscaled, tSZ_map_unscaled, noise1_map_unscaled, noise2_map_unscaled: unscaled maps of all the components
    all_wt_maps: (Nscalings, 2, 2, 2, 2, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps
                dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
                      idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
                dim1: idx0 for unscaled CMB, idx1 for scaled CMB
                dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
                dim3: idx0 for unscaled noise90, idx1 for scaled noise90
                dim4: idx0 for unscaled noise150, idx1 for scaled noise150
                Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)

    '''

    N_preserved_comps = 2

    #array for all weight maps
    Nscalings = len(inp.scaling_factors)
    all_wt_maps = np.zeros((Nscalings, 2, 2, 2, 2, N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))
    scalings = get_scalings(inp)

    for s, scaling in enumerate(scalings):

        #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
        if s==0:
            CC, T, N1, N2, CMB_map_unscaled, tSZ_map_unscaled, noise1_map_unscaled, noise2_map_unscaled = generate_freq_maps(inp, sim, scaling=scaling)
        
        #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
        if not weight_maps_exist(sim, inp, scaling=scaling): #check if not all the weight maps already exist
            
            #remove any existing weight maps for this sim and scaling to prevent pyilc errors
            if scaling is not None:  
                scaling_str = ''.join(str(e) for e in scaling)                                                  
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/{scaling_str}/sim{sim}*', shell=True, env=env)
            else:
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)
            
            #generate and save files containing frequency maps and then run pyilc
            generate_freq_maps(inp, sim, scaling=scaling)
            setup_pyilc(sim, inp, env, suppress_printing=True, scaling=scaling) #set suppress_printing=False to debug pyilc runs

        #load weight maps
        CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, scaling=scaling)
        all_wt_maps[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4]] = np.array([CMB_wt_maps, tSZ_wt_maps])

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights
    
    return CMB_map_unscaled, tSZ_map_unscaled, noise1_map_unscaled, noise2_map_unscaled, all_wt_maps



def get_scaled_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (Nscalings, 2, 2, 2, 2, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: idx0 for unscaled CMB, idx1 for scaled CMB
        dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        dim3: idx0 for unscaled noise90, idx1 for scaled noise90
        dim4: idx0 for unscaled noise150, idx1 for scaled noise150
        preserved_comps = CMB, ftSZ
        For example, Clpq[1,0,1,0,1,0,1] is cross-spectrum of CMB-preserved NILC map and 
        ftSZ-preserved NILC map when ftSZ and 150 GHz noise are scaled according to inp.scaling_factors[1]
        Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)
    '''
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, ftSZ

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = [1.,0.]
    g_noise2 = [0.,1.]

    #get maps and weight maps
    CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env)

    #get map level propagation of components
    Npix = 12*inp.nside**2
    Nscalings = len(inp.scaling_factors)
    all_map_level_prop = np.zeros((Nscalings,2,2,2,2, N_preserved_comps, Npix)) 
    scalings = get_scalings(inp)
    
    for scaling in scalings:

        CMB_amp, tSZ_amp_extra, noise1_amp, noise2_amp = 1,1,1,1
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor
        if scaling[3]: noise1_amp = scale_factor
        if scaling[4]: noise2_amp = scale_factor

        map_0 = CMB_amp*CMB_map + tSZ_amp_extra*g_tsz[0]*tSZ_map + noise1_amp*g_noise1[0]*noise1_map + noise2_amp*g_noise2[0]*noise2_map
        map_1 = CMB_amp*CMB_map + tSZ_amp_extra*g_tsz[1]*tSZ_map + noise1_amp*g_noise1[1]*noise1_map + noise2_amp*g_noise2[1]*noise2_map
        
        CMB_wt_maps, tSZ_wt_maps = all_wt_maps[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4]]

        CMB_preserved, tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[map_0, map_1])
        all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0] = CMB_preserved
        all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1] = tSZ_preserved

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights

    #define and fill in array of data vectors (dim 0 has size Nscalings for which scaling in inp.scaling_factors is used)
    Clpq_tmp = np.zeros((Nscalings,2,2,2,2, N_preserved_comps, N_preserved_comps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((Nscalings,2,2,2,2, N_preserved_comps, N_preserved_comps, inp.Nbins)) #binned

    for scaling in scalings:

        CMB_preserved = all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0]
        tSZ_preserved = all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1]
    
        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0,0] = hp.anafast(CMB_preserved, lmax=inp.ellmax)
        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1,1] = hp.anafast(tSZ_preserved, lmax=inp.ellmax)
        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0,1] = hp.anafast(CMB_preserved, tSZ_preserved, lmax=inp.ellmax)
        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1,0] = hp.anafast(tSZ_preserved, CMB_preserved, lmax=inp.ellmax)

        ells = np.arange(inp.ellmax+1)
        all_spectra = [ Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],0,0], 
                        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],1,1], 
                        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],0,1], 
                        Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],1,0]]
        
        index_mapping = {0:(0,0), 1:(1,1), 2:(0,1), 3: (1,0)} #maps idx to p,q for all_spectra
        for idx, Cl in enumerate(all_spectra):
            Dl = ells*(ells+1)/2/np.pi*Cl
            res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
            mean_ells = (res[1][:-1]+res[1][1:])/2
            p,q = index_mapping[idx]
            Clpq[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
        
        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights


    if inp.remove_files:
        #remove pyilc outputs
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/*/sim{sim}*', shell=True, env=env)
        #remove frequency map files
        subprocess.call(f'rm {inp.output_dir}/maps/*/sim{sim}_freq*.fits', shell=True, env=env)

    return Clpq


def get_maps_and_wts(sim, inp, env, pars=None):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object
    pars: array of floats [Acmb, Atsz, Anoise1, Anoise2] (if not provided, all assumed to be 1)

    RETURNS
    -------
    CMB_map, tSZ_map, noise1_map, noise2_map: maps of all the components
    all_wt_maps: (N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps

    '''

    #array for all weight maps
    N_preserved_comps = 2
    all_wt_maps = np.zeros((N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))

    #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(inp, sim, pars=pars)
       
    #ngenerate and save files containing frequency maps and then run pyilc
    generate_freq_maps(inp, sim)
    setup_pyilc(sim, inp, env, suppress_printing=True) #set suppress_printing=False to debug pyilc runs

    #load weight maps
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim)
    all_wt_maps = np.array([CMB_wt_maps, tSZ_wt_maps])

    return CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps



def get_data_vectors(inp, env, sim=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    sim: int, simulation number (if sim is None, a random simulation number will be used)
    pars: array of floats [Acmb, Atsz, Anoise1, Anoise2] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clpq: (N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        preserved_comps = CMB, ftSZ
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)
        print('sim: ', sim, flush=True)
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, ftSZ

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = [1.,0.]
    g_noise2 = [0.,1.]

    #get maps and weight maps
    CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps = get_maps_and_wts(sim, inp, env, pars=pars)

    #get map level propagation of components
    Npix = 12*inp.nside**2
    all_map_level_prop = np.zeros((N_preserved_comps, Npix)) 

    map_0 = CMB_map + g_tsz[0]*tSZ_map + g_noise1[0]*noise1_map + g_noise2[0]*noise2_map
    map_1 = CMB_map + g_tsz[1]*tSZ_map + g_noise1[1]*noise1_map + g_noise2[1]*noise2_map
    
    CMB_wt_maps, tSZ_wt_maps = all_wt_maps
    all_map_level_prop = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[map_0, map_1])

    #define and fill in array of data vectors
    Clpq_tmp = np.zeros((N_preserved_comps, N_preserved_comps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((N_preserved_comps, N_preserved_comps, inp.Nbins)) #binned

    CMB_preserved = all_map_level_prop[0]
    tSZ_preserved = all_map_level_prop[1]

    Clpq_tmp[0,0] = hp.anafast(CMB_preserved, lmax=inp.ellmax)
    Clpq_tmp[1,1] = hp.anafast(tSZ_preserved, lmax=inp.ellmax)
    Clpq_tmp[0,1] = hp.anafast(CMB_preserved, tSZ_preserved, lmax=inp.ellmax)
    Clpq_tmp[1,0] = hp.anafast(tSZ_preserved, CMB_preserved, lmax=inp.ellmax)

    ells = np.arange(inp.ellmax+1)
    all_spectra = [Clpq_tmp[0,0], Clpq_tmp[1,1], Clpq_tmp[0,1], Clpq_tmp[1,0]]
    
    index_mapping = {0:(0,0), 1:(1,1), 2:(0,1), 3: (1,0)} #maps idx to p,q for all_spectra
    for idx, Cl in enumerate(all_spectra):
        Dl = ells*(ells+1)/2/np.pi*Cl
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        mean_ells = (res[1][:-1]+res[1][1:])/2
        p,q = index_mapping[idx]
        Clpq[p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)

    if inp.remove_files:
        #remove pyilc outputs
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/*/sim{sim}*', shell=True, env=env)
        #remove frequency map files
        subprocess.call(f'rm {inp.output_dir}/maps/*/sim{sim}_freq*.fits', shell=True, env=env)

    return Clpq