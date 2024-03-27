import sys
sys.path.append('../shared')
import numpy as np
import shutil
import tempfile
import itertools
import healpy as hp
from scipy import stats
from generate_maps import generate_freq_maps, save_scaled_freq_maps
from pyilc_interface import setup_pyilc, load_wt_maps
from utils import tsz_spectral_response, cib_spectral_response, GaussianNeedlets, build_NILC_maps, get_scalings


def get_scaled_maps_and_wts(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    comp_maps_unscaled: (Ncomps, Npix) ndarray of unscaled maps of all components 
    noise_maps_unscaled: (Nfreqs, Nsplits, Npix) ndarray of unscaled maps of noise
    all_wt_maps: (Nscalings, 2**Ncomps, Nsplits, Ncomps, Nscales, Nfreqs, Npix) ndarray containing all weight maps
                dim0: idx i indicates that "scaled" means maps are scaled according to scaling factor i from input, up to idx Nscalings
                dim1: indices correspond to different combinations of scaled and unscaled components
                Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00 (all unscaled)

    '''

    Ncomps = len(inp.comps)
    Npix = 12*inp.nside**2
    Nsplits = 2

    #array for all weight maps
    Nscalings = len(inp.scaling_factors)
    all_wt_maps = np.zeros((Nscalings, 2**Ncomps, Nsplits, Ncomps, inp.Nscales, len(inp.freqs), Npix))
    scalings = get_scalings(inp)
    comp_scalings = [list(i) for i in itertools.product([0, 1], repeat=Ncomps)]

    for s, scaling in enumerate(scalings):

        #create temporary directory to place maps
        map_tmpdir = tempfile.mkdtemp(dir=inp.output_dir)
        
        #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
        for split in [1,2]:
                
            #generate and save files containing frequency maps and then run pyilc
            if split == 1: #generate_freq_maps gets maps for both splits, so only need to generate once 
                #create frequency maps (GHz) consisting of components and noise. Get power spectra of component maps.    
                if s==0: 
                    comp_spectra, comp_maps_unscaled, noise_maps_unscaled = generate_freq_maps(inp, sim, scaling=scaling, map_tmpdir=map_tmpdir)
                else:
                    save_scaled_freq_maps(inp, sim, scaling, map_tmpdir, comp_maps_unscaled, noise_maps_unscaled)
            pyilc_tmpdir = setup_pyilc(sim, split, inp, env, map_tmpdir, suppress_printing=True, scaling=scaling) #set suppress_printing=False to debug pyilc runs

            #load weight maps
            wt_maps = load_wt_maps(inp, sim, split, pyilc_tmpdir)
            all_wt_maps[scaling[0], comp_scalings.index(scaling[1:]), split-1] = wt_maps
            shutil.rmtree(pyilc_tmpdir)
        
        shutil.rmtree(map_tmpdir)

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights
        
    return comp_maps_unscaled, noise_maps_unscaled, all_wt_maps



def get_scaled_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (Nscalings, 2**Ncomps, Ncomps, Ncomps, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: indices correspond to different combinations of scaled and unscaled components
        For example, Clpq[1,1,0,1] is cross-spectrum of CMB-preserved NILC map and 
        ftSZ-preserved NILC map when ftSZ is scaled according to inp.scaling_factors[1]
        Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)
    '''
    
    Ncomps = len(inp.comps)
    Npix = 12*inp.nside**2
    Nfreqs = len(inp.freqs)
    Nscalings = len(inp.scaling_factors)
    Nsplits = 2
    scalings = get_scalings(inp)

    #get needlet filters
    h = GaussianNeedlets(inp)[1]

    # spectral response vectors
    sed_arr = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        if comp == 'tsz':
            sed_arr[c] = tsz_spectral_response(inp.freqs)
        elif comp == 'cib':
            sed_arr[c] = cib_spectral_response(inp.freqs)

    #get maps and weight maps
    comp_maps_unscaled, noise_maps, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env)

    #get map level propagation of components
    all_map_level_prop = np.zeros((Nscalings, 2**Ncomps, Nsplits, Ncomps, Npix)) 
    
    comp_scalings = [list(i) for i in itertools.product([0, 1], repeat=Ncomps)]
    for scaling in scalings:

        #Determine which components to scale
        extra_amps = np.ones(Ncomps)
        scale_factor = inp.scaling_factors[scaling[0]]
        multiplier = scale_factor*inp.scaling_factors[1:]
        multiplier[multiplier==0] = 1.
        extra_amps *= multiplier

        for split in [1,2]:

            maps = []
            for i in range(len(inp.freqs)):
                comp_total = np.sum(np.array([sed_arr[c,i]*extra_amps[c]*comp_maps_unscaled[c] for c in range(Ncomps)]), axis=0)
                maps.append(comp_total + noise_maps[i,split-1])
            
            wt_maps = all_wt_maps[scaling[0], comp_scalings.index(scaling[1:]), split-1]

            NILC_maps = build_NILC_maps(inp, sim, h, wt_maps, freq_maps=maps)
            all_map_level_prop[scaling[0], comp_scalings.index(scaling[1:]), split-1] = NILC_maps

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights

    #define and fill in array of data vectors (dim 0 has size Nscalings for which scaling in inp.scaling_factors is used)
    Clpq_tmp = np.zeros((Nscalings, 2**Ncomps, Ncomps, Ncomps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((Nscalings, 2**Ncomps, Ncomps, Ncomps, inp.Nbins)) #binned
    ells = np.arange(inp.ellmax+1)

    for scaling in scalings:

        for p in range(Ncomps):
            for q in range(Ncomps):
                map1 = all_map_level_prop[scaling[0], comp_scalings.index(scaling[1:]), 0, p]
                map2 = all_map_level_prop[scaling[0], comp_scalings.index(scaling[1:]), 1, q]
                PS = hp.anafast(map1, map2, lmax=inp.ellmax)
                Clpq_tmp[scaling[0], comp_scalings.index(scaling[1:]), p, q] = PS 
                Dl = ells*(ells+1)/2/np.pi*PS
                res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                mean_ells = (res[1][:-1]+res[1][1:])/2
                Clpq[scaling[0], comp_scalings.index(scaling[1:]), p, q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
        
        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights

    return Clpq


def get_scaled_data_vectors_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_scaled_data_vectors

    RETURNS
    -------
    function of *args, get_scaled_data_vectors(sim, inp, env)
    '''
    return get_scaled_data_vectors(*args)


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
    comp_maps_unscaled: (Ncomps, Npix) ndarray of unscaled maps of all components 
    noise_maps_unscaled: (Nfreqs, Nsplits, Npix) ndarray of unscaled maps of noise
    all_wt_maps: (Nsplits, Ncomps, Nscales, Nfreqs, Npix) ndarray containing all weight maps

    '''
    #create temporary directory to place maps
    map_tmpdir = tempfile.mkdtemp(dir=inp.output_dir)

    #array for all weight maps
    Nsplits = 2
    Ncomps = len(inp.comps)
    all_wt_maps = np.zeros((Nsplits, Ncomps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))

    #create frequency maps (GHz) consisting of components and noise. Get power spectra of component maps. 
    comp_spectra, comp_maps_unscaled, noise_maps_unscaled = generate_freq_maps(inp, sim, pars=pars, map_tmpdir=map_tmpdir)
       
    #generate and save files containing frequency maps and then run pyilc
    for split in [1,2]:
        pyilc_tmpdir = setup_pyilc(sim, split, inp, env, map_tmpdir, suppress_printing=True, pars=pars) #set suppress_printing=False to debug pyilc runs
        wt_maps = load_wt_maps(inp, sim, split, pyilc_tmpdir, pars=pars) #load weight maps
        all_wt_maps[split-1] = wt_maps
        shutil.rmtree(pyilc_tmpdir)
    
    shutil.rmtree(map_tmpdir)

    return comp_maps_unscaled, noise_maps_unscaled, all_wt_maps



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
    Clpq: (Ncomps, Ncomps, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        preserved_comps = CMB, ftSZ
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)
    
    Ncomps = len(inp.comps) #components to create NILC maps for
    Nfreqs = len(inp.freqs)
    Npix = 12*inp.nside**2
    Nsplits = 2

    #get needlet filters
    h = GaussianNeedlets(inp)[1]

    # spectral response vectors
    sed_arr = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        if comp == 'tsz':
            sed_arr[c] = tsz_spectral_response(inp.freqs)
        elif comp == 'cib':
            sed_arr[c] = cib_spectral_response(inp.freqs)

    #get maps and weight maps
    comp_maps, noise_maps, all_wt_maps = get_maps_and_wts(sim, inp, env, pars=pars)

    #get map level propagation of components
    all_map_level_prop = np.zeros((Nsplits, Ncomps, Npix)) 

    for split in [1,2]:
        maps = []
        for i in range(len(inp.freqs)):
            comp_total = np.sum(np.array([sed_arr[c,i]*comp_maps[c] for c in range(Ncomps)]), axis=0)
            maps.append(comp_total + noise_maps[i,split-1])
        wt_maps = all_wt_maps[split-1]
        all_map_level_prop[split-1] = build_NILC_maps(inp, sim, h, wt_maps, freq_maps=maps)

    #define and fill in array of data vectors
    ells = np.arange(inp.ellmax+1)
    Clpq_tmp = np.zeros((Ncomps, Ncomps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((Ncomps, Ncomps, inp.Nbins)) #binned
    for p in range(Ncomps):
        for q in range(Ncomps):
            map1 = all_map_level_prop[0, p]
            map2 = all_map_level_prop[1, q]
            Clpq_tmp[p,q] = hp.anafast(map1, map2, lmax=inp.ellmax)
            Dl = ells*(ells+1)/2/np.pi*Clpq_tmp[p,q]
            res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
            mean_ells = (res[1][:-1]+res[1][1:])/2
            Clpq[p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
            
    return Clpq


def get_data_vectors_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_data_vectors

    RETURNS
    -------
    function of *args, get_data_vectors(inp, env, sim=None, pars=None)
    '''
    return get_data_vectors(*args)