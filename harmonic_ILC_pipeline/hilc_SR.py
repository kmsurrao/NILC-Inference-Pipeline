############################################################################################
# This script contains harmonic ILC calculations used when fitting 
# parameter dependence with symbolic regression.
############################################################################################

import numpy as np
from scipy import stats
import itertools
import healpy as hp
from utils import tsz_spectral_response, cib_spectral_response, get_scalings, sublist_idx
from generate_maps import generate_freq_maps

def get_freq_power_spec(sim, inp):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    Clij: (Nscalings, 2**Ncomps, Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim0: idx i if "scaled" means maps are scaled according to the ith scaling factor from input
        dim1: indices correspond to different combinations of scaled and unscaled components
    '''

    Nfreqs = len(inp.freqs)
    Ncomps = len(inp.comps)
    Nscalings = len(inp.scaling_factors)
    Nsplits = 2
    scalings = get_scalings(inp)
    Clij = np.zeros((Nscalings, 2**Ncomps, Nsplits, Nsplits, Nfreqs, Nfreqs, inp.ellmax+1), dtype=np.float32)

    # spectral response vectors
    sed_arr = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        if comp == 'tsz':
            sed_arr[c] = tsz_spectral_response(inp.freqs)
        elif comp == 'cib':
            sed_arr[c] = cib_spectral_response(inp.freqs)

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T)
    comp_spectra, comp_maps, noise_maps = generate_freq_maps(inp, sim, save=False, scaling=scalings[0])
    
    #fill in array of data vectors
    comp_scalings = [list(i) for i in itertools.product([0, 1], repeat=Ncomps)]
    for scaling in scalings:
        extra_amps = np.ones(Ncomps)
        scale_factor = inp.scaling_factors[scaling[0]]
        multiplier = scale_factor*scaling[1:]
        multiplier[multiplier==0] = 1.
        extra_amps *= multiplier 
        for i in range(Nfreqs):
            for j in range(Nfreqs):
                for s1 in range(Nsplits):
                    for s2 in range(Nsplits):
                        map_i = np.sum(np.array([extra_amps[c]*sed_arr[c,i]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[i,s1]
                        map_j = np.sum(np.array([extra_amps[c]*sed_arr[c,j]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[j,s2]
                        spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
                        Clij[scaling[0], sublist_idx(comp_scalings, scaling[1:]), s1, s2, i, j] = spectrum
        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version for these cases 
    return Clij


def get_freq_power_spec_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_freq_power_spec

    RETURNS
    -------
    function of *args, get_freq_power_spec(sim, inp)
    '''
    return get_freq_power_spec(*args)


def get_Rlij_inv(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, ellmax+1) ndarray 
        containing auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    Rlij_inv: (Nsplits=2, Nsplits=2, ellmax+1, Nfreqs, Nfreqs) ndarray 
        containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
    Nsplits = 2
    Nfreqs = len(inp.freqs)
    Rlij_inv = np.zeros((Nsplits, Nsplits, inp.ellmax+1, Nfreqs, Nfreqs), dtype=np.float32)
    for s0 in range(Nsplits):
        for s1 in range(Nsplits):
            Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij[s0,s1])
            if not inp.delta_l:
                Rlij = Rlij_no_binning
            else:
                Rlij = np.zeros((Nfreqs, Nfreqs, inp.ellmax+1), dtype=np.float32)
                for i in range(Nfreqs):
                    for j in range(Nfreqs):
                        Rlij_no_binning[i][j][:2] = 0
                        convolution_factor = np.ones(2*inp.delta_l+1)
                        if inp.omit_central_ell:
                            convolution_factor[inp.delta_l] = 0
                        Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], convolution_factor))[inp.delta_l:inp.ellmax+1+inp.delta_l]
            Rlij_inv[s0,s1] = np.array([np.linalg.inv(Rlij[:,:,l]) for l in range(inp.ellmax+1)]) 
    return Rlij_inv #index as Rlij_inv[s0,s1,l,i,j]
    

def weights(Rlij_inv, spectral_response, spectral_response2=None):
    '''
    ARGUMENTS
    ---------
    Rlij_inv: (Nsplits=2, Nsplits=2, ellmax+1, Nfreqs, Nfreqs) 
        ndarray containing inverse Rij matrix at each ell
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components
    
    RETURNS
    -------
    w1: (Nfreqs, ellmax+1) ndarray of harmonic ILC weights for split 1 for component with spectral_response SED
    w2: (Nfreqs, ellmax+1) ndarray of harmonic ILC weights for split 2 for component with spectral_response2 SED
        if provided, otherwise for component with spectral_response SED
    '''
    numerator1 = np.einsum('lij,j->il', Rlij_inv[0,0], spectral_response)
    denominator1 = np.einsum('lkm,k,m->l', Rlij_inv[0,0], spectral_response, spectral_response)
    w1 = numerator1/denominator1 #index as w1[i][l]
    if spectral_response2 is None:
        spectral_response2 = spectral_response
    numerator2 = np.einsum('lij,j->il', Rlij_inv[1,1], spectral_response2)
    denominator2 = np.einsum('lkm,k,m->l', Rlij_inv[1,1], spectral_response2, spectral_response2)
    w2 = numerator2/denominator2 #index as w2[i][l]
    return w1, w2


def HILC_spectrum(inp, Clij, spectral_response, spectral_response2=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, ellmax+1) ndarray 
        containing auto- and cross- spectra of freq maps at freqs i and j
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components

    RETURNS
    -------
    Clpq: (ellmax+1) numpy array containing power spectrum of HILC map p and HILC map q

    '''
    if inp.compute_weights_once:
        Rlij_inv = get_Rlij_inv(inp, inp.Clij_theory)
    else:
        Rlij_inv = get_Rlij_inv(inp, Clij)
    w1, w2 = weights(Rlij_inv, spectral_response, spectral_response2=spectral_response2)
    Clpq = np.einsum('il,jl,ijl->l', w1, w2, Clij[0,1])   
    return Clpq


def get_data_vecs(inp, Clij, sim):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nscalings, 2**Ncomps, Nfreqs, Nfreqs, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim0: idx i indicates that "scaled" means maps are scaled according to scaling factor i from input, etc. up to idx Nscalings
        dim1: indices correspond to different combinations of scaled and unscaled components
    sim: int, simulation number

    RETURNS
    -------
    Clpq: (Nscalings, 2**Ncomps, Ncomps, Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim0: idx i indicates that "scaled" means maps are scaled according to scaling factor i from input, etc. up to idx Nscalings
        dim1: indices correspond to different combinations of scaled and unscaled components
    '''

    Ncomps = len(inp.comps)
    Nfreqs = len(inp.freqs)
    Nscalings = len(inp.scaling_factors)
    scalings = get_scalings(inp)
    
    # spectral response vectors
    sed_arr = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        if comp == 'tsz':
            sed_arr[c] = tsz_spectral_response(inp.freqs)
        elif comp == 'cib':     
            sed_arr[c] = cib_spectral_response(inp.freqs)

    #HILC auto- and cross-spectra
    comp_scalings = [list(i) for i in itertools.product([0, 1], repeat=Ncomps)]
    Clpq_orig = np.zeros((Nscalings, 2**Ncomps, Ncomps, Ncomps, inp.ellmax+1), dtype=np.float32)
    for p in range(Ncomps):
        for q in range(Ncomps):
            for s in scalings:
                Clpq_orig[s[0], sublist_idx(comp_scalings, s[1:]), p, q] = HILC_spectrum(inp, Clij[s[0], sublist_idx(comp_scalings, s[1:])], sed_arr[p], spectral_response2=sed_arr[q])
                if sim >= inp.Nsims_for_fits:
                    break
    
    #binning
    Clpq = np.zeros((Nscalings, 2**Ncomps, Ncomps, Ncomps, inp.Nbins), dtype=np.float32)
    ells = np.arange(inp.ellmax+1)
    for s in scalings:
        for p in range(Ncomps):
            for q in range(Ncomps):
                Dl = ells*(ells+1)/2/np.pi*Clpq_orig[s[0],s[1],s[2],p,q]
                res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                mean_ells = (res[1][:-1]+res[1][1:])/2
                Clpq[s[0],sublist_idx(comp_scalings, s[1:]),p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
    
    return Clpq


def get_data_vecs_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_data_vecs

    RETURNS
    -------
    function of *args, get_data_vecs(inp, Clij, sim)
    '''
    return get_data_vecs(*args)

