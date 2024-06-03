###################################################################
# This script contains harmonic ILC calculations used when using 
# analytic parameter dependence.
###################################################################

import numpy as np
from scipy import stats
import healpy as hp
from utils import spectral_response
from generate_maps import generate_freq_maps

def get_freq_power_spec(inp, sim=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number (if sim is None, a random simulation number will be used)
    pars: array of floats [Acomp1, etc.] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clij: (Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)

    Ncomps = len(inp.comps)
    Nfreqs = len(inp.freqs)
    Nsplits = 2

    #Create frequency maps (GHz) consisting of sky components and noise. Get power spectra of component maps.
    comp_spectra, comp_maps, noise_maps = generate_freq_maps(inp, sim, save=False, pars=pars)

    #get spectral responses
    all_g_vecs = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        all_g_vecs[c] = spectral_response(inp.freqs, comp)

    #define and fill in array of data vectors
    Clij = np.zeros((Nsplits, Nsplits, Nfreqs, Nfreqs, 1+Ncomps, inp.ellmax+1))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
        for s1 in range(Nsplits):
            for s2 in range(Nsplits):
                map_i = np.sum(np.array([all_g_vecs[c,i]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[i,s1]
                map_j = np.sum(np.array([all_g_vecs[c,j]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[j,s2]
                spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
                Clij[s1,s2,i,j,0] = spectrum
                for y in range(Ncomps):
                    Clij[s1,s2,i,j,1+y] = all_g_vecs[y,i]*all_g_vecs[y,j]*comp_spectra[y]
    
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
    function of *args, get_freq_power_spec(inp, sim=None, pars=None)
    '''
    return get_freq_power_spec(*args)


def get_Rlij_inv(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component
    
    RETURNS
    -------
    Rlij_inv: (Nsplits=2, Nsplits=2, ellmax+1, Nfreqs, Nfreqs) 
        ndarray containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
    Nsplits = 2
    Nfreqs = len(inp.freqs)
    Rlij_inv = np.zeros((Nsplits, Nsplits, inp.ellmax+1, Nfreqs, Nfreqs), dtype=np.float32)
    for s0 in range(Nsplits):
        for s1 in range(Nsplits):
            Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij[s0,s1,:,:,0,:])
            if not inp.delta_l:
                Rlij = Rlij_no_binning
            else:
                Rlij = np.zeros((len(inp.freqs), len(inp.freqs), inp.ellmax+1)) 
                for i in range(Nfreqs):
                    for j in range(Nfreqs):
                        Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*inp.delta_l+1)))[inp.delta_l:inp.ellmax+1+inp.delta_l]
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
    Clij: (Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components

    RETURNS
    -------
    Clpq: (1+Ncomps, ellmax+1) ndarray containing contributions of each component
        to the power spectrum of harmonic ILC map p and harmonic ILC map q
        dim0: index0 is total power spectrum of HILC map p and HILC map q

    '''
    if inp.compute_weights_once:
        Rlij_inv = get_Rlij_inv(inp, inp.Clij_theory)
    else:
        Rlij_inv = get_Rlij_inv(inp, Clij)
    w1, w2 = weights(Rlij_inv, spectral_response, spectral_response2=spectral_response2)
    Clpq = np.einsum('il,jl,ijal->al', w1, w2, Clij[0,1])   
    return Clpq

    

def get_data_vecs(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs, Nfreqs, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component

    RETURNS
    -------
    Clpq: (Ncomps, Ncomps, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component
    '''

    Ncomps = len(inp.comps)
    Nfreqs = len(inp.freqs)

    #get spectral responses
    all_g_vecs = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        all_g_vecs[c] = spectral_response(inp.freqs, comp)

    #HILC auto- and cross-spectra
    Clpq_orig = np.zeros((Ncomps, Ncomps, 1+Ncomps, inp.ellmax+1))
    for p in range(Ncomps):
        for q in range(Ncomps):
            Clpq_orig[p,q] = HILC_spectrum(inp, Clij, all_g_vecs[p], spectral_response2=all_g_vecs[q])
    
    #binning
    Clpq = np.zeros((Ncomps, Ncomps, 1+Ncomps, inp.Nbins))
    ells = np.arange(inp.ellmax+1)
    for p in range(Ncomps):
        for q in range(Ncomps):
            for y in range(1+Ncomps):
                Dl = ells*(ells+1)/2/np.pi*Clpq_orig[p,q,y]
                res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                mean_ells = (res[1][:-1]+res[1][1:])/2
                Clpq[p,q,y] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
    
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
    function of *args, get_data_vecs(inp, Clij)
    '''
    return get_data_vecs(*args)