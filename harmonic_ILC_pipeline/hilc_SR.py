############################################################################################
# This script contains harmonic ILC calculations used when fitting 
# parameter dependence with symbolic regression.
############################################################################################

import numpy as np
from scipy import stats
import healpy as hp
from utils import tsz_spectral_response, get_scalings
from generate_maps import generate_freq_maps

def get_freq_power_spec(sim, inp):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    Clij: (Nscalings, 2,2,2,2, Nfreqs=2, Nfreqs=2, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: idx0 for unscaled CMB, idx1 for scaled CMB
        dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        dim3: idx0 for unscaled noise90, idx1 for scaled noise90
        dim4: idx0 for unscaled noise150, idx1 for scaled noise150
    '''

    Nfreqs = len(inp.freqs)
    Nscalings = len(inp.scaling_factors)
    scalings = get_scalings(inp)
    Clij = np.zeros((Nscalings, 2,2,2,2, Nfreqs, Nfreqs, inp.ellmax+1), dtype=np.float32)

    #get spectral responses
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = np.array([1.,0.])
    g_noise2 = np.array([0.,1.])

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, save=False, scaling=scalings[0])
    
    #fill in array of data vectors
    for scaling in scalings:
        CMB_amp, tSZ_amp_extra, noise1_amp, noise2_amp = 1,1,1,1
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor
        if scaling[3]: noise1_amp = scale_factor
        if scaling[4]: noise2_amp = scale_factor
        for i in range(Nfreqs):
            for j in range(Nfreqs):
                map_i = CMB_amp*CMB_map + tSZ_amp_extra*g_tsz[i]*tSZ_map + noise1_amp*g_noise1[i]*noise1_map + noise2_amp*g_noise2[i]*noise2_map
                map_j = CMB_amp*CMB_map + tSZ_amp_extra*g_tsz[j]*tSZ_map + noise1_amp*g_noise1[j]*noise1_map + noise2_amp*g_noise2[j]*noise2_map
                spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
                Clij[scaling[0],scaling[1],scaling[2],scaling[3],scaling[4],i,j] = spectrum
        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version for these cases 
    return Clij


def get_Rlij_inv(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nfreqs=2, Nfreqs=2, ellmax+1) ndarray 
        containing auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    Rlij_inv: (ellmax+1, Nfreqs=2, Nfreqs=2) ndarray containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
    Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij)
    if not inp.delta_l:
        Rlij = Rlij_no_binning
    else:
        Rlij = np.zeros((len(inp.freqs), len(inp.freqs), inp.ellmax+1)) 
        for i in range(len(inp.freqs)):
            for j in range(len(inp.freqs)):
                Rlij_no_binning[i][j][:2] = 0
                convolution_factor = np.ones(2*inp.delta_l+1)
                if inp.omit_central_ell:
                    convolution_factor[inp.delta_l] = 0
                Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], convolution_factor))[inp.delta_l:inp.ellmax+1+inp.delta_l]
    Rlij_inv = np.array([np.linalg.inv(Rlij[:,:,l]) for l in range(inp.ellmax+1)]) 
    return Rlij_inv #index as Rlij_inv[l][i][j]
    

def weights(Rlij_inv, spectral_response, spectral_response2=None):
    '''
    ARGUMENTS
    ---------
    Rlij_inv: (ellmax+1, Nfreqs=2, Nfreqs=2) ndarray containing inverse Rij matrix at each ell
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components
    
    RETURNS
    -------
    w, w: w is (Nfreqs, ellmax+1) ndarray of harmonic ILC weights
    if spectral_response2 given: returns w, w2 where w and w2 are HILC weights for different maps
    
    '''
    numerator = np.einsum('lij,j->il', Rlij_inv, spectral_response)
    denominator = np.einsum('lkm,k,m->l', Rlij_inv, spectral_response, spectral_response)
    w = numerator/denominator #index as w[i][l]
    if spectral_response2 is None:
        return w, w
    numerator2 = np.einsum('lij,j->il', Rlij_inv, spectral_response2)
    denominator2 = np.einsum('lkm,k,m->l', Rlij_inv, spectral_response2, spectral_response2)
    w2 = numerator2/denominator2 #index as w[i][l]
    return w, w2


def HILC_spectrum(inp, Clij, spectral_response, spectral_response2=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nfreqs=2, Nfreqs=2, ellmax+1) ndarray 
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
    Clpq = np.einsum('il,jl,ijl->l', w1, w2, Clij)   
    return Clpq


def get_data_vecs(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nscalings, 2,2,2,2, Nfreqs=2, Nfreqs=2, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
              idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: idx0 for unscaled CMB, idx1 for scaled CMB
        dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        dim3: idx0 for unscaled noise90, idx1 for scaled noise90
        dim4: idx0 for unscaled noise150, idx1 for scaled noise150

    RETURNS
    -------
    Clpq: (Nscalings, 2,2,2,2, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
              idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: idx0 for unscaled CMB, idx1 for scaled CMB
        dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        dim3: idx0 for unscaled noise90, idx1 for scaled noise90
        dim4: idx0 for unscaled noise150, idx1 for scaled noise150
    '''

    N_preserved_comps = 2
    Nscalings = len(inp.scaling_factors)
    scalings = get_scalings(inp)
    
    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = np.array([1.,0.])
    g_noise2 = np.array([0.,1.])
    all_g_vecs = np.array([g_cmb, g_tsz, g_noise1, g_noise2])

    #HILC auto- and cross-spectra
    Clpq_orig = np.zeros((Nscalings, 2,2,2,2, N_preserved_comps, N_preserved_comps, inp.ellmax+1), dtype=np.float32)
    for s in scalings:
        for p in range(N_preserved_comps):
            for q in range(N_preserved_comps):
                Clpq_orig[s[0],s[1],s[2],s[3],s[4],p,q] = HILC_spectrum(inp, Clij[s[0],s[1],s[2],s[3],s[4]], all_g_vecs[p], spectral_response2=all_g_vecs[q])
    
    #binning
    Clpq = np.zeros((Nscalings, 2,2,2,2, N_preserved_comps, N_preserved_comps, inp.Nbins), dtype=np.float32)
    ells = np.arange(inp.ellmax+1)
    for s in scalings:
        for p in range(N_preserved_comps):
            for q in range(N_preserved_comps):
                Dl = ells*(ells+1)/2/np.pi*Clpq_orig[s[0],s[1],s[2],s[3],s[4],p,q]
                res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                mean_ells = (res[1][:-1]+res[1][1:])/2
                Clpq[s[0],s[1],s[2],s[3],s[4],p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
    
    return Clpq
