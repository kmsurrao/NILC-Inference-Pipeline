import numpy as np
from scipy import stats
from utils import tsz_spectral_response

def get_Rlij_inv(inp, Clij, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
    pars: array of [Acmb, Atsz, Anoise1, Anoise2]
    
    RETURNS
    -------
    Rlij_inv: (ellmax+1, Nfreqs=2, Nfreqs=2) ndarray containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
    if pars:
        Rlij_no_binning = np.einsum('l,ijal,a->ijl', prefactor, Clij[:,:,1:,:], pars)
    else:
        Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij[:,:,0,:])
    if not inp.delta_l:
        Rlij = Rlij_no_binning
    else:
        Rlij = np.zeros((len(inp.freqs), len(inp.freqs), inp.ellmax+1)) 
        for i in range(len(inp.freqs)):
            for j in range(len(inp.freqs)):
                Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*inp.delta_l+1)))[inp.delta_l:inp.ellmax+1+inp.delta_l]
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


def HILC_spectrum(inp, Clij, spectral_response, spectral_response2=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components
    pars: array of [Acmb, Atsz, Anoise1, Anoise2]

    RETURNS
    -------
    Clpq: (1+Ncomps, ellmax+1) ndarray containing contributions of each component
        to the power spectrum of harmonic ILC map p and harmonic ILC map q
        dim0: index0 is total power spectrum of HILC map p and HILC map q

    '''
    if inp.compute_weights_once:
        Rlij_inv = get_Rlij_inv(inp, inp.Clij_theory, pars=pars)
    else:
        Rlij_inv = get_Rlij_inv(inp, Clij, pars=pars)
    w1, w2 = weights(Rlij_inv, spectral_response, spectral_response2=spectral_response2)
    if pars:
        Clpq = np.einsum('il,jl,ijal,a->al', w1, w2, Clij, [1]+[par for par in pars])
    else:
        Clpq = np.einsum('il,jl,ijal->al', w1, w2, Clij)   
    return Clpq

def get_data_vecs(inp, Clij, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
    pars: array of [Acmb, Atsz, Anoise1, Anoise2]
        only needed if computing weights individually for each realization 

    RETURNS
    -------
    Clpq: (N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component
    '''

    N_preserved_comps = 2
    Ncomps = 4
    
    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = np.array([1.,0.])
    g_noise2 = np.array([0.,1.])
    all_g_vecs = np.array([g_cmb, g_tsz, g_noise1, g_noise2])

    #HILC auto- and cross-spectra
    Clpq_orig = np.zeros((N_preserved_comps, N_preserved_comps, 1+Ncomps, inp.ellmax+1))
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            Clpq_orig[p,q] = HILC_spectrum(inp, Clij, all_g_vecs[p], spectral_response2=all_g_vecs[q], pars=pars)
    
    #binning
    Clpq = np.zeros((N_preserved_comps, N_preserved_comps, 1+Ncomps, inp.Nbins))
    ells = np.arange(inp.ellmax+1)
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            for y in range(1+Ncomps):
                Dl = ells*(ells+1)/2/np.pi*Clpq_orig[p,q,y]
                res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                mean_ells = (res[1][:-1]+res[1][1:])/2
                Clpq[p,q,y] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
    
    return Clpq