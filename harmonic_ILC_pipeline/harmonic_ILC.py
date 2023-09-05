import numpy as np

def get_Rlij_inv(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
    
    RETURNS
    -------
    Rlij_inv: (ellmax+1, Nfreqs=2, Nfreqs=2) ndarray containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
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


def HILC_spectrum(inp, Clij, spectral_response, spectral_response2=None):
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

    RETURNS
    -------
    Clpq: (1+Ncomps, ellmax+1) ndarray containing contributions of each component
        to the power spectrum of harmonic ILC map p and harmonic ILC map q
        dim0: index0 is total power spectrum of HILC map p and HILC map q

    '''
    if inp.compute_weights_once:
        Rlij_inv = get_Rlij_inv(inp, inp.Clij_data)
    else:
        Rlij_inv = get_Rlij_inv(inp, Clij)
    w1, w2 = weights(Rlij_inv, spectral_response, spectral_response2=spectral_response2)
    Clpq = np.einsum('il,jl,ijal->al', w1, w2, Clij) 
    return Clpq