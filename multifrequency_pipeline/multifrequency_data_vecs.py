import sys
import numpy as np
sys.path.append('../shared')
from scipy import stats
import healpy as hp
from generate_maps import generate_freq_maps
from utils import tsz_spectral_response

def get_data_vectors(sim, inp, pars=None):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    pars: array of floats [Acmb, Atsz, Anoise1, Anoise2] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clij: (Nfreqs=2, Nfreqs=2, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
    '''
    Ncomps = 4 #CMB, tSZ, noise 90 nGHz, noise 150 GHz
    Nfreqs = len(inp.freqs)

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, save=False, pars=pars)
    all_spectra_orig = [CC, T, N1, N2]
    all_spectra = []
    ells = np.arange(inp.ellmax+1)
    for Cl in all_spectra_orig:
        Dl = ells*(ells+1)/2/np.pi*Cl
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        mean_ells = (res[1][:-1]+res[1][1:])/2
        all_spectra.append(res[0]/(mean_ells*(mean_ells+1)/2/np.pi))

    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = np.array([1.,0.])
    g_noise2 = np.array([0.,1.])
    all_g_vecs = np.array([g_cmb, g_tsz, g_noise1, g_noise2])

    #define and fill in array of data vectors
    Clij = np.zeros((Nfreqs, Nfreqs, 1+Ncomps, inp.Nbins))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
        map_i = CMB_map + g_tsz[i]*tSZ_map + g_noise1[i]*noise1_map + g_noise2[i]*noise2_map
        map_j = CMB_map + g_tsz[j]*tSZ_map + g_noise1[j]*noise1_map + g_noise2[j]*noise2_map
        spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
        Dl = ells*(ells+1)/2/np.pi*spectrum
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        Clij[i,j,0] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
        for y in range(Ncomps):
            Clij[i,j,1+y] = all_g_vecs[y,i]*all_g_vecs[y,j]*all_spectra[y]
    
    return Clij