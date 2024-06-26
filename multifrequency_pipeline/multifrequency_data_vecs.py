import sys
import numpy as np
sys.path.append('../shared')
from scipy import stats
import healpy as hp
from generate_maps import generate_freq_maps
from utils import spectral_response

def get_data_vectors(inp, sim=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number (if sim is None, a random simulation number will be used)
    pars: array of floats [Acomp1, Acomp2, etc.] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clij: (Nfreqs, Nfreqs, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)

    Ncomps = len(inp.comps)
    Nfreqs = len(inp.freqs)

    #Create frequency maps (GHz) consisting of sky components and noise. Get power spectra of component maps.
    comp_spectra_orig, comp_maps, noise_maps = generate_freq_maps(inp, sim, save=False, pars=pars)
    comp_spectra = []
    ells = np.arange(inp.ellmax+1)
    for Cl in comp_spectra_orig:
        Dl = ells*(ells+1)/2/np.pi*Cl
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        mean_ells = (res[1][:-1]+res[1][1:])/2
        comp_spectra.append(res[0]/(mean_ells*(mean_ells+1)/2/np.pi))

    #get spectral responses
    all_g_vecs = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        all_g_vecs[c] = spectral_response(inp.freqs, comp)

    #define and fill in array of data vectors
    Clij = np.zeros((Nfreqs, Nfreqs, 1+Ncomps, inp.Nbins))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
        map_i = np.sum(np.array([all_g_vecs[c,i]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[i,0]
        map_j = np.sum(np.array([all_g_vecs[c,j]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[j,1]
        spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
        Dl = ells*(ells+1)/2/np.pi*spectrum
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        Clij[i,j,0] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
        for y in range(Ncomps):
            Clij[i,j,1+y] = all_g_vecs[y,i]*all_g_vecs[y,j]*comp_spectra[y]
    
    return Clij


def get_data_vectors_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_data_vectors

    RETURNS
    -------
    function of *args, get_data_vectors(inp, sim=None, pars=None)
    '''
    return get_data_vectors(*args)