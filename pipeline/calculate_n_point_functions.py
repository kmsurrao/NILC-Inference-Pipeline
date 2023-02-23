import healpy as hp
import numpy as np
import pickle
import warnings
from bispectrum import Bispectrum
from trispectrum import rho
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


def get_cross_spectrum_two_maps(map1, map2, ellmax):
    '''
    ARGUMENTS
    ---------
    map1:
    map2:
    ellmax:

    RETURNS
    -------
    cross power spectrum of two weight maps, padded with zeros to be of length ellmax
    '''
    nside = min(hp.get_nside(map1), hp.get_nside(map2))
    map1 = hp.ud_grade(map1, nside)
    map2 = hp.ud_grade(map2, nside)
    if 3*nside - 1 > ellmax:
        return hp.anafast(map1, map2 = map2, lmax=ellmax)
    else:
        return np.pad(hp.anafast(map1, map2 = map2), (0, ellmax-(3*nside-1)), 'constant', constant_values=(0., 0.))

def get_Clzz(CC, T, N):
    '''
    ARGUMENTS
    ---------
    CC: 1D numpy array of length ell_sum_max containing CMB power spectrum
    T: 1D numpy array of length ell_sum_max containing tSZ power spectrum
    N: 1D numpy array of length ell_sum_max containing noise power spectrum

    RETURNS
    -------
    (3, ell_sum_max) numpy array containing CMB, tSZ, and noise power spectra
    '''
    return np.array([CC,T,N])

def get_Clw1w2(inp, CMB_wt_maps, tSZ_wt_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    CMB_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component CMB, index as CMB_wt_maps[n][i]
    tSZ_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component tSZ, index as tSZ_wt_maps[n][i]

    RETURNS
    -------
    wt_map_power_spectrum: 7D array, index as wt_map_power_spectrum[p,q,n,m,i,j,l] to get cross spectra of weight map pi(n) and qj(m)
    '''
    Nfreqs = 2
    wt_map_power_spectrum = np.full((2, 2, inp.Nscales, inp.Nscales, Nfreqs, Nfreqs, inp.ell_sum_max+1), None)
    for p in range(2): #p=0 corresponds to preserved CMB, p=1 for preserved tSZ
        for q in range(2): #q=0 corresponds to preserved CMB, q=1 for preserved tSZ
            for n in range(inp.Nscales):
                for m in range(inp.Nscales):
                    for i in range(Nfreqs):
                        for j in range(Nfreqs):
                            if wt_map_power_spectrum[p,q,n,m,i,j,0] == None:
                                if p==q==0:
                                    wt_map_power_spectrum[p,q,n,m,i,j] = get_cross_spectrum_two_maps(CMB_wt_maps[n][i], CMB_wt_maps[m][j], inp.ell_sum_max)
                                    wt_map_power_spectrum[p,q,m,n,j,i] = wt_map_power_spectrum[p,q,n,m,i,j]
                                elif p==0 and q==1:
                                    wt_map_power_spectrum[p,q,n,m,i,j] = get_cross_spectrum_two_maps(CMB_wt_maps[n][i], tSZ_wt_maps[m][j], inp.ell_sum_max)
                                    wt_map_power_spectrum[q,p,m,n,j,i] = wt_map_power_spectrum[p,q,n,m,i,j]
                                elif p==q==1:
                                    wt_map_power_spectrum[p,q,n,m,i,j] = get_cross_spectrum_two_maps(tSZ_wt_maps[n][i], tSZ_wt_maps[m][j], inp.ell_sum_max)
                                    wt_map_power_spectrum[p,q,m,n,j,i] = wt_map_power_spectrum[p,q,n,m,i,j]
    wt_map_power_spectrum = wt_map_power_spectrum.astype(np.float32)
    return wt_map_power_spectrum


def get_Clzw(inp, CMB_wt_maps, tSZ_wt_maps, CMB_map, tSZ_map, noise_map):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    CMB_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component CMB, index as CMB_wt_maps[n][i]
    tSZ_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component tSZ, index as tSZ_wt_maps[n][i]
    CMB_map: 1D numpy array, map for CMB in healpix format
    tSZ_map: 1D numpy array, map for tSZ in healpix format
    noise_map: 1D numpy array, map for noise in healpix format

    RETURNS
    -------
    cross_spectra: (N_comps, N_preserved_comps, Nscales, N_freqs, ell_sum_max+1) 5D numpy array,
    index as cross_spectra[z,p,n,i,l]

    '''
    N_comps = 3
    N_preserved_comps = 2
    Nfreqs = 2
    comp_maps = [CMB_map, tSZ_map, noise_map]
    cross_spectra = np.full((N_comps, N_preserved_comps, inp.Nscales, Nfreqs, inp.ell_sum_max+1), None)
    for z in range(N_comps):
        for p in range(N_preserved_comps):
            for n in range(inp.Nscales):
                for i in range(Nfreqs):
                    if p==0:
                        wt_maps = CMB_wt_maps
                    else:
                        wt_maps = tSZ_wt_maps
                    cross_spectra[z,p,n,i] = get_cross_spectrum_two_maps(comp_maps[z], wt_maps[n][i], inp.ell_sum_max)
    return cross_spectra.astype(np.float32)

def get_w(inp, CMB_wt_maps, tSZ_wt_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    CMB_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component CMB, index as CMB_wt_maps[n][i]
    tSZ_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component tSZ, index as tSZ_wt_maps[n][i]

    RETURNS
    -------
    w: (N_preserved_comps, Nscales, N_freqs) 3D numpy array, indexed as w[p,n,i], gives means of weight maps

    '''
    N_preserved_comps = 2
    Nfreqs = 2
    w = np.zeros((N_preserved_comps, inp.Nscales, Nfreqs))
    for p, wt_maps in enumerate([CMB_wt_maps, tSZ_wt_maps]):
        for n in range(inp.Nscales):
            for i in range(Nfreqs):
                w[p,n,i] = np.mean(wt_maps[n][i])
    return w

def get_a(CMB_map, tSZ_map, noise_map):
    '''
    ARGUMENTS
    ---------
    CMB_map: 1D numpy array, map for CMB in healpix format
    tSZ_map: 1D numpy array, map for tSZ in healpix format
    noise_map: 1D numpy array, map for noise in healpix format

    RETURNS
    -------
    a: 1D numpy array of length 3, contains means of CMB, tSZ, and noise maps
    '''
    return np.array([np.mean(CMB_map), np.mean(tSZ_map), np.mean(noise_map)])

def get_bispectrum_zzw(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    CMB_map: 1D numpy array, map for CMB in healpix format
    tSZ_map: 1D numpy array, map for tSZ in healpix format
    noise_map: 1D numpy array, map for noise in healpix format
    CMB_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component CMB, index as CMB_wt_maps[n][i]
    tSZ_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component tSZ, index as tSZ_wt_maps[n][i]

    RETURNS
    -------
    bispectra: indexed as bispectra[z,q,m,j,b1,b2,b3]

    TODO
    optimize by setting ell ranges outside 3*nside-1 to 0 in bispectrum itself
    '''
    N_comps = 3
    N_preserved_comps = 2
    Nfreqs = 2
    comp_maps = [CMB_map, tSZ_map, noise_map]
    wt_maps = [CMB_wt_maps, tSZ_wt_maps]
    Nbins = inp.ellmax//inp.dl_bispectrum
    Nbins_sum = inp.ell_sum_max//inp.dl_bispectrum
    bispectra = np.zeros((N_comps, N_preserved_comps, inp.Nscales, Nfreqs, Nbins, Nbins_sum, Nbins_sum), dtype=np.float32)
    for z in range(N_comps):
        for q in range(N_preserved_comps):
            for m in range(inp.Nscales):
                for j in range(Nfreqs):
                    bispectra[z,q,m,j] = Bispectrum(inp, 
                        comp_maps[z]-np.mean(comp_maps[z]), comp_maps[z]-np.mean(comp_maps[z]), 
                        wt_maps[q][m][j]-np.mean(wt_maps[q][m][j]), equal12=True)
                    nside = min(hp.get_nside(comp_maps[z]), hp.get_nside(wt_maps[q][m][j]))
                    if 3*nside-1 < inp.ell_sum_max:
                        max_bin = (3*nside-1-inp.ellmin)//inp.dl
                        bispectra[z,z,q,m,j,max_bin+1:,max_bin+1:,max_bin+1:].fill(0.)
    return bispectra

def get_bispectrum_wzw(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    CMB_map: 1D numpy array, map for CMB in healpix format
    tSZ_map: 1D numpy array, map for tSZ in healpix format
    noise_map: 1D numpy array, map for noise in healpix format
    CMB_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component CMB, index as CMB_wt_maps[n][i]
    tSZ_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component tSZ, index as tSZ_wt_maps[n][i]

    RETURNS
    -------
    bispectra: indexed as bispectra[p,n,i,z,q,m,j,b1,b2,b3]

    TODO
    optimize by setting ell ranges outside 3*nside-1 to 0 in bispectrum itself
    '''
    N_comps = 3
    N_preserved_comps = 2
    Nfreqs = 2
    comp_maps = [CMB_map, tSZ_map, noise_map]
    wt_maps = [CMB_wt_maps, tSZ_wt_maps]
    Nbins = inp.ellmax//inp.dl_bispectrum
    Nbins_sum = inp.ell_sum_max//inp.dl_bispectrum
    bispectra = np.zeros((N_preserved_comps, inp.Nscales, Nfreqs, N_comps, N_preserved_comps, inp.Nscales, Nfreqs, Nbins, Nbins_sum, Nbins_sum), dtype=np.float32)
    for p in range(N_preserved_comps):
        for n in range(inp.Nscales):
            for i in range(Nfreqs):
                for z in range(N_comps):
                    for q in range(N_preserved_comps):
                        for m in range(inp.Nscales):
                            for j in range(Nfreqs):
                                bispectra[p,n,i,z,q,m,j] = Bispectrum(inp, 
                                    wt_maps[p][n][i]-np.mean(wt_maps[p][n][i]), 
                                    comp_maps[z]-np.mean(comp_maps[z]), wt_maps[q][m][j]-np.mean(wt_maps[q][m][j]))
                                nside = min(hp.get_nside(comp_maps[z]), hp.get_nside(wt_maps[q][m][j]))
                                if 3*nside-1 < inp.ell_sum_max:
                                    max_bin = (3*nside-1-inp.ellmin)//inp.dl
                                    bispectra[z,z,q,m,j,max_bin+1:,max_bin+1:,max_bin+1:].fill(0.)
    return bispectra

def get_rho(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    CMB_map: 1D numpy array, map for CMB in healpix format
    tSZ_map: 1D numpy array, map for tSZ in healpix format
    noise_map: 1D numpy array, map for noise in healpix format
    CMB_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component CMB, index as CMB_wt_maps[n][i]
    tSZ_wt_maps: (Nscales, Nfreqs) list, NILC weight maps for preserved component tSZ, index as tSZ_wt_maps[n][i]

    RETURNS
    -------
    rho: indexed as rho[z,p,n,i,q,m,j,b2,b4,b3,b5,b1]

    TODO
    optimize by setting ell ranges outside 3*nside-1 to 0 in rho itself
    '''
    N_comps = 3
    N_preserved_comps = 2
    Nfreqs = 2
    comp_maps = [CMB_map, tSZ_map, noise_map]
    wt_maps = [CMB_wt_maps, tSZ_wt_maps]
    Nbins = inp.ellmax//inp.dl_bispectrum
    Nbins_sum = inp.ell_sum_max//inp.dl_bispectrum
    Rho = np.zeros((N_comps, N_preserved_comps, inp.Nscales, Nfreqs, N_preserved_comps, inp.Nscales, Nfreqs, Nbins_sum, Nbins_sum, Nbins_sum, Nbins_sum, Nbins), dtype=np.float32)
    for z in range(N_comps):
        for p in range(N_preserved_comps):
            for n in range(inp.Nscales):
                for i in range(Nfreqs):
                    for q in range(N_preserved_comps):
                        for m in range(inp.Nscales):
                            for j in range(Nfreqs):
                                Rho[z,p,n,i,q,m,j] = rho(inp, comp_maps[z]-np.mean(comp_maps[z]), 
                                    wt_maps[p][n][i]-np.mean(wt_maps[p][n][i]), wt_maps[q][m][j]-np.mean(wt_maps[q][m][j]))
                                nside = min(hp.get_nside(comp_maps[z]), hp.get_nside(wt_maps[p][n][i]), hp.get_nside(wt_maps[q][m][j]))
                                if 3*nside-1 < inp.ell_sum_max:
                                    max_bin = (3*nside-1-inp.ellmin)//inp.dl
                                    Rho[z,p,n,i,q,m,j,max_bin+1:,max_bin+1:,max_bin+1:,max_bin+1,max_bin+1].fill(0.)
    return Rho