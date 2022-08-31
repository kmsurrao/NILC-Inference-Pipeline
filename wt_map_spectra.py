import healpy as hp
import numpy as np
import pickle
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

def load_wt_maps(sim, Nscales, scratch_path, comps=['CMB', 'tSZ']):
    nfreqs = 2
    CMB_wt_maps = [[[],[]] for i in range(Nscales)]
    tSZ_wt_maps = [[[],[]] for i in range(Nscales)]
    for comp in comps:
        for scale in range(Nscales):
            for freq in range(2):
                wt_map_path = f'{scratch_path}/wt_maps/{comp}/{sim}_weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                wt_map = hp.read_map(wt_map_path)
                if comp=='CMB':
                    CMB_wt_maps[scale][freq] = wt_map*10**(-6) #since pyilc outputs CMB map in uK
                else:
                    tSZ_wt_maps[scale][freq] = wt_map
    return CMB_wt_maps, tSZ_wt_maps


def get_wt_map_spectrum_two_maps(comp1_wt_maps, comp2_wt_maps, ellmax, n,m,i,j):
    '''
    returns cross power spectrum of two weight maps, padded with zeros to be of length ellmax
    '''
    map1, map2 = comp1_wt_maps[n][i], comp2_wt_maps[m][j]
    nside = min(hp.get_nside(map1), hp.get_nside(map2))
    map1 = hp.ud_grade(map1, nside)
    map2 = hp.ud_grade(map2, nside)
    if 3*nside - 1 > ellmax:
        return hp.anafast(map1, map2 = map2, lmax=ellmax)
    else:
        return np.pad(hp.anafast(map1, map2 = map2), (0, ellmax-(3*nside-1)), 'constant', constant_values=(0., 0.))




def get_wt_map_spectra(sim, ellmax, Nscales, verbose, scratch_path, comps=['CMB', 'tSZ']):
    '''
    ARGUMENTS
    ---------
    ellmax: int, maximum ell for power spectrum of weight maps

    RETURNS
    -------
    wt_map_power_spectrum: 6D array, index as wt_map_power_spectrum[0-2][n][m][i][j][l] to get cross spectra at different filter scales and frequencies
    '''
    hp.disable_warnings() 
    Nfreqs = 2
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(sim, Nscales, scratch_path, comps)
    wt_map_power_spectrum = np.full((3, Nscales, Nscales, Nfreqs, Nfreqs, ellmax+1), None)
    for c in range(3): #TT, Ty, yy
        for n in range(Nscales):
            for m in range(Nscales):
                for i in range(Nfreqs):
                    for j in range(Nfreqs):
                        if wt_map_power_spectrum[c][n][m][i][j][0] == None:
                            if c==0 and ('CMB' in comps): #TT
                                wt_map_power_spectrum[c][n][m][i][j] = get_wt_map_spectrum_two_maps(CMB_wt_maps, CMB_wt_maps, ellmax, n,m,i,j)
                            elif c==1 and ('CMB' in comps) and ('tSZ' in comps): #Ty
                                wt_map_power_spectrum[c][n][m][i][j] = get_wt_map_spectrum_two_maps(CMB_wt_maps, tSZ_wt_maps, ellmax, n,m,i,j)
                            elif c==2 and ('tSZ' in comps): #yy
                                wt_map_power_spectrum[c][n][m][i][j] = get_wt_map_spectrum_two_maps(tSZ_wt_maps, tSZ_wt_maps, ellmax, n,m,i,j)
                            if (c==0 and ('CMB' in comps)) or (c==2 and ('tSZ' in comps)): #if not Ty spectrum
                                wt_map_power_spectrum[c][m][n][j][i] = wt_map_power_spectrum[c][n][m][i][j]
    wt_map_power_spectrum = wt_map_power_spectrum.astype(np.float32)
    return wt_map_power_spectrum #only makes sense at wt_map_power_spectrum[0] if only CMB comp and wt_map_power_spectrum[2] if only tSZ comp




def get_comp_and_wt_map_cross_spectrum_two_maps(comp_map, wt_maps, ellmax, n, i):
    '''
    returns cross power spectrum of two weight maps, padded with zeros to be of length ellmax
    '''
    map1 = comp_map
    map2 = wt_maps[n][i]
    nside = min(hp.get_nside(map1), hp.get_nside(map2))
    map1 = hp.ud_grade(map1, nside)
    map2 = hp.ud_grade(map2, nside)
    if 3*nside - 1 > ellmax:
        return hp.anafast(map1, map2 = map2, lmax=ellmax)
    else:
        return np.pad(hp.anafast(map1, map2 = map2), (0, ellmax-(3*nside-1)), 'constant', constant_values=(0., 0.))




def get_comp_and_wt_map_cross_spectra(sim, ellmax, Nscales, comp_map, verbose, scratch_path, comp_wt_maps=['CMB', 'tSZ']):
   
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    ellmax: int, maximum ell for power spectrum of weight maps
    Nscales: int, number of needlet filter scales
    comp_map: 1d numpy array, map of component y
    verbose: int, 0 to suppress printing, other number otherwise
    scratch_path: str, location of pyilc weight maps
    comps: list of components whose weight maps to load


    RETURNS
    -------
    W: 4D array of component map and weight map for p cross-spectra, W[p][n][i] to get cross spectra for component y's map and the weight map for component p at filter scale n and frequency i
    '''

    hp.disable_warnings() 
    Nfreqs = 2
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(sim, Nscales, scratch_path, comp_wt_maps)
    W = np.full((2, Nscales, Nfreqs, ellmax+1), None)
    for p in range(2): #CMB wt maps and tSZ wt maps
        for n in range(Nscales):
            for i in range(Nfreqs):
                if p==0 and 'CMB' in comp_wt_maps: #CMB wt maps
                    W[p][n][i] = get_comp_and_wt_map_cross_spectrum_two_maps(comp_map, CMB_wt_maps, ellmax, n, i)
                elif p==1 and 'tSZ' in comp_wt_maps: #tSZ wt maps
                    W[p][n][i] = get_comp_and_wt_map_cross_spectrum_two_maps(comp_map, tSZ_wt_maps, ellmax, n, i)
    W = W.astype(np.float32)
    return W #only makes sense at W[0] if only CMB comp_wt_map and W[1] if only tSZ comp_wt_map