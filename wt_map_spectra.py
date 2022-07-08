import healpy as hp
import numpy as np
import pickle

def load_wt_maps(Nscales, nside):
    CMB_wt_maps = np.zeros((Nscales, 2, 12*nside**2))
    tSZ_wt_maps = np.zeros((Nscales, 2, 12*nside**2))
    for comp in ['CMB', 'tSZ']:
        for scale in range(Nscales):
            for freq in range(2):
                wt_map_path = f'wt_maps/{comp}needletILCweightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                wt_map = hp.read_map(wt_map_path)
                if comp=='CMB:
                    CMB_wt_maps[scale][freq] = wt_map
                else:
                    tSZ_wt_maps[scale][freq] = wt_map
    return CMB_wt_maps, tSZ_wt_maps




def get_wt_map_spectra(sim, ellmax, Nscales, nside, verbose):
    '''
    ARGUMENTS
    ---------
    loop_max: int, maximum ell for power spectrum of weight maps

    RETURNS
    -------
    mask_power_spectrum: 6D array, index as mask_power_spectrum[0-2][n][m][i][j][l] to get cross spectra at different filter scales and frequencies
    '''
    Nfreqs = 2
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(Nscales, nside)
    wt_map_power_spectrum = np.full((3, Nscales, Nscales, Nfreqs, Nfreqs, ellmax), None)
    for c in range(3): #TT, Ty, yy
        for n in range(Nscales):
            for m in range(Nscales):
                for i in range(Nfreqs):
                    for j in range(Nfreqs):
                        if wt_map_power_spectrum[c][n][m][i][j][0] == None:
                            if c==0: #TT
                                spectrum = hp.anafast(CMB_wt_maps[n][i], map2 = CMB_wt_maps[m][j], lmax=ellmax)
                            elif c==1: #Ty
                                spectrum = hp.anafast(CMB_wt_maps[n][i], map2 = tSZ_wt_maps[m][j], lmax=ellmax)
                            else: #yy
                                spectrum = hp.anafast(tSZ_wt_maps[n][i], map2 = tSZ_wt_maps[m][j], lmax=ellmax)
                            wt_map_power_spectrum[c][n][m][i][j] = spectrum
                            if c != 1: #if not Ty spectrum
                                wt_map_power_spectrum[c][m][n][j][i] = spectrum
    wt_map_power_spectrum = wt_map_power_spectrum.astype(np.float32)
    pickle.dump(wt_map_power_spectrum, open(f'wt_maps/sim{sim}_wt_map_spectra.p', "wb"),protocol=4)
    if verbose:
        print(f'created wt_maps/sim{sim}_wt_map_spectra.p')
    return wt_map_power_spectrum