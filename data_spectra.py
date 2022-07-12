import numpy as np
import pickle
import healpy as hp
from nilc_power_spectrum_calc import calculate_all_cl


def tsz_spectral_response(freqs): #input frequency in GHz
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    f = 1. #fsky
    response = []
    for freq in freqs:
        x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
        response.append(T_cmb*(x*1/np.tanh(x/2)-4)) #was factor of tcmb microkelvin before
    return np.array(response)

def GaussianNeedlets(ELLMAX, FWHM_arcmin=np.array([600., 60., 30., 15., 10.])):
    # FWHM need to be in strictly decreasing order, otherwise you'll get nonsense
    if ( any( i <= j for i, j in zip(FWHM_arcmin, FWHM_arcmin[1:]))):
        raise AssertionError
    ell = np.arange(ELLMAX+1)
    filters = np.zeros((N_scales, ELLMAX+1))
    N_scales = len(FWHM_arcmin) + 1
    FWHM = FWHM_arcmin * np.pi/(180.*60.)
    # define gaussians
    Gaussians = np.zeros((N_scales-1, ELLMAX+1))
    for i in range(N_scales-1):
        Gaussians[i] = hp.gauss_beam(FWHM[i], lmax=ELLMAX)
    # define needlet filters in harmonic space
    filters[0] = Gaussians[0]
    for i in range(1,N_scales-1):
        filters[i] = np.sqrt(Gaussians[i]**2. - Gaussians[i-1]**2.)
    filters[N_scales-1] = np.sqrt(1. - Gaussians[N_scales-2]**2.)
    # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
    assert (np.absolute( np.sum( filters**2., axis=0 ) - np.ones(ELLMAX+1,dtype=float)) < 1.e-3).all(), "wavelet filter transmission check failed"
    return ell, filters

def get_data_spectra(sim, freqs, Nscales, tsz_amp, ellmax, wigner_file, CC, T, N, verbose):
    wigner = pickle.load(open(wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]
    nfreqs = len(freqs)
    h = GaussianNeedlets(ellmax)[1]
    a = np.array([1., 1.])
    g = tsz_spectral_response(freqs)
    wt_map_spectra = pickle.load(open(f'wt_maps/sim{sim}_wt_map_spectra.p', 'rb')) #[0-2 for TT, Ty, yy][n][m][i][j][l]
    ClTT = np.zeros((3, ellmax+1))
    ClTy = np.zeros((3, ellmax+1))
    Clyy = np.zeros((3, ellmax+1))
    for j in range(3): #ClTT, ClTy, Clyy
        if j==0: #ClTT
            M = wt_map_spectra[0]
            ClTT[0] = CC[:ellmax+1]
            ClTT[1] = calculate_all_cl(nfreqs, ellmax, h, g, T, M, wigner)
            ClTT[2] = calculate_all_cl(nfreqs, ellmax, h, a, N, M, wigner, delta_ij=True)
        elif j==1: #ClTy
            M = wt_map_spectra[1]
            ClTy[0] = calculate_all_cl(nfreqs, ellmax, h, a, CC, M, wigner)
            ClTy[1] = calculate_all_cl(nfreqs, ellmax, h, g, T, M, wigner)
            ClTy[2] = calculate_all_cl(nfreqs, ellmax, h, a, N, M, wigner, delta_ij=True)
        elif j==2: #Clyy
            M = wt_map_spectra[2]
            Clyy[0] = calculate_all_cl(nfreqs, ellmax, h, a, CC, M, wigner)
            Clyy[1] = T[:ellmax+1]
            Clyy[2] = calculate_all_cl(nfreqs, ellmax, h, a, N, M, wigner, delta_ij=True)

    if sim == 0:
        ClTT_array, ClTy_array, Clyy_array = [], [], []
    else:
        ClTT_array = pickle.load(open('power_spectra/clTT.p', 'rb'))
        ClTy_array = pickle.load(open('power_spectra/clTy.p', 'rb'))
        Clyy_array = pickle.load(open('power_spectra/clyy.p', 'rb'))       

    ClTT_array.append(ClTT)
    ClTy_array.append(ClTy)
    Clyy_array.append(Clyy)

    pickle.dump(ClTT_array, open('power_spectra/clTT.p', 'wb'))
    pickle.dump(ClTy_array, open('power_spectra/clTy.p', 'wb'))
    pickle.dump(Clyy_array, open('power_spectra/clyy.p', 'wb'))

    if verbose:
        print('modified files power_spectra/clTT.p, power_spectra/clTy.p, power_spectra/clyy.p')

    return ClTT, ClTy, Clyy


