import numpy as np
import pickle
import healpy as hp
from nilc_power_spectrum_calc import calculate_all_cl_corrected


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

def GaussianNeedlets(ELLMAX, FWHM_arcmin=np.array([600., 60., 30., 15.])):
    # FWHM need to be in strictly decreasing order, otherwise you'll get nonsense
    if ( any( i <= j for i, j in zip(FWHM_arcmin, FWHM_arcmin[1:]))):
        raise AssertionError
    ell = np.arange(ELLMAX+1)
    N_scales = len(FWHM_arcmin) + 1
    filters = np.zeros((N_scales, ELLMAX+1))
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
    taper_width = 200.
    taper_func = (1.0 - 0.5*(np.tanh(0.025*(ell - (ELLMAX - taper_width))) + 1.0)) #smooth taper to zero from ELLMAX-taper_width to ELLMAX
    taper_func *= 0.5*(np.tanh(0.5*(ell-10)))+0.5 #smooth taper to zero for low ell
    for i, filt in enumerate(filters):
        filters[i] = filters[i]*taper_func
    return ell, filters

def get_data_spectra(sim, freqs, Nscales, tsz_amp, ellmax, wigner_zero_m, wigner_nonzero_m, CC, T, N, wt_map_spectra, FWHM_arcmin, scratch_path, verbose):
    nfreqs = len(freqs)
    h = GaussianNeedlets(ellmax, FWHM_arcmin)[1]
    a = np.array([1., 1.])
    g = tsz_spectral_response(freqs)
    ClTT = np.zeros((3, ellmax+1))
    ClTy = np.zeros((3, ellmax+1))
    Clyy = np.zeros((3, ellmax+1))
    for j in range(3): #ClTT, ClTy, Clyy
        if j==0: #ClTT
            M = wt_map_spectra[0]
            ClTT[0] = CC[:ellmax+1]
            ClTT[1] = calculate_all_cl_corrected(nfreqs, ellmax, h, g, T, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m)
            ClTT[2] = calculate_all_cl_corrected(nfreqs, ellmax, h, a, N, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m, delta_ij=True)
        elif j==1: #ClTy
            M = wt_map_spectra[1]
            ClTy[0] = calculate_all_cl_corrected(nfreqs, ellmax, h, a, CC, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m)
            ClTy[1] = calculate_all_cl_corrected(nfreqs, ellmax, h, g, T, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m)
            ClTy[2] = calculate_all_cl_corrected(nfreqs, ellmax, h, a, N, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m, delta_ij=True)
        elif j==2: #Clyy
            M = wt_map_spectra[2]
            Clyy[0] = calculate_all_cl_corrected(nfreqs, ellmax, h, a, CC, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m)
            Clyy[1] = T[:ellmax+1]
            Clyy[2] = calculate_all_cl_corrected(nfreqs, ellmax, h, a, N, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m, delta_ij=True)
    output = np.array([ClTT, ClTy, Clyy]) #has dim (3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)
    pickle.dump(output, open(f'{scratch_path}/power_spectra/{sim}_data_spectra.p', 'wb'), protocol=4)
    if verbose:
        print(f'created {scratch_path}/power_spectra/{sim}_data_spectra.p', flush=True)
    return output 


