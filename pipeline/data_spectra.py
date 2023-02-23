import numpy as np
import pickle
import healpy as hp
from nilc_power_spectrum_calc import calculate_all_cl
from utils import tsz_spectral_response, GaussianNeedlets



def get_data_spectra(inp, sim, Clzz, Clw1w2, Clzw, w, a, bispectrum_zzw, bispectrum_wzw, Rho):
    '''
    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    sim: int, simulation number
    Clzz: (3, ell_sum_max) 2D numpy array containing component power spectra, index as Clzz[comp, ell] for comp CMB, tSZ, noise
    Clw1w2: 7D array, index as Clw1w2[p,q,n,m,i,j,l] to get cross spectra of weight map pi(n) and qj(m)
    Clzw: (N_comps, N_preserved_comps, N_scales, N_freqs, ell_sum_max) 5D numpy array,
            index as cross_spectra[z,p,n,i,l]
    w: 3D numpy array, indexed as w[p,n,i], gives means of weight maps
    a: 1D numpy array of length 3, contains means of CMB, tSZ, and noise maps
    bispectrum_zzw: indexed as bispectra[z,q,m,j,b1,b2,b3]
    bispectrum_wzw: indexed as bispectra[p,n,i,z,q,m,j,b1,b2,b3]
    Rho: indexed as rho[z,p,n,i,q,m,j,b2,b4,b3,b5,b1]
    '''
    nfreqs = len(inp.freqs)
    h = GaussianNeedlets(inp.ellmax, inp.FWHM_arcmin)[1]
    g_cmb = np.array([1., 1.])
    g_tsz = tsz_spectral_response(inp.freqs)
    N_preserved_comps = 2
    N_comps = 2
    Clpq_array = np.zeros((N_comps, N_preserved_comps, N_preserved_comps, inp.ellmax))
    for z in range(3): #CMB, tSZ, noise
        if z==0:
            Clpq_array[0] = calculate_all_cl(inp, h, g_cmb, Clzz[0], Clw1w2, Clzw[0], w, a[0],
                        bispectrum_zzw[0], bispectrum_wzw[:,:,:,0,:,:,:,:,:,:], Rho[0])
        elif z==1:
            Clpq_array[1] = calculate_all_cl(inp, h, g_tsz, Clzz[1], Clw1w2, Clzw[1], w, a[1],
                        bispectrum_zzw[1], bispectrum_wzw[:,:,:,1,:,:,:,:,:,:], Rho[1])
    
    ClTT = Clpq_array[:,0,0,:] #shape (N_comps, ellmax)
    ClTy = Clpq_array[:,0,1,:]
    Clyy = Clpq_array[:,1,1,:]

    output = np.array([ClTT, ClTy, Clyy]) #has dim (3 for ClTT ClTy Clyy, 2 for CMB tSZ components, ellmax+1)
    pickle.dump(output, open(f'{inp.output_dir}/power_spectra/{sim}_data_spectra.p', 'wb'), protocol=4)
    if inp.verbose:
        print(f'created {inp.output_dir}/power_spectra/{sim}_data_spectra.p', flush=True)
    return output 


