import numpy as np
import mpmath
import multiprocessing as mp
import pickle


def calculate_all_cl(nfreqs, ellmax, h, a, cl, M, wigner, delta_ij=False):
    '''
    Does vectorized power spectrum calculation for one ell

    ARGUMENTS
    ---------
    nfreqs: int, number of frequency channels
    ellmax: int, max ell for which you want to calculate Cl
    h: 2D list of needlet filter function values, indexed as h[n][l] where n is needlet filter scale
    a: list of a_i spectral response at different frequencies, indexed as a[i]
    cl: list of power spectrum values of contaminating component, indexed as cl[l]
    M: 5D array of mask cross-spectra, M[n][m][i][j] to get cross spectra at filter scales n,m and frequencies i,j
    wigner: 3D array of wigner3j symbols, index by wigner[l1][l2][l3]
    delta_ij: True if delta_{ij} is attached to term. False by default

    RETURNS
    -------
    Cl: numpy array of floats containing value of power spectrum at each ell through ellmax
    '''
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]
    if not delta_ij:
        Cl = float(1/(4*mpmath.pi))*np.einsum('p,q,lpq,lpq,p,nl,ml,np,mp,i,j,nmijq->l',2*l2+1,2*l3+1,wigner,wigner,cl,h[:,ellmax+1],h[:,:ellmax+1],h,h,a,a,M,optimize=True)
    else:
        Cl = float(1/(4*mpmath.pi))*np.einsum('p,q,lpq,lpq,p,nl,ml,np,mp,i,i,nmiiq->l',2*l2+1,2*l3+1,wigner,wigner,cl,h[:,:ellmax+1],h[:,:ellmax+1],h,h,a,a,M,optimize=True)
    return Cl



