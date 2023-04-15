import numpy as np
import healpy as hp


def calculate_all_cl(inp, h, g, Clzz, Clw1w2, Clzw=None, w=None, a=None,
    bispectrum_zzw=None, bispectrum_wzw=None, Rho=None, delta_ij=False):
    '''
    Does vectorized power spectrum calculation for one ell

    ARGUMENTS
    ---------
    inp: Info object, contains input specifications
    h: 2D list of needlet filter function values, indexed as h[n][l] where n is needlet filter scale
    g: list of g_i spectral response at different frequencies, indexed as g[i]
    Clzz: 1D numpy array of size ell_sum_max containing component power spectrum
    Clw1w2: 7D array, index as Clw1w2[p,q,n,m,i,j,l] to get cross spectra of weight map pi(n) and qj(m)
    Clzw: 4D numpy array,index as Clzw[p,n,i,l]
    w: 3D numpy array, indexed as w[p,n,i], gives means of weight maps
    a: mean of map
    bispectrum_zzw: indexed as bispectra[q,m,j,l1,l2,l3]
    bispectrum_wzw: indexed as bispectra[p,n,i,q,m,j,l1,l2,l3]
    Rho: indexed as rho[p,n,i,q,m,j,l2,l4,l3,l5,l1]
    delta_ij: True if delta_{ij} is attached to term. False by default

    RETURNS
    -------
    Cl: (N_preserved_comps, N_preserved_comps, ell_max) numpy array of floats 
        containing value of power spectrum from one component's contribution
        at each ell through ellmax, index as Cl[p,q,l]

    INDICES MAPPING IN EINSUM
    -------------------------
    l1, l2, l3, l4, l5 -> l, a, b, c, d

    '''
    wigner = inp.wigner3j[:inp.ellmax+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
    l1 = np.arange(inp.ellmax+1)
    l2 = np.arange(inp.ell_sum_max+1)
    l3 = np.arange(inp.ell_sum_max+1)

    if not delta_ij:
        aa_ww_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,na,ma,a,pqnmijb->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzz,Clw1w2,optimize=True)
        aw_aw_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,na,mb,qmja,pnib->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzw,Clzw,optimize=True)
        w_aaw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,nl,ma,pni,qmjlab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,:inp.ellmax+1],h,w,bispectrum_zzw,optimize=True) \
                    +float(2/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,nl,ma,qmj,pnilab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,:inp.ellmax+1],h,w,bispectrum_zzw,optimize=True)
        a_waw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,n,ma,,pniqmjlab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,0],h,a,bispectrum_wzw,optimize=True) \
                    +float(2/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,m,na,,pniqmjlab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,0],h,a,bispectrum_wzw,optimize=True)
        aaww_term = np.einsum('l,nl,ml,i,j,na,mb,pniqmjacbdl->pql', 1/(2*l1+1),h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,h,h,Rho,optimize=True)
    else:
        aa_ww_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,na,ma,a,pqnmiib->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzz,Clw1w2,optimize=True)
        aw_aw_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,na,mb,qmia,pnib->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzw,Clzw,optimize=True)
        w_aaw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,nl,ma,pni,qmilab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,:inp.ellmax+1],h,w,bispectrum_zzw,optimize=True) \
                    +float(2/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,nl,ma,qmi,pnilab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,:inp.ellmax+1],h,w,bispectrum_zzw,optimize=True)
        a_waw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,n,ma,,pniqmilab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,0],h,a,bispectrum_wzw,optimize=True) \
                    +float(2/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,m,na,,pniqmilab->pql', h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,0],h,a,bispectrum_wzw,optimize=True)
        aaww_term = np.einsum('l,nl,ml,i,i,na,mb,pniqmiacbdl->pql', 1/(2*l1+1),h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],g,g,h,h,Rho,optimize=True)
    
    Cl = aa_ww_term + aw_aw_term + w_aaw_term + a_waw_term + aaww_term
    return Cl, np.array([aa_ww_term, aw_aw_term, w_aaw_term, a_waw_term, aaww_term])