import numpy as np
import healpy as hp


def calculate_all_cl(inp, h, g, Clzz, Clw1w2, Clzw, w, a,
    bispectrum_zzw, bispectrum_wzw, Rho, delta_ij=False):
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
    bispectrum_zzw: indexed as bispectra[q,m,j,b1,b2,b3]
    bispectrum_wzw: indexed as bispectra[p,n,i,q,m,j,b1,b2,b3]
    Rho: indexed as rho[p,n,i,q,m,j,b2,b4,b3,b5,b1]
    delta_ij: True if delta_{ij} is attached to term. False by default

    RETURNS
    -------
    Cl: (N_preserved_comps, N_preserved_comps, ell_max) numpy array of floats 
        containing value of power spectrum from one component's contribution
        at each ell through ellmax, index as Cl[p,q,l]

    INDICES MAPPING IN EINSUM
    -------------------------
    l1, l2, l3, l4, l5 -> l, a, b, c, d
    b1, b2, b3, b4, b5 -> e, f, g, h, k

    '''
    wigner = inp.wigner3j[:inp.ellmax+1, inp.ell_sum_max+1, inp.ell_sum_max+1]
    l1 = np.arange(inp.ellmax+1)
    l2 = np.arange(inp.ell_sum_max+1)
    l3 = np.arange(inp.ell_sum_max+1)

    #Define theta functions where ell_bins[b,l]=1 if l in bin b, 0 otherwise
    ells_sum = np.arange(inp.ell_sum_max+1)
    Nl_sum_bispec = (inp.ell_sum_max-inp.ellmin)//inp.dl_bispectrum
    ell_sum_bins_bispec = [(ells_sum>=inp.ellmin+inp.dl_bispec*bin1)&(ells_sum<inp.ellmin+inp.dl_bispec*(bin1+1)) for bin1 in range(Nl_sum_bispec)]
    ell_sum_bins_bispec[-1,-1] = 1 #put highest ell value in last bin
    Nl_sum_trispec = (inp.ell_sum_max-inp.ellmin)//inp.dl_trispectrum
    ell_sum_bins_trispec = [(ells_sum>=inp.ellmin+inp.dl_trispec*bin1)&(ells_sum<inp.ellmin+inp.dl_trispec*(bin1+1)) for bin1 in range(Nl_sum_trispec)]
    ell_sum_bins_trispec[-1,-1] = 1 #put highest ell value in last bin

    ells = np.arange(inp.ellmax+1)
    Nl_bispec = (inp.ellmax-inp.ellmin)//inp.dl_bispectrum
    ell_bins_bispec = [(ells>=inp.ellmin+inp.dl_bispec*bin1)&(ells<inp.ellmin+inp.dl_bispec*(bin1+1)) for bin1 in range(Nl_bispec)]
    ell_bins_bispec[-1,-1] = 1 #put highest ell value in last bin
    Nl_trispec = (inp.ellmax-inp.ellmin)//inp.dl_trispectrum
    ell_bins_trispec = [(ells>=inp.ellmin+inp.dl_trispec*bin1)&(ells<inp.ellmin+inp.dl_trispec*(bin1+1)) for bin1 in range(Nl_trispec)]
    ell_bins_trispec[-1,-1] = 1 #put highest ell value in last bin


    if not delta_ij:
        aa_ww_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,na,ma,a,pqnmijb->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzz,Clw1w2,optimize=True)
        aw_aw_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,na,mb,pnia,qmjb->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzw,Clzw,optimize=True)
        w_aaw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,nl,mb,pni,qmjefg,el,fa,gb->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,w,bispectrum_zzw,ell_bins_bispec,ell_sum_bins_bispec,ell_sum_bins_bispec,optimize=True)
        a_waw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,j,a,b,lab,lab,n,ma,,pniqmjefg,el,fa,gb->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,0],h,a,bispectrum_wzw,ell_bins_bispec,ell_sum_bins_bispec,ell_sum_bins_bispec,optimize=True)
        aaww_term = np.einsum('l,nl,ml,i,j,na,mb,pniqmjfhgke,el,fa,gb,hc,kd->pql', 1/(2*l1+1),h,h,g,g,h,h,a,Rho,ell_bins_trispec,ell_sum_bins_trispec,ell_sum_bins_trispec,ell_sum_bins_trispec,ell_sum_bins_trispec,optimize=True)
    else:
        aa_ww_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,na,ma,a,pqnmiib->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzz,Clw1w2,optimize=True)
        aw_aw_term = float(1/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,na,mb,pnia,qmib->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,Clzw,Clzw,optimize=True)
        w_aaw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,nl,mb,pni,qmiefg,el,fa,gb->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h,h,w,bispectrum_zzw,ell_bins_bispec,ell_sum_bins_bispec,ell_sum_bins_bispec,optimize=True)
        a_waw_term = float(2/(4*np.pi))*np.einsum('nl,ml,i,i,a,b,lab,lab,n,ma,,pniqmiefg,el,fa,gb->pql', h,h,g,g,2*l2+1,2*l3+1,wigner,wigner,h[:,0],h,a,bispectrum_wzw,ell_bins_bispec,ell_sum_bins_bispec,ell_sum_bins_bispec,optimize=True)
        aaww_term = np.einsum('l,nl,ml,i,i,na,mb,pniqmifhgke,el,fa,gb,hc,kd->pql', 1/(2*l1+1),h,h,g,g,h,h,a,Rho,ell_bins_trispec,ell_sum_bins_trispec,ell_sum_bins_trispec,ell_sum_bins_trispec,ell_sum_bins_trispec,optimize=True)
    
    Cl = aa_ww_term + aw_aw_term + w_aaw_term + a_waw_term + aaww_term
    return Cl, np.array([aa_ww_term, aw_aw_term, w_aaw_term, a_waw_term, aaww_term])