import numpy as np

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
        Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,nl,ml,np,mp,i,j,nmijq->l',2*l2+1,2*l3+1,wigner,wigner,cl,h[:,:ellmax+1],h[:,:ellmax+1],h,h,a,a,M,optimize=True)
    else:
        Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,nl,ml,np,mp,i,i,nmiiq->l',2*l2+1,2*l3+1,wigner,wigner,cl,h[:,:ellmax+1],h[:,:ellmax+1],h,h,a,a,M,optimize=True)
    return Cl



def calculate_all_cl_corrected(nfreqs, ellmax, h, g, cl, M, Wp, Wq, wigner_zero_m, wigner_nonzero_m, delta_ij=False):
    '''
    Does vectorized power spectrum calculation for one ell

    ARGUMENTS
    ---------
    nfreqs: int, number of frequency channels
    ellmax: int, max ell for which you want to calculate Cl
    h: 2D list of needlet filter function values, indexed as h[n][l] where n is needlet filter scale
    g: list of g_i spectral response at different frequencies, indexed as g[i]
    cl: list of power spectrum values of contaminating component, indexed as cl[l]
    M: 5D array of weight map cross-spectra, M[n][m][i][j] to get cross spectra at filter scales n,m and frequencies i,j
    Wp: 3D array of component map, weight map for p cross-spectra, W[n][i] to get cross spectra for component p weight map at filter scale n and frequency i
    Wq: 3D array of component map, weight map for q cross-spectra, W[n][i] to get cross spectra for component q weight map at filter scale n and frequency i
    wigner_zero_m: 3D array of wigner3j symbols with m1=m2=m3=0, index by wigner[l1][l2][l3]
    wigner_nonzero_m: 3D array of wigner3j symbols of form (l1 l2 l2 0 -m2 m2), index by wigner[l1][l2][m2]. Dim (ellmax+1, ellmax+1, 2*ellmax+1)
    delta_ij: True if delta_{ij} is attached to term. False by default

    RETURNS
    -------
    Cl: numpy array of floats containing value of power spectrum at each ell through ellmax

    INDICES MAPPING IN EINSUM
    -------------------------
    l1, l2, l3 -> l, a, b
    m2, m3 -> c, d

    '''
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]

    m = np.ones(2*ellmax+1) #acount for (-1)^{m_2+m_3} factor in term3
    zero_idx = ellmax
    for i in range(ellmax):
        if abs(i-zero_idx)%2==0:
            m.append(1)
        else:
            m.append(-1)
    m = np.array(m)


    if not delta_ij:
        term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,nl,ml,na,ma,i,j,nmijb->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,cl,h[:,:ellmax+1],h[:,:ellmax+1],h,h,g,g,M,optimize=True)
        term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,nl,ml,na,mb,i,j,mja,nib->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,h[:,:ellmax+1],h[:,:ellmax+1],h,h,g,g,Wq,Wp,optimize=True)
        term3 = float(1/(4*np.pi))*np.einsum('a,b,laa,lbb,lac,lbd,nl,ml,na,mb,i,j,nia,mjb,c,d->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,wigner_nonzero_m,wigner_nonzero_m,h[:,:ellmax+1],h[:,:ellmax+1],h,h,g,g,Wp,Wq,m,m,optimize=True)
    else:
        term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,nl,ml,na,ma,i,i,nmiib->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,cl,h[:,:ellmax+1],h[:,:ellmax+1],h,h,g,g,M,optimize=True)
        term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,nl,ml,na,mb,i,i,mia,nib->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,h[:,:ellmax+1],h[:,:ellmax+1],h,h,g,g,Wq,Wp,optimize=True)
        term3 = float(1/(4*np.pi))*np.einsum('a,b,laa,lbb,lac,lbd,nl,ml,na,mb,i,i,nia,mib,c,d->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,wigner_nonzero_m,wigner_nonzero_m,h[:,:ellmax+1],h[:,:ellmax+1],h,h,g,g,Wp,Wq,m,m,optimize=True)
    Cl = term1 + term2 + term3
    return Cl