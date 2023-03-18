#Code adapted from PolyBin O. H. E. Philcox (2023), in prep.

import healpy as hp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def check_bin(inp, bin1, bin2, bin3):
    '''
    Return one if modes in the bin satisfy the even-parity triangle conditions, or zero else.
    This is used either for all triangles in the bin, or just the center of the bin.
    
    ARGUMENTS
    ----------
    inp: Info object, contains input specifications
    bin1, bin2, bin3: int, bin numbers for which to check triangle conditions

    RETURNS
    -------
    int: 1 if center of bin satsifies Wigner 3j triangle relations and 0 otherwise
    '''

    l1 = int((bin1+0.5)*inp.dl_bispectrum)
    l2 = int((bin2+0.5)*inp.dl_bispectrum)
    l3 = int((bin3+0.5)*inp.dl_bispectrum)
    if l3<abs(l1-l2) or l3>l1+l2: return 0
    if l1<abs(l2-l3) or l1>l2+l3: return 0
    if l2<abs(l1-l3) or l2>l1+l3: return 0
    return 1

def safe_divide(x,y):
    """
    Function to divide maps without zero errors.

    ARGUMENTS
    ---------
    x: numpy array containing map (numerator)
    y: numpy array containing map (denominator)

    RETURNS
    -------
    out: numpy array of x/y where y != 0

    """
    out = np.zeros_like(x)
    out[y!=0] = x[y!=0]/y[y!=0]
    return out

def to_map(input_lm, Nside):
    """
    Convert from harmonic-space to map-space

    ARGUMENTS
    ---------
    input_lm: alm in healpy order
    Nside: Nside of map to create

    RETURNS
    -------
    1D numpy array of map in healpix ordering, with specified Nside
    """
    return hp.alm2map(input_lm, Nside, pol=False)

def to_lm(inp, input_map):
    """
    Convert from map-space to harmonic-space

    ARGUMENTS
    ---------
    input_map: 1D numpy array of map in healpix ordering 

    RETURNS
    -------
    alm of map in healpy order, up to 3*Nside-1
    """
    return hp.map2alm(input_map, pol=False, lmax=3*inp.nside-1)


def Bl_norm(inp, Cl1, Cl2, Cl3):
    '''
    Computes bispectrum normalization

    ARGUMENTS
    ---------
    inp: Info() object, contains information about input parameters
    Cl{i}: 1D numpy array, ith map's power spectrum

    RETURNS
    -------
    norm: 3D numpy array, indexed as norm[l1,l2,l3]
    '''

    # Compute normalization matrix
    Nl = inp.ellmax//inp.dl_bispectrum
    Nl_sum = inp.ell_sum_max//inp.dl_bispectrum
    norm = np.zeros((Nl, Nl_sum, Nl_sum), dtype=np.float32)
    for bin1 in range(Nl):
        for bin2 in range(Nl_sum):
            for bin3 in range(Nl_sum):
                # skip bins outside the triangle conditions
                if not check_bin(inp,bin1,bin2,bin3): continue
                value = 0.
                # Now iterate over l bins
                for l1 in range(bin1*inp.dl_bispectrum, (bin1+1)*inp.dl_bispectrum):
                    for l2 in range(bin2*inp.dl_bispectrum, (bin2+1)*inp.dl_bispectrum):
                        for l3 in range(bin3*inp.dl_bispectrum, (bin3+1)*inp.dl_bispectrum):
                            if (-1)**(l1+l2+l3)==-1: continue # 3j = 0 here
                            if l3<abs(l1-l2) or l3>l1+l2: continue
                            if l1<abs(l2-l3) or l1>l2+l3: continue
                            if l2<abs(l1-l3) or l2>l1+l3: continue
                            value += inp.wigner3j[l1,l2,l3]**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi)/Cl1[l1]/Cl2[l2]/Cl3[l3]
                norm[bin1, bin2, bin3] = value

    return norm



def Bl_numerator(inp, data1, data2, data3, Cl1, Cl2, Cl3, equal12=False,equal23=False,equal13=False):
    """
    Compute the numerator of the idealized bispectrum estimator. 
    NB: this doesn't subtract off the disconnected terms, so requires mean-zero maps!
    
    ARGUMENTS
    ---------
    inp: Info() object, contains information about input parameters
    data{i}: 1D numpy array, ith map input to bispectrum
    Cl{i}: 1D numpy array, ith map's power spectrum
    equal{i}{j}: Bool, whether data{i}==data{j}

    RETURNS
    -------
    b_num_ideal: 3D numpy array, indexed as b_num_ideal[l1,l2,l3]
    """

    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data)
    Nl = inp.ellmax//inp.dl_bispectrum
    Nl_sum = inp.ell_sum_max//inp.dl_bispectrum

    Cl1_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl1)
    Cl1_lm = Cl1_interp(l_arr)
    Cl2_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl2)
    Cl2_lm = Cl2_interp(l_arr)
    Cl3_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl3)
    Cl3_lm = Cl3_interp(l_arr)

    # Define pixel area
    Npix = 12*inp.nside**2
    A_pix = 4.*np.pi/Npix
    
    # Define ell bins
    ell_bins = [(l_arr>=inp.dl_bispectrum*bin1)&(l_arr<inp.dl_bispectrum*(bin1+1)) for bin1 in range(Nl)]
    ell_bins_sum = [(l_arr>=inp.dl_bispectrum*bin1)&(l_arr<inp.dl_bispectrum*(bin1+1)) for bin1 in range(Nl_sum)]
    
    # Transform to harmonic space + compute I maps
    I1_map = [to_map(ell_bins[bin1]*safe_divide(to_lm(inp,data1), Cl1_lm), inp.nside) for bin1 in range(Nl)]

    if equal12:
        I2_map = I1_map + [to_map(ell_bins_sum[bin1]*safe_divide(to_lm(inp,data2), Cl2_lm), inp.nside) for bin1 in range(Nl, Nl_sum)]
    else:
        I2_map = [to_map(ell_bins_sum[bin1]*safe_divide(to_lm(inp,data2), Cl2_lm), inp.nside) for bin1 in range(Nl_sum)]
    
    if equal13:
        I3_map = I1_map + [to_map(ell_bins_sum[bin1]*safe_divide(to_lm(inp,data3), Cl3_lm), inp.nside) for bin1 in range(Nl, Nl_sum)]
    elif equal23:
        I3_map = I2_map
    else:
        I3_map = [to_map(ell_bins_sum[bin1]*safe_divide(to_lm(inp,data3), Cl3_lm), inp.nside) for bin1 in range(Nl_sum)]
    
    # Combine to find numerator
    b_num_ideal = np.zeros((Nl, Nl_sum, Nl_sum), dtype=np.float32)
    for bin1 in range(Nl):
        for bin2 in range(Nl_sum):
            for bin3 in range(Nl_sum):
                # skip bins outside the triangle conditions
                if not check_bin(inp,bin1,bin2,bin3): continue
                # compute numerators
                b_num_ideal[bin1,bin2,bin3] = A_pix*np.sum(I1_map[bin1]*I2_map[bin2]*I3_map[bin3])    

    return b_num_ideal



def Bispectrum(inp, data1, data2, data3, equal12=False,equal23=False,equal13=False):
    '''
    Computes bispectrum

    ARGUMENTS
    ---------
    inp: Info() object, contains information about input parameters
    data{i}: 1D numpy array, ith map input to bispectrum (must have mean removed)
    equal{i}{j}: Bool, whether data{i}==data{j}

    RETURNS
    -------
    bl_out: 3D numpy array, indexed as bl_out[bin1,bin2,bin3]
    '''
    Cl1 = hp.anafast(data1, lmax=3*inp.nside-1)
    Cl2 = hp.anafast(data2, lmax=3*inp.nside-1)
    Cl3 = hp.anafast(data3, lmax=3*inp.nside-1)
    bl_norm = Bl_norm(inp, Cl1, Cl2, Cl3)
    bl_out = Bl_numerator(inp, data1, data2, data3, Cl1, Cl2, Cl3, equal12=equal12, equal23=equal23, equal13=equal13)
    # Normalize bispectrum
    bl_out = safe_divide(bl_out, bl_norm)
    return bl_out


