#Code adapted from PolyBin O. H. E. Philcox (2023), in prep.

import healpy as hp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from bispectrum import check_bin, safe_divide, to_map, to_lm


def Tl_numerator(inp, data1, data2, data3, data4,
                Cl1_th, Cl2_th, Cl3_th, Cl4_th, Cl13_th, Cl24_th,
                equal12=False,equal13=False,equal14=False,equal23=False,equal24=False,equal34=False,
                remove_two_point=True):
    """
    Compute the numerator of the idealized trispectrum estimator. 
    Note that we weight according to *spin-zero* Gaunt symbols, which is different to Philcox (in prep.).
    This necessarily subtracts off the disconnected pieces, given input theory Cl_th spectra (plus noise, if appropriate).
    Note that we index the output (if no binning) as t[l1,l2,l3,l4,L] for diagonal momentum L.
    We also drop the L=0 pieces, since these would require non-mean-zero correlators.
   
    ARGUMENTS
    ---------
    inp: Info() object, contains information about input parameters
    data{i}: 1D numpy array, ith map input to trispectrum
    Cl{i}_th: 1D numpy array, auto-spectrum of data{i}
    Cl{i}{j}_th: 1D numpy array, cross-spectrum of data{i} and data{j}
    equal{i}{j}: Bool, whether data{i}==data{j}
    remove_two_point: Bool, whether to subtract two-point disconnected pieces

    RETURNS
    -------
    t4_num_ideal+t2_num_ideal+t0_num_ideal: 5D numpy array, indexed with [b1,b2,b3,b4,bL]
    """

    lmax_data = 3*inp.nside-1
    l_arr,m_arr = hp.Alm.getlm(lmax_data+1)
    Nl = inp.ellmax//inp.dl_trispectrum
    Nl_sum = inp.ell_sum_max//inp.dl_trispectrum

    Cl1_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl1_th)
    Cl1_lm = Cl1_interp(l_arr)
    Cl2_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl2_th)
    Cl2_lm = Cl2_interp(l_arr)
    Cl3_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl3_th)
    Cl3_lm = Cl3_interp(l_arr)
    Cl4_interp = InterpolatedUnivariateSpline(np.arange(lmax_data+1),Cl4_th)
    Cl4_lm = Cl4_interp(l_arr)
    
    # Define ell bins
    ell_bins = [(l_arr>=inp.dl_trispectrum*bin1)&(l_arr<inp.dl_trispectrum*(bin1+1)) for bin1 in range(Nl_sum)]
    
    ## Transform to harmonic space + compute I maps
    if inp.verbose: print("Computing I^a maps")
    
    # Map 1
    I1_map = [to_map(ell_bins[bin1]*safe_divide(to_lm(inp,data1), Cl1_lm), inp.nside) for bin1 in range(Nl_sum)]
    
    # Map 2
    if equal12:
        I2_map = I1_map
    else:
        I2_map = [to_map(ell_bins[bin2]*safe_divide(to_lm(inp,data2), Cl2_lm), inp.nside) for bin2 in range(Nl_sum)]

    # Map 3
    if equal13:
        I3_map = I1_map
    elif equal23:
        I3_map = I2_map
    else:
        I3_map = [to_map(ell_bins[bin3]*safe_divide(to_lm(inp,data3), Cl3_lm), inp.nside) for bin3 in range(Nl_sum)]
    
    # Map 4
    if equal14:
        I4_map = I1_map
    elif equal24:
        I4_map = I2_map
    elif equal34:
        I4_map = I3_map
    else:
        I4_map = [to_map(ell_bins[bin4]*safe_divide(to_lm(inp,data4), Cl4_lm), inp.nside) for bin4 in range(Nl_sum)]
    
    ## Define maps of A^{ab}_lm = int[dn Y_lm(n) I^a(n)I^b(n)] for two I maps
    if inp.verbose: print("Computing A^{ab} maps")
    A12_lm = [[to_lm(inp,I1_map[b1]*I2_map[b2]) for b2 in range(Nl_sum)] for b1 in range(Nl_sum)]
    A34_lm = [[to_lm(inp,I3_map[b3]*I4_map[b4]) for b4 in range(Nl_sum)] for b3 in range(Nl_sum)]
    
    # Create output arrays (for 4-field, 2-field and 0-field terms)
    t4_num_ideal = np.zeros((Nl_sum,Nl_sum,Nl_sum,Nl_sum,Nl), dtype=np.float32)
    t2_num_ideal = np.zeros_like(t4_num_ideal, dtype=np.float32)
    t0_num_ideal = np.zeros_like(t4_num_ideal, dtype=np.float32)
    
    ## Compute four-field term
    if inp.verbose: print("Computing four-field term")
    
    # Iterate over bins
    for b1 in range(Nl_sum):
        for b2 in range(Nl_sum):
            for b3 in range(Nl_sum):
                for b4 in range(Nl_sum):
                    # Iterate over L bins
                    for bL in range(Nl):
                        # skip bins outside the triangle conditions
                        if not check_bin(inp,b1,b2,bL): continue
                        if not check_bin(inp,b3,b4,bL): continue

                        # Compute four-field term
                        summand = A12_lm[b1][b2]*A34_lm[b3][b4].conj()
                        t4_num_ideal[b1,b2,b3,b4,bL] = np.sum(summand*(ell_bins[bL])*(1.+(m_arr>0))).real/np.sum(ell_bins[0])
    
    if not remove_two_point:
        return t4_num_ideal

    ## Compute two-field term
    if inp.verbose: print("Computing two-field and zero-field terms")
    
    # Compute empirical power spectra
    Cl13 = hp.anafast(data1, data3, lmax=inp.ell_sum_max)
    Cl24 = hp.anafast(data2, data4, lmax=inp.ell_sum_max)

    # Iterate over bins
    for b1 in range(Nl_sum):
        for b2 in range(Nl_sum):
            for b3 in range(Nl_sum):
                for b4 in range(Nl_sum):
                        
                        # second permutation (only one that's relevant here)
                        if (b1==b3 and b2==b4):
                            for bL in range(Nl):
                                if not check_bin(inp,b1,b2,bL) or not check_bin(inp,b3,b4,bL): continue
                                for l1 in range(b1*inp.dl_trispectrum, (b1+1)*inp.dl_trispectrum):
                                    l3 = l1
                                    for l2 in range(b2*inp.dl_trispectrum, (b2+1)*inp.dl_trispectrum):
                                        l4 = l2
                                        for L in range(bL*inp.dl_trispectrum, (bL+1)*inp.dl_trispectrum):
                                            if L<abs(l1-l2) or L>l1+l2: continue
                                            if L<abs(l3-l4) or L>l3+l4: continue
                                            if (-1)**(l1+l2+L)==-1: continue # drop parity-odd modes
                                            if (-1)**(l3+l4+L)==-1: continue 

                                            # Compute two-field term
                                            prefactor = (2*l1+1)*(2*l2+1)*(2*L+1)/(4.*np.pi)*inp.wigner3j[l1,l2,L]**2
                                            t2_num_ideal[b1,b2,b3,b4,bL] += -prefactor*(Cl13_th[l1]*Cl24[l2]+Cl13[l1]*Cl24_th[l2])
                                            t0_num_ideal[b1,b2,b3,b4,bL] += prefactor*Cl13_th[l1]*Cl24_th[l2]
    t2_num_ideal /= inp.dl_trispectrum**5
    t0_num_ideal /= inp.dl_trispectrum**5

    return t4_num_ideal+t2_num_ideal+t0_num_ideal

def rho(inp, a_map, w1_map, w2_map, remove_two_point=True):
    '''
    Compute trispectrum without normalization
    PARAMETERS
    inp: Info() object, contains information about input parameters
    a_map: 1D numpy array, map of signal with average subtracted
    w1_map: 1D numpy array, map of first weight map with average subtracted
    w2_map: 1D numpy array, map of second weight map with average subtracted
    remove_two_point: Bool, whether to subtract two-point disconnected pieces
    
    RETURNS
    tl_out: 5D numpy array, indexed as tl_out[l2,l4,l3,l5,l1]

    '''
    Cl_aa = hp.anafast(a_map, lmax=inp.ell_sum_max)
    Cl_w1w1 = hp.anafast(w1_map, lmax=inp.ell_sum_max)
    Cl_w2w2 = hp.anafast(w2_map, lmax=inp.ell_sum_max)
    Cl_w1w2 = hp.anafast(w1_map, w2_map, lmax=inp.ell_sum_max)
    tl_out = Tl_numerator(inp,a_map,w1_map,a_map,w2_map,
                          Cl_aa, Cl_w1w1, Cl_aa, Cl_w2w2,
                          Cl_aa, Cl_w1w2, equal13=True, 
                          remove_two_point=remove_two_point)
    return tl_out