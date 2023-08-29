import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import sys
sys.path.append('../shared')
sys.path.append('../template_fitting_pipeline')
from input import Info
from generate_maps import generate_freq_maps
from utils import tsz_spectral_response


def get_cov(inp):
  '''
  Demonstrate equality of template-fitting and harmonic ILC with data-split cross-spectra

  ARGUMENTS
  ---------
  inp: Info object containing input parameter specifications

  RETURNS
  -------
  FinalCovILC: (2, 2, ellmax+1) ndarray containing parameter covariance matrix from harmonic ILC
  FinalCovTemplateFitting: (2, 2, ellmax+1) ndarray containing parameter covariance matrix from template-fitting
  '''


  CC_all, T_all, N1Split1_all, N2Split1_all, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(0, inp, save=False)
  CC_all, T_all, N1Split2_all, N2Split2_all, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(1, inp, save=False)


  #define parameters
  f = 1.0 #f_sky
  g1, g2 = tsz_spectral_response([90., 150.]) #frequency dependence at 90 and 150 GHz
  ells = np.arange(2, inp.ellmax+1)

  #set up arrays
  CMB_var_ILC = []
  CMB_tSZ_covar_ILC = []
  tSZ_var_ILC = []
  CMB_var_template = []
  CMB_tSZ_covar_template = []
  tSZ_var_template = []

  for ell in ells:
    CC = CC_all[ell]
    T = T_all[ell]
    N1Split1 = N1Split1_all[ell]
    N2Split1 = N2Split1_all[ell]
    N1Split2 = N1Split2_all[ell]
    N2Split2 = N2Split2_all[ell]

    '''
    Define matrix of auto- and cross-frequency power spectra at some multipole ell.
    Here, CC = C_ell**CMB , N1 = N_ell**{freq1} , N2 = N_ell**{freq2} ,
    g1 = tSZ spectral function at freq1, g2 = tSZ spectral function at freq2,
    T = C_ell**yy (tSZ power spectrum in frequency-independent Compton-y units).
    Account here for the fact that we have to make two independent ILC maps from separate
    subsets of the data; to be fully general let's allow them to each
    have their own arbitrary noise power spectra (later we can specialize
    to the case where they are equal).
    '''
    ClijSplit1 = [[CC + N1Split1 + g1**2*T, CC + g1*g2*T], [CC + g1*g2*T, CC + N2Split1 + g2**2*T]]
    ClijSplit2 = [[CC + N1Split2 + g1**2*T, CC + g1*g2*T], [CC + g1*g2*T, CC + N2Split2 + g2**2*T]]
    ClijSplit1Inv = np.linalg.inv(ClijSplit1)
    ClijSplit2Inv = np.linalg.inv(ClijSplit2)

    '''
    also define the 'cross' Clij, which has no noise as it comes from cross-power of two independent subsets of the data
    '''
    ClijNoNoise = [[CC + g1**2*T, CC + g1*g2*T], [CC + g1*g2*T, CC + g2**2*T]]
    gvec = [g1,g2]
    hvec = [1,1]

    '''
    Get the ILC weights for the y -maps and the CMB maps explicitly
    '''
    weightsILCySplit1 = (gvec @ ClijSplit1Inv)/(gvec @ ClijSplit1Inv @ gvec)
    weightsILCySplit2 = (gvec @ ClijSplit2Inv)/(gvec @ ClijSplit2Inv @ gvec)
    weightsILCCMBSplit1 = (hvec @ ClijSplit1Inv)/(hvec @ ClijSplit1Inv @ hvec)
    weightsILCCMBSplit2 = (hvec @ ClijSplit2Inv)/(hvec @ ClijSplit2Inv @ hvec)

    '''
    Find the total power at this ell in each independent map from our two data subsets.
    '''
    ClyySplit1 = 1/(gvec @ ClijSplit1Inv @ gvec)
    ClyySplit2 = 1/(gvec @ ClijSplit2Inv @ gvec)
    ClTTSplit1 = 1/(hvec @ ClijSplit1Inv @ hvec)
    ClTTSplit2 = 1/(hvec @ ClijSplit2Inv @ hvec)
    ClTySplit1 = (gvec @ ClijSplit1Inv @ hvec)/((gvec @ ClijSplit1Inv @ gvec)*(hvec @ ClijSplit1Inv @ hvec))
    ClTySplit2 = (gvec @ ClijSplit2Inv @ hvec)/((gvec @ ClijSplit2Inv @ gvec)*(hvec @ ClijSplit2Inv @ hvec))

    '''
    Now we need the cross-power spectrum of ymap1 and ymap2, Tmap1 and Tmap2, etc
    '''
    ClijNoNoiseInv = np.linalg.inv(ClijNoNoise)
    ClyyCross = weightsILCySplit1 @ ClijNoNoise @ weightsILCySplit2
    ClTTCross = weightsILCCMBSplit1 @ ClijNoNoise @ weightsILCCMBSplit2
    ClTSplit1ySplit2Cross = weightsILCCMBSplit1 @ ClijNoNoise @ weightsILCySplit2
    ClTSplit2ySplit1Cross = weightsILCCMBSplit2 @ ClijNoNoise @ weightsILCySplit1

    '''
    First, construct the covariance matrix of the auto- and cross-frequency power spectra.
    Explicitly include 150x90 terms in power spectrum covariance matrix, i.e., it is a 4x4 matrix.
    '''
    Nmodes = f*(2*ell+1)
    PScovFull = (1/
        Nmodes)*np.array([[2*(CC + g1**2*T) + (CC + g1**2*T)*(N1Split2 +
            N1Split1) + N1Split1*N1Split2,
        2*(CC + g1**2*T)*(CC + g1*g2*T) + (CC + g1*g2*T)*N1Split1,
        2*(CC + g1*g2*T)*(CC + g1**2*T) + (CC + g1*g2*T)*N1Split2,
        2*(CC + g1*g2*T)**2],
        [2*(CC + g1**2*T)*(CC + g1*g2*T) + (CC + g1*g2*T)*N1Split1,
        (CC + g1**2*T)*(CC + g2**2*T) + (CC + g1*g2*T)**2 + (CC + g1**2*T)*
          N2Split2 + (CC + g2**2*T)*N1Split1 + N1Split1*N2Split2,
        (CC + g1*g2*T)**2 + (CC + g1**2*T)*(CC + g2**2*T),
        2*(CC + g1*g2*T)*(CC + g2**2*T) + (CC + g1*g2*T)*N2Split2],
        [2*(CC + g1*g2*T)*(CC + g1**2*T) + (CC + g1*g2*T)*N1Split2,
        (CC + g1*g2*T)**2 + (CC + g1**2*T)*(CC + g2**2*T),
        (CC + g2**2*T)*(CC + g1**2*T) + (CC + g1*g2*T)**2 + (CC + g2**2*T)*
          N1Split2 + (CC + g1**2*T)*N2Split1 + N2Split1*N1Split2,
        2*(CC + g2**2*T)*(CC + g1*g2*T) + (CC + g1*g2*T)*N2Split1],
        [2*(CC + g1*g2*T)**2,
        2*(CC + g1*g2*T)*(CC + g2**2*T) + (CC + g1*g2*T)*N2Split2,
        2*(CC + g2**2*T)*(CC + g1*g2*T) + (CC + g1*g2*T)*N2Split1,
        2*(CC + g2**2*T)**2 + (CC + g2**2*T)*(N2Split2 + N2Split1) +
          N2Split1*N2Split2]])

    '''
    We are really representing a four-index tensor, each of whose indices can take 2 values, as a
    4x4 matrix -- both objects contain 16 elements. To make the summation in Eq. 87 straightforward, let's
    just construct the four-index tensor explicitly.
    '''
    PScovFullTensor = np.zeros((2,2,2,2))
    PScovFullTensor[0, 0, 0, 0] = PScovFull[0, 0]
    PScovFullTensor[0, 0, 0, 1] = PScovFull[0, 1]
    PScovFullTensor[0, 0, 1, 0] = PScovFull[0, 2]
    PScovFullTensor[0, 0, 1, 1] = PScovFull[0, 3]
    PScovFullTensor[0, 1, 0, 0] = PScovFull[1, 0]
    PScovFullTensor[0, 1, 0, 1] = PScovFull[1, 1]
    PScovFullTensor[0, 1, 1, 0] = PScovFull[1, 2]
    PScovFullTensor[0, 1, 1, 1] = PScovFull[1, 3]
    PScovFullTensor[1, 0, 0, 0] = PScovFull[2, 0]
    PScovFullTensor[1, 0, 0, 1] = PScovFull[2, 1]
    PScovFullTensor[1, 0, 1, 0] = PScovFull[2, 2]
    PScovFullTensor[1, 0, 1, 1] = PScovFull[2, 3]
    PScovFullTensor[1, 1, 0, 0] = PScovFull[3, 0]
    PScovFullTensor[1, 1, 0, 1] = PScovFull[3, 1]
    PScovFullTensor[1, 1, 1, 0] = PScovFull[3, 2]
    PScovFullTensor[1, 1, 1, 1] = PScovFull[3, 3]

    '''
    Compute the sums involing weights and PScov
    '''
    CovClTTClTT = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCCMBSplit2, weightsILCCMBSplit1, weightsILCCMBSplit2, PScovFullTensor)
    CovClTTClTy = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCCMBSplit2, weightsILCCMBSplit1, weightsILCySplit2, PScovFullTensor)
    CovClTTClyT = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCCMBSplit2, weightsILCySplit1, weightsILCCMBSplit2, PScovFullTensor)
    CovClTTClyy = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCCMBSplit2, weightsILCySplit1, weightsILCySplit2, PScovFullTensor)
    CovClTyClTT = CovClTTClTy
    CovClTyClTy = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCySplit2, weightsILCCMBSplit1, weightsILCySplit2, PScovFullTensor)
    CovClTyClyT = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCySplit2, weightsILCySplit1, weightsILCCMBSplit2, PScovFullTensor)
    CovClTyClyy = np.einsum('i,j,k,l,ijkl->', weightsILCCMBSplit1, weightsILCySplit2, weightsILCySplit1, weightsILCySplit2, PScovFullTensor)
    CovClyTClTT = CovClTTClyT
    CovClyTClTy = CovClTyClyT
    CovClyTClyT = np.einsum('i,j,k,l,ijkl->', weightsILCySplit1, weightsILCCMBSplit2, weightsILCySplit1, weightsILCCMBSplit2, PScovFullTensor)
    CovClyTClyy = np.einsum('i,j,k,l,ijkl->', weightsILCySplit1, weightsILCCMBSplit2, weightsILCySplit1, weightsILCySplit2, PScovFullTensor)
    CovClyyClTT = CovClTTClyy
    CovClyyClTy = CovClTyClyy
    CovClyyClyT = CovClyTClyy
    CovClyyClyy = np.einsum('i,j,k,l,ijkl->', weightsILCySplit1, weightsILCySplit2, weightsILCySplit1, weightsILCySplit2, PScovFullTensor)


    '''
    Construct Cov_ {ab,cd} (matrix where Cov_ {ab, cd} = sigma^2 _ {Cl^{ab, cd}}
    where ab and cd are either TT, Ty, yT, or yy) as 4 x4 matrix and find its inverse
    '''
    Covabcd = [[CovClTTClTT, CovClTTClTy, CovClTTClyT, CovClTTClyy],
    [CovClTyClTT, CovClTyClTy, CovClTyClyT, CovClTyClyy],
    [CovClyTClTT, CovClyTClTy, CovClyTClyT, CovClyTClyy],
    [CovClyyClTT, CovClyyClTy, CovClyyClyT, CovClyyClyy]]
    Covabcdinv = np.linalg.inv(Covabcd)

    '''
    Calculate derivatives of ClTT, ClTy, ClyT, and Clyy with respect to Acmb and Atsz at Acmb=Atsz=1
    Note that Acmb and Atsz should be inserted next to CC and T, respectively, in ClijNoNoise but not in weights
    '''
    derivClTTAcmb = CC
    derivClTyAcmb = (CC*(g2*N1Split2+g1*N2Split2))/(CC*(g1-g2)**2+g2**2*N1Split2+g1**2*N2Split2)
    derivClyTAcmb = (CC*(g2*N1Split1+g1*N2Split1))/(CC*(g1-g2)**2+g2**2*N1Split1+g1**2*N2Split1)
    derivClyyAcmb = (CC*(g2*N1Split1+g1*N2Split1)*(g2*N1Split2+g1*N2Split2))/((CC*(g1-g2)**2+g2**2*N1Split1+g1**2*N2Split1)*(CC*(g1-g2)**2+g2**2*N1Split2+g1**2*N2Split2))
    derivClTTAtsz = ((g2*N1Split1+g1*N2Split1)*(g2*N1Split2+g1*N2Split2)*T)/((N1Split1+N2Split1+(g1-g2)**2*T)*(N1Split2+N2Split2+(g1-g2)**2*T))
    derivClTyAtsz = ((g2*N1Split1+g1*N2Split1)*T)/(N1Split1+N2Split1+(g1-g2)**2*T)
    derivClyTAtsz = ((g2*N1Split2+g1*N2Split2)*T)/(N1Split2+N2Split2+(g1-g2)**2*T)
    derivClyyAtsz = T
    derivAcmb = [derivClTTAcmb, derivClTyAcmb, derivClyTAcmb, derivClyyAcmb]
    derivAtsz = [derivClTTAtsz, derivClTyAtsz, derivClyTAtsz, derivClyyAtsz]

    '''
    Calculate Fisher matrix from ILC approach
    '''
    FisherAcmb = np.einsum('i,j,ij->',derivAcmb,derivAcmb,Covabcdinv)
    FisherAtsz = np.einsum('i,j,ij->',derivAtsz,derivAtsz,Covabcdinv)
    FisherAcmbAtsz = 1/2*(np.einsum('i,j,ij->',derivAcmb,derivAtsz,Covabcdinv) + np.einsum('i,j,ij->',derivAtsz,derivAcmb,Covabcdinv))
    Fisher = [[FisherAcmb, FisherAcmbAtsz], [FisherAcmbAtsz, FisherAtsz]]

    '''
    Calculate final covariance matrix from ILC approach
    '''
    FinalCovILC = np.linalg.inv(Fisher)

    '''
    Begin multifrequency template fitting approach. Find inverse of power spectrum covariance matrix Clij^-1.
    '''
    PScovInv = np.linalg.inv(PScovFull)

    '''
    Find K_ {alpha,beta} by summing over each term of PScovInv once.
    '''
    KCMBCMB = CC**2*(PScovInv[0, 0] + PScovInv[0, 1] + PScovInv[0, 2] + PScovInv[0, 3] +
      PScovInv[1, 0] + PScovInv[1, 1] + PScovInv[1, 2] + PScovInv[1, 3] +
      PScovInv[2, 0] + PScovInv[2, 1] + PScovInv[2, 2] + PScovInv[2, 3] +
      PScovInv[3, 0] + PScovInv[3, 1] + PScovInv[3, 2] + PScovInv[3, 3])
    KCMBtSZ = CC*T*(PScovInv[0, 0]*g1**2 + PScovInv[0, 1]*g1*g2 +
        PScovInv[0, 2]*g2*g1 + PScovInv[0, 3]*g2**2 +
        PScovInv[1, 0]*g1**2 + PScovInv[1, 1]*g1*g2 +
        PScovInv[1, 2]*g2*g1 + PScovInv[1, 3]*g2**2 +
        PScovInv[2, 0]*g1**2 + PScovInv[2, 1]*g1*g2 +
        PScovInv[2, 2]*g2*g1 + PScovInv[2, 3]*g2**2 +
        PScovInv[3, 0]*g1**2 + PScovInv[3, 1]*g1*g2 +
        PScovInv[3, 2]*g2*g1 + PScovInv[3, 3]*g2**2)
    KtSZtSZ = T**2*(g1**2*PScovInv[0, 0]*g1**2 + g1**2*PScovInv[0, 1]*g1*g2 +
        g1**2*PScovInv[0, 2]*g2*g1 + g1**2*PScovInv[0, 3]*g2**2 +
        g1*g2*PScovInv[1, 0]*g1**2 + g1*g2*PScovInv[1, 1]*g1*g2 +
        g1*g2*PScovInv[1, 2]*g2*g1 + g1*g2*PScovInv[1, 3]*g2**2 +
        g2*g1*PScovInv[2, 0]*g1**2 + g2*g1*PScovInv[2, 1]*g1*g2 +
        g2*g1*PScovInv[2, 2]*g2*g1 + g2*g1*PScovInv[2, 3]*g2**2 +
        g2**2*PScovInv[3, 0]*g1**2 + g2**2*PScovInv[3, 1]*g1*g2 +
        g2**2*PScovInv[3, 2]*g2*g1 + g2**2*PScovInv[3, 3]*g2**2)
    K = [[KCMBCMB, KCMBtSZ], [KCMBtSZ, KtSZtSZ]]

    '''
    Find final covariance matrix from template fitting approach using eq.67
    '''
    FinalCovTemplateFitting = np.linalg.inv(K)

    '''
    Compare results from two approaches and fill in arrays
    '''
    if ell%20==0:
          print(f'ell={ell}')
    CMB_var_ILC.append(FinalCovILC[0,0])
    CMB_tSZ_covar_ILC.append(FinalCovILC[0,1])
    tSZ_var_ILC.append(FinalCovILC[1,1])
    CMB_var_template.append(FinalCovTemplateFitting[0,0])
    CMB_tSZ_covar_template.append(FinalCovTemplateFitting[0,1])
    tSZ_var_template.append(FinalCovTemplateFitting[1,1])
  
  FinalCovILC = np.array([[CMB_var_ILC, CMB_tSZ_covar_ILC], [CMB_tSZ_covar_ILC, tSZ_var_ILC]])
  FinalCovTemplateFitting = np.array([[CMB_var_template, CMB_tSZ_covar_template], [CMB_tSZ_covar_template, tSZ_var_template]])

  #save files if requested
  if inp.save_files:
    pickle.dump(FinalCovILC, open(f'{inp.output_dir}/data_splits_hilc_cov.p', 'wb'))
    pickle.dump(FinalCovTemplateFitting, open(f'{inp.output_dir}/data_splits_template_fitting_cov.p', 'wb'))
  
  return FinalCovILC, FinalCovTemplateFitting



def main():
    '''
    RETURNS
    -------
    FinalCovILC: (2, 2, ellmax+1) ndarray containing parameter covariance matrix from harmonic ILC
    FinalCovTemplateFitting: (2, 2, ellmax+1) ndarray containing parameter covariance matrix from template-fitting
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from harmonic ILC vs. template-fitting approach for data-split cross-spectra.")
    parser.add_argument("--config", default="../template_fitting_pipeline/stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    start_time = time.time()

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    # compute variances
    FinalCovILC, FinalCovTemplateFitting = get_cov(inp)
    
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return FinalCovILC, FinalCovTemplateFitting

if __name__ == '__main__':
    main()

