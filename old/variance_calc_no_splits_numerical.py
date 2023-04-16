import numpy as np
from classy import Class
from fgspectra import cross as fgc
from fgspectra import power as fgp
from fgspectra import frequency as fgf
import matplotlib.pyplot as plt


#get tSZ power spectrum from fgpsectra
T_cmb = 2.726
h = 6.62607004*10**(-34)
kb = 1.38064852*10**(-23)
ell_max = 6000
def tsz_spectral_response(freq): #input frequency in GHz
    x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
    return T_cmb*10**6*(x*1/np.tanh(x/2)-4) #was factor of tcmb microkelvin before
tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
freqs = np.array([150.]) #input in GHz
ells = np.arange(ell_max)
a_tSZ = 4.66
tsz = a_tSZ * tsz(
        {'nu':freqs, 'nu_0':150.0,},
        {'ell':ells, 'ell_0':3000})
tsz = tsz[0][0]/(tsz_spectral_response(150.))**2 #given in microKelvin**2*l(l+1)/(2pi)?
for l in range(1,len(tsz)):
    tsz[l] = 2*np.pi/(l*(l+1))*tsz[l] #in clyy units



#get CMB power spectrum from Class
LambdaCDM = Class()
LambdaCDM.set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842, 'l_max_scalars':10000})
LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
LambdaCDM.compute()
cls = LambdaCDM.lensed_cl(ell_max)
clTT = cls['tt']*(T_cmb*10**6)**2 #Class gives clTT in K, represents temp as DeltaT/T



#define parameters
f = 1.0 #f_sky
g1 = tsz_spectral_response(90.) #frequency dependence at 90 GHz
g2 = tsz_spectral_response(150.) #frequency dependence at 150 GHz
theta_fwhm = (1.4/60)*(np.pi/180)
sigma = theta_fwhm/np.sqrt(8*np.log(2))
W = (1/60)*(np.pi/180)
ells = [ell for ell in range(100,3000)]

#set up arrays
CMB_var_ILC = []
CMB_tSZ_covar_ILC = []
tSZ_var_ILC = []
CMB_var_template = []
CMB_tSZ_covar_template = []
tSZ_var_template = []

for ell in ells:
    CC = clTT[ell]
    T = tsz[ell]
    N1 = W**2*np.exp(ell*(ell+1)*sigma**2) #noise at 90 GHz
    N2 = N1 #noise at 150 GHz


    '''
    Define matrix of auto- and cross-frequency power spectra at some multipole ell.
    Here, CC = C_ell**CMB , N1 = N_ell**{freq1} , N2 = N_ell**{freq2} ,
    g1 = tSZ spectral function at freq1, g2 = tSZ spectral function at freq2,
    T = C_ell**yy (tSZ power spectrum in frequency-independent Compton-y units).
    '''
    Clij = np.array([[CC + N1 + g1**2*T, CC + g1*g2*T], [CC + g1*g2*T, CC + N2 + g2**2*T]])
    ClijInv = np.linalg.inv(Clij)


    '''
    Define spectra response vectors for T and CC
    '''
    gvec = [g1,g2]
    hvec = [1,1]

    '''
    Get the ILC weights for the y -maps and the CMB maps explicitly
    '''
    weightsILCy = (gvec @ ClijInv)/(gvec @ ClijInv @ gvec)
    weightsILCCMB = (hvec @ ClijInv)/(hvec @ ClijInv @ hvec)

    '''
    Find the total power at this ell in the harmonic ILC map.
    '''
    Clyy = np.einsum('i,j,ij->', weightsILCy, weightsILCy, Clij)
    ClTT = np.einsum('i,j,ij->', weightsILCCMB, weightsILCCMB, Clij)
    ClTy = np.einsum('i,j,ij->', weightsILCCMB, weightsILCy, Clij)


    '''
    First, construct the covariance matrix of the auto- and cross-frequency power spectra.
    Explicitly include 150x90 terms in power spectrum covariance matrix, i.e., it is a 4x4 matrix.
    '''
    Nmodes = f*(2*ell+1)
    PScovFull = (1/Nmodes)*np.array([[2*Clij[0, 0]**2,
       2*(CC + g1**2*T)*Clij[0, 1] + 2*N1*Clij[0, 1],
       2*(CC + g1**2*T)*Clij[0, 1] + 2*N1*Clij[0, 1],
       2*Clij[0, 1]**2], [2*(CC + g1**2*T)*Clij[0, 1] +
        2*N1*Clij[0, 1], Clij[0, 0]*Clij[1, 1] + Clij[0, 1]**2,
       Clij[0, 0]*Clij[1, 1] + Clij[0, 1]**2,
       2*(CC + g2**2*T)*Clij[0, 1] +
        2*N2*Clij[0, 1]], [2*(CC + g1**2*T)*Clij[0, 1] +
        2*N1*Clij[0, 1], Clij[0, 0]*Clij[1, 1] + Clij[0, 1]**2,
       Clij[0, 0]*Clij[1, 1] + Clij[0, 1]**2,
       2*(CC + g2**2*T)*Clij[0, 1] + 2*N2*Clij[0, 1]], [2*
        Clij[0, 1]**2, 2*(CC + g2**2*T)*Clij[0, 1] + 2*N2*Clij[0, 1],
       2*(CC + g2**2*T)*Clij[0, 1] + 2*N2*Clij[0, 1],
       2*Clij[1, 1]**2]])

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
    Compute the sums in equation 87
    '''
    CovClTTClTT = np.einsum('i,j,k,l,ijkl->', weightsILCCMB, weightsILCCMB, weightsILCCMB, weightsILCCMB, PScovFullTensor)
    CovClTTClTy = np.einsum('i,j,k,l,ijkl->', weightsILCCMB, weightsILCCMB, weightsILCCMB, weightsILCy, PScovFullTensor)
    CovClTTClyy = np.einsum('i,j,k,l,ijkl->', weightsILCCMB, weightsILCCMB, weightsILCy, weightsILCy, PScovFullTensor)
    CovClTyClTT = CovClTTClTy
    CovClTyClTy = np.einsum('i,j,k,l,ijkl->', weightsILCCMB, weightsILCy, weightsILCCMB, weightsILCy, PScovFullTensor)
    CovClTyClyy = np.einsum('i,j,k,l,ijkl->', weightsILCCMB, weightsILCy, weightsILCy, weightsILCy, PScovFullTensor)
    CovClyyClTT = CovClTTClyy
    CovClyyClTy = CovClTyClyy
    CovClyyClyy = np.einsum('i,j,k,l,ijkl->', weightsILCy, weightsILCy, weightsILCy, weightsILCy, PScovFullTensor)


    '''
    Construct Cov_ {ab,cd} (matrix where Cov_ {ab, cd} = sigma^2 _ {Cl^{ab, cd}}
    where ab and cd are either TT, Ty, or yy) as 3x3 matrix and find its inverse
    '''
    Covabcd = [[CovClTTClTT, CovClTTClTy, CovClTTClyy],
    [CovClTyClTT, CovClTyClTy, CovClTyClyy],
    [CovClyyClTT, CovClyyClTy, CovClyyClyy]]
    Covabcdinv = np.linalg.inv(Covabcd)

    '''
    Calculate derivatives of ClTT, ClTy, ClyT, and Clyy with respect to Acmb and Atsz at Acmb=Atsz=1
    Note that Acmb and Atsz should be inserted next to CC and T, respectively, in ClijNoNoise but not in weights
    '''
    derivClTTAcmb = CC
    derivClTyAcmb = (CC *(g2*N1 + g1*N2))/(CC *(g1 - g2)**2 + g2**2*N1 + g1**2*N2)
    derivClyyAcmb = (CC *(g2*N1 + g1*N2)**2)/(CC *(g1 - g2)**2 + g2**2*N1 + g1**2*N2)**2
    derivClTTAtsz = ((g2*N1 + g1*N2)**2 *T)/(N1 + N2 + (g1 - g2)**2 *T)**2
    derivClTyAtsz = ((g2*N1 + g1*N2) *T)/(N1 + N2 + (g1 - g2)**2*T)
    derivClyyAtsz = T
    derivAcmb = [derivClTTAcmb, derivClTyAcmb, derivClyyAcmb]
    derivAtsz = [derivClTTAtsz, derivClTyAtsz, derivClyyAtsz]

    '''
    Calculate Fisher matrix from ILC approach as in eq .99
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
    PScov = [[PScovFull[0, 0], PScovFull[0, 1], PScovFull[0, 3]],
    [PScovFull[1, 0], PScovFull[1, 1], PScovFull[1, 3]],
    [PScovFull[3, 0], PScovFull[3, 1], PScovFull[3, 3]]]
    PScovInv = np.linalg.inv(PScov)

    '''
    Find K_ {alpha,beta} from eq.65 by summing over each term of PScovInv once.
    '''
    KCMBCMB = CC**2*(PScovInv[0, 0] + PScovInv[0, 1] + PScovInv[0, 2] +
      PScovInv[1, 0] + PScovInv[1, 1] + PScovInv[1, 2] +
      PScovInv[2, 0] + PScovInv[2, 1] + PScovInv[2, 2])
    KCMBtSZ = CC*T*(PScovInv[0, 0]*g1**2 + PScovInv[0, 1]*g1*g2 +
         PScovInv[0, 2]*g2**2 + PScovInv[1, 0]*g1**2 + PScovInv[1, 1]*g1*g2 +
         PScovInv[1, 2]*g2**2 + PScovInv[2, 0]*g1**2 + PScovInv[2, 1]*g1*g2 +
         PScovInv[2, 2]*g2**2)
    KtSZtSZ = T**2*(g1**2*PScovInv[0, 0]*g1**2 + g1**2*PScovInv[0, 1]*g1*g2 +
         g1**2*PScovInv[0, 2]*g2**2 + g1*g2*PScovInv[1, 0]*g1**2 +
         g1*g2*PScovInv[1, 1]*g1*g2 + g1*g2*PScovInv[1, 2]*g2**2 +
         g2**2*PScovInv[2, 0]*g1**2 + g2**2*PScovInv[2, 1]*g1*g2 +
         g2**2*PScovInv[2, 2]*g2**2)
    K = [[KCMBCMB, KCMBtSZ], [KCMBtSZ, KtSZtSZ]]

    '''
    Find final covariance matrix from template fitting approach using eq.69
    '''
    FinalCovTemplateFitting = np.linalg.inv(K)

    '''
    Compare results from two approaches and fill in arrays
    '''
    if ell%200==0:
          print(f'ell={ell}')
          print(FinalCovILC)
          print(FinalCovTemplateFitting)
          print()
    CMB_var_ILC.append(FinalCovILC[0,0])
    CMB_tSZ_covar_ILC.append(FinalCovILC[0,1])
    tSZ_var_ILC.append(FinalCovILC[1,1])
    CMB_var_template.append(FinalCovTemplateFitting[0,0])
    CMB_tSZ_covar_template.append(FinalCovTemplateFitting[0,1])
    tSZ_var_template.append(FinalCovTemplateFitting[1,1])

#make plots
plt.clf()
plt.plot(ells, CMB_var_ILC, label='Harmonic ILC')
plt.plot(ells, CMB_var_template, label='Template Fitting')
plt.xlabel(r'$\ell$')
plt.ylabel('CMB variance')
plt.legend()
plt.savefig('variance_plots/no_splits/CMB_variance')

plt.clf()
plt.plot(ells, CMB_tSZ_covar_ILC, label='Harmonic ILC')
plt.plot(ells, CMB_tSZ_covar_template, label='Template Fitting')
plt.xlabel(r'$\ell$')
plt.ylabel('CMB tSZ Covariance')
plt.legend()
plt.savefig('variance_plots/no_splits/CMB_tSZ_covariance')

plt.clf()
plt.plot(ells, tSZ_var_ILC, label='Harmonic ILC')
plt.plot(ells, tSZ_var_template, label='Template Fitting')
plt.xlabel(r'$\ell$')
plt.ylabel('tSZ variance')
plt.legend()
plt.savefig('variance_plots/no_splits/tSZ_variance')