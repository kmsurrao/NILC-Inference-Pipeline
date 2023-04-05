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
CMB_var_array = []
CMB_tSZ_covar_array = []
tSZ_var_array = []

for ell in ells:
    CC = clTT[ell]
    T = tsz[ell]
    N1 = W**2*np.exp(ell*(ell+1)*sigma**2) #noise at 90 GHz
    N2 = N1 #noise at 150 GHz
    Nmodes = (2*ell+1)*f


    #use analytic expressions derived from Mathematica notebook
    CMB_var = (2* (CC* (g1 - g2)**2 + g2**2*N1 + g1**2*N2)**2 *(g1**2*N2*T +
       CC* (N1 + N2 + (g1 - g2)**2*T) + N1* (N2 + g2**2*T)))/(CC**2 *(g1 -
       g2)**2 *Nmodes *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
       g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
       CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2 *T)))
    CMB_tSZ_covar = -((2 *(g2*N1 + g1*N2)**2 *(g1**2*N2*T + CC
       *(N1 + N2 + (g1 - g2)**2*T) + N1 *(N2 + g2**2*T)))/
       (CC *(g1 - g2)**2*Nmodes*T *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
       g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
       CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2*T))))
    tSZ_var = (2 *(N1 + N2 + (g1 - g2)**2*T)**2 *(g1**2*N2*T +
      CC *(N1 + N2 + (g1 - g2)**2*T) + N1 *(N2 + g2**2*T)))/((g1 -
      g2)**2 *Nmodes *T**2 *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
      g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
      CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2 *T)))

    CMB_var_array.append(CMB_var)
    CMB_tSZ_covar_array.append(CMB_tSZ_covar)
    tSZ_var_array.append(tSZ_var)

#make plots
plt.clf()
plt.plot(ells, CMB_var_array)
plt.xlabel(r'$\ell$')
plt.ylabel('CMB variance')
plt.savefig('variance_plots/no_splits/CMB_variance')

plt.clf()
plt.plot(ells, CMB_tSZ_covar_array)
plt.xlabel(r'$\ell$')
plt.ylabel('CMB tSZ Covariance')
plt.savefig('variance_plots/no_splits/CMB_tSZ_covariance')

plt.clf()
plt.plot(ells, tSZ_var_array)
plt.xlabel(r'$\ell$')
plt.ylabel('tSZ variance')
plt.savefig('variance_plots/no_splits/tSZ_variance')
