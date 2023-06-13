# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import kv
import PRyM.PRyM_init as PRyMini
if(PRyMini.numba_flag):
    from numba import njit

my_dir = PRyMini.working_dir
if(PRyMini.verbose_flag):
    print("PRyM_thermo.py: Loading SM rates for thermal bath")
    print("Natural units adopted here. Temperatures in MeV.")
 
###########################################################
# Standard Model matrix elements & plasma QED corrections #
###########################################################
# Credit for dataset to NUDEC_BSM:
# ArXiv:1812.05605 [JCAP 1902 (2019) 007] and ArXiv:2001.04466 [JCAP 05 (2020) 048])
# Effect of finite electron mass in scattering matrix elements (standard value for me assumed)
fnu_e_scat_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"nue_scatt.txt")
fnu_e_scat = interp1d(fnu_e_scat_tab[:,0],fnu_e_scat_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
fnu_mu_scat_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"numu_scatt.txt")
fnu_mu_scat = interp1d(fnu_mu_scat_tab[:,0],fnu_mu_scat_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
# Effect of finite electron mass in annihilation matrix elements (standard value for me assumed)
fnu_e_ann_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"nue_ann.txt")
fnu_e_ann = interp1d(fnu_e_ann_tab[:,0],fnu_e_ann_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
fnu_mu_ann_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"numu_ann.txt")
fnu_mu_ann = interp1d(fnu_mu_ann_tab[:,0],fnu_mu_ann_tab[:,1], bounds_error=False, fill_value="extrapolate", kind='linear')
# QED plasma corrections (standard value for alphaem and me assumed)
P_QED_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_P_int.txt")
PofT = interp1d(P_QED_tab[:,0],P_QED_tab[:,1]+P_QED_tab[:,2], bounds_error=False, fill_value="extrapolate", kind='linear')
dPdT_QED_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_dP_intdT.txt")
dPdT = interp1d(dPdT_QED_tab[:,0],dPdT_QED_tab[:,1]+dPdT_QED_tab[:,2], bounds_error=False, fill_value="extrapolate", kind='linear')
d2PdT2_QED_tab = np.loadtxt(my_dir+"/PRyMrates/thermo/"+"QED_d2P_intdT2.txt")
d2PdT2 = interp1d(d2PdT2_QED_tab[:,0],d2PdT2_QED_tab[:,1]+d2PdT2_QED_tab[:,2], bounds_error=False, fill_value="extrapolate", kind='linear')

##################
# Photon species #
##################
# Photon energy density
def rho_g(Tg):
    return 2.*(np.pi**2/30.)*Tg**4
# drho_g/dT
def drho_g_dT(Tg):
    return 4.*rho_g(Tg)/Tg
    
###############
# e+- species #
###############
# e+- energy density
if(PRyMini.numba_flag):
    @njit
    def rho_e_int(E,Tg):
        return E**2*(E**2-(PRyMini.me/Tg)**2)**0.5/(np.exp(E)+1.)
else:
    def rho_e_int(E,Tg):
        return E**2*(E**2-(PRyMini.me/Tg)**2)**0.5/(np.exp(E)+1.)
def rho_e(Tg):
    if Tg < PRyMini.me/30.:
        return 0.0
    else:
        res_int = quad(rho_e_int,PRyMini.me/Tg,100.,args=(Tg),epsabs=1e-12,epsrel=1e-12)[0]
        return 4./(2*np.pi**2)*Tg**4*res_int
# drho_e/dT
if(PRyMini.numba_flag):
    @njit
    def drho_e_dT_int(E,Tg):
        return E**3*(E**2-(PRyMini.me/Tg)**2)**0.5/np.cosh(E/2.0)**2
else:
    def drho_e_dT_int(E,Tg):
        return E**3*(E**2-(PRyMini.me/Tg)**2)**0.5/np.cosh(E/2.0)**2
def drho_e_dT(Tg):
    if Tg < PRyMini.me/30.:
        return 0.0
    else:
        res_int = quad(drho_e_dT_int,PRyMini.me/Tg,100.,args=(Tg),epsabs=1e-12,epsrel = 1e-12)[0]
        return 1./(2*np.pi**2)*Tg**3*res_int
# e+- pressure density
if(PRyMini.numba_flag):
    @njit
    def p_e_int(E,Tg):
        return (E**2-(PRyMini.me/Tg)**2)**1.5/(np.exp(E)+1.)
else:
    def p_e_int(E,Tg):
        return (E**2-(PRyMini.me/Tg)**2)**1.5/(np.exp(E)+1.)
def p_e(Tg):
    if Tg < PRyMini.me/30.:
        return 0.0
    else:
        res_int = quad(p_e_int,PRyMini.me/Tg,100.,args=(Tg),epsabs=1e-12,epsrel=1e-12)[0]
        return 4./(6*np.pi**2)*Tg**4*res_int

####################
# Neutrino species #
####################
# Neutrino energy density
def rho_nu(Tnu):
    return 2.*(7./8.)*(np.pi**2)/30.*Tnu**4
# drho_nu/dT
def drho_nu_dT(Tnu):
    return 4.*rho_nu(Tnu)/Tnu
 
##########################
# e+- nu matrix elements #
##########################
# Pauli blocking for relativistic fermions as in [JCAP 05 (2020) 048]
fannFD, fscatFD = 0.884, 0.829
def f_nu_e(T1,T2):
    res = 32.*fannFD*(T1**9-T2**9)*fnu_e_ann(T1)+56.*fscatFD*fnu_e_scat(T1)*T1**4*T2**4*(T1-T2)
    return res
def f_nu_mu(T1,T2):
    res = 32.*fannFD*(T1**9-T2**9)*fnu_mu_ann(T1)+56.*fscatFD*fnu_mu_scat(T1)*T1**4*T2**4*(T1-T2)
    return res
def f_g(T1,T2):
    res = 32.*fannFD*(T1**9-T2**9)+56.*fscatFD*T1**4*T2**4*(T1-T2)
    return res
# Collision terms in Boltzmann equation for energy densities
def delta_rho_nue(Tg,Tnue,Tnumu):
    return PRyMini.MeV_to_secm1*PRyMini.GF**2/np.pi**5*(4.*(PRyMini.geL**2+PRyMini.geR**2)*f_nu_e(Tg,Tnue)+2.*f_g(Tnumu,Tnue))
def delta_rho_numu(Tg,Tnue,Tnumu):
    return PRyMini.MeV_to_secm1*PRyMini.GF**2/np.pi**5*(4.*(PRyMini.gmuL**2+PRyMini.gmuR**2)*f_nu_mu(Tg,Tnue)-f_g(Tnumu,Tnue))

#######################
# Standard Model (SM) #
#######################
# Total SM energy density
def rho_SM(Tg,Tnue,Tnumu):
    rho_3nu = rho_nu(Tnue)+2.*rho_nu(Tnumu)
    rho_plasma = rho_g(Tg)+rho_e(Tg)
    delta_rho_QED = Tg*dPdT(Tg)-PofT(Tg)
    return rho_plasma+rho_3nu+delta_rho_QED
# Total SM pressure density
def p_SM(Tg,Tnue,Tnumu):
    p_3nu = (rho_nu(Tnue)+2.*rho_nu(Tnumu))/3.
    p_plasma = rho_g(Tg)/3.+p_e(Tg)
    delta_p_QED = PofT(Tg)
    return p_plasma+p_3nu+delta_p_QED

############################
# New Physics (NP) species #
############################
# NP energy density
def rho_NP(T_NP):
    return 0.
# NP pressure density
def p_NP(T_NP):
    return 0.
# drho_NP/dT
def drho_NP_dT(T_NP):
    return 0.
# Collision terms in Boltzmann equation for rho_NP
def delta_rho_NP(Tg,Tnue,Tnumu,T_NP):
    return 0.

##########################
# Plasma entropy density #
##########################
def spl(Tg):
    rho_pl = rho_g(Tg)+rho_e(Tg)
    p_pl = rho_g(Tg)/3.+p_e(Tg)
    delta_rho_QED = Tg*dPdT(Tg)-PofT(Tg)
    delta_p_QED = PofT(Tg)
    spl_T = (rho_pl+p_pl+(delta_rho_QED+delta_p_QED))/Tg
    # NP species in equilibrium with e+-, gamma (i.e. SM plasma)
    if(PRyMini.NP_e_flag):
        spl_T += (rho_NP(Tg)+p_NP(Tg))/Tg
    return spl_T
