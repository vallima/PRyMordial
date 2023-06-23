# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini

if(PRyMini.verbose_flag):
    print("PRyM_nuclear_rates.py: Loading and interpolating nuclear rates")
    print(" ")

class UpdateNuclearRates(object):
    def __init__(self,p_npdg,p_dpHe3g,p_ddHe3n,p_ddtp,p_tpag,p_tdan,p_taLi7g, p_He3ntp,p_He3dap,p_He3aBe7g,p_Be7nLi7p,p_Li7paa):
        npdg_mu = PRyMini.npdg_median*np.exp(p_npdg*np.log(PRyMini.npdg_expsigma))
        dpHe3g_mu = PRyMini.dpHe3g_median*np.exp(p_dpHe3g*np.log(PRyMini.dpHe3g_expsigma))
        ddHe3n_mu = PRyMini.ddHe3n_median*np.exp(p_ddHe3n*np.log(PRyMini.ddHe3n_expsigma))
        ddtp_mu = PRyMini.ddtp_median*np.exp(p_ddtp*np.log(PRyMini.ddtp_expsigma))
        tpag_mu = PRyMini.tpag_median*np.exp(p_tpag*np.log(PRyMini.tpag_expsigma))
        tdan_mu = PRyMini.tdan_median*np.exp(p_tdan*np.log(PRyMini.tdan_expsigma))
        taLi7g_mu = PRyMini.taLi7g_median*np.exp(p_taLi7g*np.log(PRyMini.taLi7g_expsigma))
        He3ntp_mu = PRyMini.He3ntp_median*np.exp(p_He3ntp*np.log(PRyMini.He3ntp_expsigma))
        He3dap_mu = PRyMini.He3dap_median*np.exp(p_He3dap*np.log(PRyMini.He3dap_expsigma))
        He3aBe7g_mu = PRyMini.He3aBe7g_median*np.exp(p_He3aBe7g*np.log(PRyMini.He3aBe7g_expsigma))
        Be7nLi7p_mu = PRyMini.Be7nLi7p_median*np.exp(p_Be7nLi7p*np.log(PRyMini.Be7nLi7p_expsigma))
        Li7paa_mu = PRyMini.Li7paa_median*np.exp(p_Li7paa*np.log(PRyMini.Li7paa_expsigma))
        if(PRyMini.NP_nuclear_flag):
            npdg_mu += PRyMini.NP_delta_npdg*PRyMini.npdg_median
            dpHe3g_mu += PRyMini.NP_delta_dpHe3g*PRyMini.dpHe3g_median
            ddHe3n_mu += PRyMini.NP_delta_ddHe3n*PRyMini.ddHe3n_median
            ddtp_mu += PRyMini.NP_delta_ddtp*PRyMini.ddtp_median
            tpag_mu += PRyMini.NP_delta_tpag*PRyMini.tpag_median
            tdan_mu += PRyMini.NP_delta_tdan*PRyMini.tdan_median
            taLi7g_mu += PRyMini.NP_delta_taLi7g*PRyMini.taLi7g_median
            He3ntp_mu += PRyMini.NP_delta_He3ntp*PRyMini.He3ntp_median
            He3dap_mu += PRyMini.NP_delta_He3dap*PRyMini.He3dap_median
            He3aBe7g_mu += PRyMini.NP_delta_He3aBe7g*PRyMini.He3aBe7g_median
            Be7nLi7p_mu += PRyMini.NP_delta_Be7nLi7p*PRyMini.Be7nLi7p_median
            Li7paa_mu += PRyMini.NP_delta_Li7paa*PRyMini.Li7paa_median
        self.npdg_spline = interp1d(PRyMini.npdg_T9,npdg_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.dpHe3g_spline = interp1d(PRyMini.dpHe3g_T9,dpHe3g_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.ddHe3n_spline = interp1d(PRyMini.ddHe3n_T9,ddHe3n_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.ddtp_spline = interp1d(PRyMini.ddtp_T9,ddtp_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.tpag_spline = interp1d(PRyMini.tpag_T9,tpag_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.tdan_spline = interp1d(PRyMini.tdan_T9,tdan_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.taLi7g_spline = interp1d(PRyMini.taLi7g_T9,taLi7g_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.He3ntp_spline = interp1d(PRyMini.He3ntp_T9,He3ntp_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.He3dap_spline = interp1d(PRyMini.He3dap_T9,He3dap_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.He3aBe7g_spline = interp1d(PRyMini.He3aBe7g_T9,He3aBe7g_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7nLi7p_spline = interp1d(PRyMini.Be7nLi7p_T9,Be7nLi7p_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li7paa_spline = interp1d(PRyMini.Li7paa_T9,Li7paa_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
    
    # n p --> d g
    def npdg_frwrd(self,T):
        T9 = T*1.e-9
        return self.npdg_spline(T9)
    def npdg_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_npdg
        beta = PRyMini.beta_npdg
        gamma = PRyMini.gamma_npdg
        return alpha*T9**beta*np.exp(gamma/T9)*self.npdg_spline(T9)

    # d p --> He3 g
    def dpHe3g_frwrd(self,T):
        T9 = T*1.e-9
        return self.dpHe3g_spline(T9)
    def dpHe3g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_dpHe3g
        beta = PRyMini.beta_dpHe3g
        gamma = PRyMini.gamma_dpHe3g
        return alpha*T9**beta*np.exp(gamma/T9)*self.dpHe3g_spline(T9)

    # d d --> He3 n
    def ddHe3n_frwrd(self,T):
        T9 = T*1.e-9
        return self.ddHe3n_spline(T9)
    def ddHe3n_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_ddHe3n
        beta = PRyMini.beta_ddHe3n
        gamma = PRyMini.gamma_ddHe3n
        return alpha*T9**beta*np.exp(gamma/T9)*self.ddHe3n_spline(T9)
    
    # d d --> t p
    def ddtp_frwrd(self,T):
        T9 = T*1.e-9
        return self.ddtp_spline(T9)
    def ddtp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_ddtp
        beta = PRyMini.beta_ddtp
        gamma = PRyMini.gamma_ddtp
        return alpha*T9**beta*np.exp(gamma/T9)*self.ddtp_spline(T9)
    
    # t p --> a g
    def tpag_frwrd(self,T):
        T9 = T*1.e-9
        return self.tpag_spline(T9)
    def tpag_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_tpag
        beta = PRyMini.beta_tpag
        gamma = PRyMini.gamma_tpag
        return alpha*T9**beta*np.exp(gamma/T9)*self.tpag_spline(T9)
    
    # t d --> a n
    def tdan_frwrd(self,T):
        T9 = T*1.e-9
        return self.tdan_spline(T9)
    def tdan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_tdan
        beta = PRyMini.beta_tdan
        gamma = PRyMini.gamma_tdan
        return alpha*T9**beta*np.exp(gamma/T9)*self.tdan_spline(T9)
    
    # t a --> Li7 g
    def taLi7g_frwrd(self,T):
        T9 = T*1.e-9
        return self.taLi7g_spline(T9)
    def taLi7g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_taLi7g
        beta = PRyMini.beta_taLi7g
        gamma = PRyMini.gamma_taLi7g
        return alpha*T9**beta*np.exp(gamma/T9)*self.taLi7g_spline(T9)
    
    # He3 n --> t p
    def He3ntp_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3ntp_spline(T9)
    def He3ntp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3ntp
        beta = PRyMini.beta_He3ntp
        gamma = PRyMini.gamma_He3ntp
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3ntp_spline(T9)
    
    # He3 d --> a p
    def He3dap_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3dap_spline(T9)
    def He3dap_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3dap
        beta = PRyMini.beta_He3dap
        gamma = PRyMini.gamma_He3dap
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3dap_spline(T9)
    
    # He3 a --> Be7 g
    def He3aBe7g_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3aBe7g_spline(T9)
    def He3aBe7g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3aBe7g
        beta = PRyMini.beta_He3aBe7g
        gamma = PRyMini.gamma_He3aBe7g
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3aBe7g_spline(T9)
    
    # Be7 n --> Li7 p
    def Be7nLi7p_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7nLi7p_spline(T9)
    def Be7nLi7p_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7nLi7p
        beta = PRyMini.beta_Be7nLi7p
        gamma = PRyMini.gamma_Be7nLi7p
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7nLi7p_spline(T9)
    
    # Li7 p --> a a
    def Li7paa_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7paa_spline(T9)
    def Li7paa_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7paa
        beta = PRyMini.beta_Li7paa
        gamma = PRyMini.gamma_Li7paa
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7paa_spline(T9)

    # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4}
    
    def dYndt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return -nTOp_frwrd(T_t)*Yn1p0 + nTOp_bkwrd(T_t)*Yn0p1 - rhoBBN*self.npdg_frwrd(T_t)*Yn1p0*Yn0p1 + self.npdg_bkwrd(T_t)*Yn1p1 + 0.5*rhoBBN* self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 + rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 + rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 - rhoBBN* self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 + rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4
        
    def dYpdt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return nTOp_frwrd(T_t)*Yn1p0 - nTOp_bkwrd(T_t)*Yn0p1 - rhoBBN*self.npdg_frwrd(T_t)*Yn1p0*Yn0p1 + self.npdg_bkwrd(T_t)*Yn1p1 - rhoBBN*self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 + 0.5*rhoBBN*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - rhoBBN*self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 + self.dpHe3g_bkwrd(T_t)*Yn1p2 + rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 + rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + self.tpag_bkwrd(T_t)*Yn2p2 - rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 + 0.5*rhoBBN*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 - rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 + rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4
        
    def dYddt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return rhoBBN*self.npdg_frwrd(T_t)*Yn1p0*Yn0p1 - self.npdg_bkwrd(T_t)*Yn1p1 - rhoBBN*self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 - rhoBBN*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 - rhoBBN*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 + 2.*rhoBBN*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + self.dpHe3g_bkwrd(T_t)*Yn1p2 + 2.*rhoBBN*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 + rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2
        
    def dYtdt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return 0.5*rhoBBN*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - rhoBBN*self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 + self.tpag_bkwrd(T_t)*Yn2p2 + rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 - rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 + self.taLi7g_bkwrd(T_t)*Yn4p3
        
    def dYHe3dt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return rhoBBN*self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 + 0.5*rhoBBN*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 + rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 - self.dpHe3g_bkwrd(T_t)*Yn1p2 - rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2 *Yn2p2 + self.He3aBe7g_bkwrd(T_t)*Yn3p4
        
    def dYadt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return rhoBBN*self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 + rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 - self.tpag_bkwrd(T_t)*Yn2p2 - rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 - rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 - rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2 *Yn2p2 - rhoBBN*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 + self.taLi7g_bkwrd(T_t)*Yn4p3 + 2*rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3 + self.He3aBe7g_bkwrd(T_t)*Yn3p4
        
    def dYLi7dt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 + 0.5*rhoBBN*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 - self.taLi7g_bkwrd(T_t)*Yn4p3 - rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 + rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4
        
    def dYBe7dt(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y
        return rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2 *Yn2p2 + rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - self.He3aBe7g_bkwrd(T_t)*Yn3p4 - rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4
        
    def Jacobian(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4}
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4 = Y

        # Yn
        dYn_primeOdYn = -nTOp_frwrd(T_t) + rhoBBN*(-self.npdg_frwrd(T_t)*Yn0p1 - (self.He3ntp_frwrd(T_t) + self.ddHe3n_bkwrd(T_t))*Yn1p2 - self.tdan_bkwrd(T_t)*Yn2p2 - self.Be7nLi7p_frwrd(T_t)*Yn3p4)
        dYn_primeOdYp = nTOp_bkwrd(T_t) + rhoBBN*(self.He3ntp_bkwrd(T_t)*Yn2p1-self.npdg_frwrd(T_t)*Yn1p0 + self.Be7nLi7p_bkwrd(T_t)*Yn4p3)
        dYn_primeOdYd = rhoBBN*(self.ddHe3n_frwrd(T_t)*Yn1p1 + self.tdan_frwrd(T_t)*Yn2p1) + self.npdg_bkwrd(T_t)
        dYn_primeOdYt = rhoBBN*(self.He3ntp_bkwrd(T_t)*Yn0p1 + self.tdan_frwrd(T_t)*Yn1p1)
        dYn_primeOdYHe3 = -rhoBBN*(self.He3ntp_frwrd(T_t) + self.ddHe3n_bkwrd(T_t))*Yn1p0
        dYn_primeOdYa = -rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0
        dYn_primeOdYLi7 = rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1
        dYn_primeOdYBe7 = -rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0
        dYn_row = [dYn_primeOdYn,dYn_primeOdYp,dYn_primeOdYd,dYn_primeOdYt,dYn_primeOdYHe3,dYn_primeOdYa,dYn_primeOdYLi7,dYn_primeOdYBe7]

        # Yp
        dYp_primeOdYn = nTOp_frwrd(T_t) + rhoBBN*(- self.npdg_frwrd(T_t)*Yn0p1 + self.He3ntp_frwrd(T_t)*Yn1p2 + self.Be7nLi7p_frwrd(T_t)*Yn3p4)
        dYp_primeOdYp = - nTOp_bkwrd(T_t) + rhoBBN*(- self.npdg_frwrd(T_t)*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn1p1 - (self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn2p1 - self.He3dap_bkwrd(T_t)*Yn2p2 - (self.Li7paa_frwrd(T_t) + self.Be7nLi7p_bkwrd(T_t))*Yn4p3)
        dYp_primeOdYd = rhoBBN*(self.ddtp_frwrd(T_t)*Yn1p1 - self.dpHe3g_frwrd(T_t)*Yn0p1 + self.He3dap_frwrd(T_t)*Yn1p2) + self.npdg_bkwrd(T_t)
        dYp_primeOdYt = -rhoBBN*(self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn0p1
        dYp_primeOdYHe3 = rhoBBN*(self.He3ntp_frwrd(T_t)*Yn1p0 + self.He3dap_frwrd(T_t)*Yn1p1) + self.dpHe3g_bkwrd(T_t)
        dYp_primeOdYa = rhoBBN*(-self.He3dap_bkwrd(T_t)*Yn0p1 + self.Li7paa_bkwrd(T_t)*Yn2p2) + self.tpag_bkwrd(T_t)
        dYp_primeOdYLi7 = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0
        dYp_primeOdYBe7 = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0
        dYp_row = [dYp_primeOdYn,dYp_primeOdYp,dYp_primeOdYd,dYp_primeOdYt,dYp_primeOdYHe3,dYp_primeOdYa,dYp_primeOdYLi7,dYp_primeOdYBe7]

        # Yd
        dYd_primeOdYn = rhoBBN*(self.npdg_frwrd(T_t)*Yn0p1 + 2.*self.ddHe3n_bkwrd(T_t)*Yn1p2 + self.tdan_bkwrd(T_t)*Yn2p2)
        dYd_primeOdYp = rhoBBN*(self.npdg_frwrd(T_t)*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn1p1 + 2.*self.ddtp_bkwrd(T_t)*Yn2p1 + self.He3dap_bkwrd(T_t)*Yn2p2)
        dYd_primeOdYd = rhoBBN*(- self.dpHe3g_frwrd(T_t)*Yn0p1 - 2.*(self.ddHe3n_frwrd(T_t) + self.ddtp_frwrd(T_t))*Yn1p1 - self.tdan_frwrd(T_t)*Yn2p1 - self.He3dap_frwrd(T_t)*Yn1p2) - self.npdg_bkwrd(T_t)
        dYd_primeOdYt = rhoBBN*(2.*self.ddtp_bkwrd(T_t)*Yn0p1 - self.tdan_frwrd(T_t)*Yn1p1)
        dYd_primeOdYHe3 = rhoBBN*(2.*self.ddHe3n_bkwrd(T_t)*Yn1p0 - self.He3dap_frwrd(T_t)*Yn1p1) + self.dpHe3g_bkwrd(T_t)
        dYd_primeOdYa = rhoBBN*(self.tdan_bkwrd(T_t)*Yn1p0 + self.He3dap_bkwrd(T_t)*Yn0p1)
        dYd_primeOdYLi7 = 0.
        dYd_primeOdYBe7 = 0.
        dYd_row = [dYd_primeOdYn,dYd_primeOdYp,dYd_primeOdYd,dYd_primeOdYt,dYd_primeOdYHe3,dYd_primeOdYa,dYd_primeOdYLi7,dYd_primeOdYBe7]

        # Yt
        dYt_primeOdYn = rhoBBN*(self.He3ntp_frwrd(T_t)*Yn1p2 + self.tdan_bkwrd(T_t)*Yn2p2)
        dYt_primeOdYp = -rhoBBN*(self.tpag_frwrd(T_t)+self.ddtp_bkwrd(T_t)+self.He3ntp_bkwrd(T_t))*Yn2p1
        dYt_primeOdYd = rhoBBN*(self.ddtp_frwrd(T_t)*Yn1p1 - self.tdan_frwrd(T_t)*Yn2p1)
        dYt_primeOdYt = -rhoBBN*((self.tpag_frwrd(T_t)+self.ddtp_bkwrd(T_t)+self.He3ntp_bkwrd(T_t))*Yn0p1 + self.tdan_frwrd(T_t)*Yn1p1 + self.taLi7g_frwrd(T_t)*Yn2p2)
        dYt_primeOdYHe3 = rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0
        dYt_primeOdYa = rhoBBN*(self.tdan_bkwrd(T_t)*Yn1p0 - self.taLi7g_frwrd(T_t)*Yn2p1) + self.tpag_bkwrd(T_t)
        dYt_primeOdYLi7 = self.taLi7g_bkwrd(T_t)
        dYt_primeOdYBe7 = 0.
        dYt_row = [dYt_primeOdYn,dYt_primeOdYp,dYt_primeOdYd,dYt_primeOdYt,dYt_primeOdYHe3,dYt_primeOdYa,dYt_primeOdYLi7,dYt_primeOdYBe7]

        # YHe3
        dYHe3_primeOdYn = -rhoBBN*(self.He3ntp_frwrd(T_t)+self.ddHe3n_bkwrd(T_t))*Yn1p2
        dYHe3_primeOdYp = rhoBBN*(self.dpHe3g_frwrd(T_t)*Yn1p1 + self.He3ntp_bkwrd(T_t)*Yn2p1 + self.He3dap_bkwrd(T_t)*Yn2p2)
        dYHe3_primeOdYd = rhoBBN*(self.dpHe3g_frwrd(T_t)*Yn0p1 + self.ddHe3n_frwrd(T_t)*Yn1p1 - self.He3dap_frwrd(T_t)*Yn1p2)
        dYHe3_primeOdYt = rhoBBN*(self.He3ntp_bkwrd(T_t)*Yn0p1)
        dYHe3_primeOdYHe3 = rhoBBN*(- self.He3dap_frwrd(T_t)*Yn1p1 - (self.He3ntp_frwrd(T_t)+self.ddHe3n_bkwrd(T_t))*Yn1p0 - self.He3aBe7g_frwrd(T_t)*Yn2p2) - self.dpHe3g_bkwrd(T_t)
        dYHe3_primeOdYa = rhoBBN*(self.He3dap_bkwrd(T_t)*Yn0p1 - self.He3aBe7g_frwrd(T_t)*Yn1p2)
        dYHe3_primeOdYLi7 = 0.
        dYHe3_primeOdYBe7 = self.He3aBe7g_bkwrd(T_t)
        dYHe3_row = [dYHe3_primeOdYn,dYHe3_primeOdYp,dYHe3_primeOdYd,dYHe3_primeOdYt,dYHe3_primeOdYHe3,dYHe3_primeOdYa,dYHe3_primeOdYLi7,dYHe3_primeOdYBe7]

        # Ya
        dYa_primeOdYn = -rhoBBN*self.tdan_bkwrd(T_t)*Yn2p2
        dYa_primeOdYp = rhoBBN*(- self.He3dap_bkwrd(T_t)*Yn2p2 + 2.*self.Li7paa_frwrd(T_t)*Yn4p3 + self.tpag_frwrd(T_t)*Yn2p1)
        dYa_primeOdYd = rhoBBN*(self.He3dap_frwrd(T_t)*Yn1p2 + self.tdan_frwrd(T_t)*Yn2p1)
        dYa_primeOdYt = rhoBBN*(- self.taLi7g_frwrd(T_t)*Yn2p2 + self.tdan_frwrd(T_t)*Yn1p1 + self.tpag_frwrd(T_t)*Yn0p1)
        dYa_primeOdYHe3 = rhoBBN*(- self.He3aBe7g_frwrd(T_t)*Yn2p2 + self.He3dap_frwrd(T_t)*Yn1p1)
        dYa_primeOdYa = -rhoBBN*(self.He3aBe7g_frwrd(T_t)*Yn1p2 + self.He3dap_bkwrd(T_t)*Yn0p1 + 2.*self.Li7paa_bkwrd(T_t)*Yn2p2 + self.taLi7g_frwrd(T_t)*Yn2p1 + self.tdan_bkwrd(T_t)*Yn1p0) - self.tpag_bkwrd(T_t)
        dYa_primeOdYLi7 = 2.*rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1+self.taLi7g_bkwrd(T_t)
        dYa_primeOdYBe7 = self.He3aBe7g_bkwrd(T_t)
        dYa_row = [dYa_primeOdYn,dYa_primeOdYp,dYa_primeOdYd,dYa_primeOdYt,dYa_primeOdYHe3,dYa_primeOdYa,dYa_primeOdYLi7,dYa_primeOdYBe7]

        # YLi7
        dYLi7_primeOdYn = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn3p4
        dYLi7_primeOdYp = -rhoBBN*(self.Be7nLi7p_bkwrd(T_t) + self.Li7paa_frwrd(T_t))*Yn4p3
        dYLi7_primeOdYd = 0.
        dYLi7_primeOdYt = rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p2
        dYLi7_primeOdYHe3 = 0.
        dYLi7_primeOdYa = rhoBBN*(self.Li7paa_bkwrd(T_t)*Yn2p2 + self.taLi7g_frwrd(T_t)*Yn2p1)
        dYLi7_primeOdYLi7 = -rhoBBN*(self.Be7nLi7p_bkwrd(T_t) + self.Li7paa_frwrd(T_t))*Yn0p1 - self.taLi7g_bkwrd(T_t)
        dYLi7_primeOdYBe7 = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0
        dYLi7_row = [dYLi7_primeOdYn,dYLi7_primeOdYp,dYLi7_primeOdYd,dYLi7_primeOdYt,dYLi7_primeOdYHe3,dYLi7_primeOdYa,dYLi7_primeOdYLi7,dYLi7_primeOdYBe7]

        # YBe7
        dYBe7_primeOdYn = -rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn3p4
        dYBe7_primeOdYp = rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn4p3
        dYBe7_primeOdYd = 0.
        dYBe7_primeOdYt = 0.
        dYBe7_primeOdYHe3 = rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn2p2
        dYBe7_primeOdYa = rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2
        dYBe7_primeOdYLi7 = rhoBBN* self.Be7nLi7p_bkwrd(T_t)*Yn0p1
        dYBe7_primeOdYBe7 = -rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0 - self.He3aBe7g_bkwrd(T_t)
        dYBe7_row = [dYBe7_primeOdYn,dYBe7_primeOdYp,dYBe7_primeOdYd,dYBe7_primeOdYt,dYBe7_primeOdYHe3,dYBe7_primeOdYa,dYBe7_primeOdYLi7,dYBe7_primeOdYBe7]

        return [dYn_row,dYp_row,dYd_row,dYt_row,dYHe3_row,dYa_row,dYLi7_row,dYBe7_row]
