# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini

if(PRyMini.verbose_flag):
    print("PRyM_nuclear_rates.py: Interpolating nuclear rates")
    print(" ")

class UpdateNuclearRates(object):
    def __init__(self,p_npdg,p_dpHe3g,p_ddHe3n,p_ddtp,p_tpag,p_tdan,p_taLi7g, p_He3ntp,p_He3dap,p_He3aBe7g,p_Be7nLi7p,p_Li7paa,p_Li7paag,p_Be7naa,p_Be7daap,p_daLi6g,p_Li6pBe7g,p_Li6pHe3a, p_B8naap, p_Li6He3aap, p_Li6taan, p_Li6tLi8p, p_Li7He3Li6a, p_Li8He3Li7a, p_Be7tLi6a, p_B8tBe7a, p_B8nLi6He3, p_B8nBe7d, p_Li6tLi7d, p_Li6He3Be7d, p_Li7He3aad, p_Li8He3aat, p_Be7taad, p_Be7tLi7He3, p_B8dBe7He3, p_B8taaHe3, p_Be7He3ppaa, p_ddag, p_He3He3app, p_Be7pB8g, p_Li7daan, p_dntg, p_ttann, p_He3nag, p_He3tad, p_He3tanp, p_Li7taan, p_Li7He3aanp, p_Li8dLi7t, p_Be7taanp, p_Be7He3aapp, p_Li6nta, p_He3tLi6g, p_anpLi6g, p_Li6nLi7g, p_Li6dLi7p, p_Li6dBe7n, p_Li7nLi8g, p_Li7dLi8p, p_Li8paan, p_annHe6g, p_ppndp, p_Li7taann):
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
        Li7paag_mu = PRyMini.Li7paag_median*np.exp(p_Li7paag*np.log(PRyMini.Li7paag_expsigma))
        Be7naa_mu = PRyMini.Be7naa_median*np.exp(p_Be7naa*np.log(PRyMini.Be7naa_expsigma))
        Be7daap_mu = PRyMini.Be7daap_median*np.exp(p_Be7daap*np.log(PRyMini.Be7daap_expsigma))
        daLi6g_mu = PRyMini.daLi6g_median*np.exp(p_daLi6g*np.log(PRyMini.daLi6g_expsigma))
        Li6pBe7g_mu = PRyMini.Li6pBe7g_median*np.exp(p_Li6pBe7g*np.log(PRyMini.Li6pBe7g_expsigma))
        Li6pHe3a_mu = PRyMini.Li6pHe3a_median*np.exp(p_Li6pHe3a*np.log(PRyMini.Li6pHe3a_expsigma))
        B8naap_mu = PRyMini.B8naap_median*np.exp(p_B8naap*np.log(PRyMini.B8naap_expsigma))
        Li6He3aap_mu = PRyMini.Li6He3aap_median*np.exp(p_Li6He3aap*np.log(PRyMini.Li6He3aap_expsigma))
        Li6taan_mu = PRyMini.Li6taan_median*np.exp(p_Li6taan*np.log(PRyMini.Li6taan_expsigma))
        Li6tLi8p_mu = PRyMini.Li6tLi8p_median*np.exp(p_Li6tLi8p*np.log(PRyMini.Li6tLi8p_expsigma))
        Li7He3Li6a_mu = PRyMini.Li7He3Li6a_median*np.exp(p_Li7He3Li6a*np.log(PRyMini.Li7He3Li6a_expsigma))
        Li8He3Li7a_mu = PRyMini.Li8He3Li7a_median*np.exp(p_Li8He3Li7a*np.log(PRyMini.Li8He3Li7a_expsigma))
        Be7tLi6a_mu = PRyMini.Be7tLi6a_median*np.exp(p_Be7tLi6a*np.log(PRyMini.Be7tLi6a_expsigma))
        B8tBe7a_mu = PRyMini.B8tBe7a_median*np.exp(p_B8tBe7a*np.log(PRyMini.B8tBe7a_expsigma))
        B8nLi6He3_mu = PRyMini.B8nLi6He3_median*np.exp(p_B8nLi6He3*np.log(PRyMini.B8nLi6He3_expsigma))
        B8nBe7d_mu = PRyMini.B8nBe7d_median*np.exp(p_B8nBe7d*np.log(PRyMini.B8nBe7d_expsigma))
        Li6tLi7d_mu = PRyMini.Li6tLi7d_median*np.exp(p_Li6tLi7d*np.log(PRyMini.Li6tLi7d_expsigma))
        Li6He3Be7d_mu = PRyMini.Li6He3Be7d_median*np.exp(p_Li6He3Be7d*np.log(PRyMini.Li6He3Be7d_expsigma))
        Li7He3aad_mu = PRyMini.Li7He3aad_median*np.exp(p_Li7He3aad*np.log(PRyMini.Li7He3aad_expsigma))
        Li8He3aat_mu = PRyMini.Li8He3aat_median*np.exp(p_Li8He3aat*np.log(PRyMini.Li8He3aat_expsigma))
        Be7taad_mu = PRyMini.Be7taad_median*np.exp(p_Be7taad*np.log(PRyMini.Be7taad_expsigma))
        Be7tLi7He3_mu = PRyMini.Be7tLi7He3_median*np.exp(p_Be7tLi7He3*np.log(PRyMini.Be7tLi7He3_expsigma))
        B8dBe7He3_mu = PRyMini.B8dBe7He3_median*np.exp(p_B8dBe7He3*np.log(PRyMini.B8dBe7He3_expsigma))
        B8taaHe3_mu = PRyMini.B8taaHe3_median*np.exp(p_B8taaHe3*np.log(PRyMini.B8taaHe3_expsigma))
        Be7He3ppaa_mu = PRyMini.Be7He3ppaa_median*np.exp(p_Be7He3ppaa*np.log(PRyMini.Be7He3ppaa_expsigma))
        ddag_mu = PRyMini.ddag_median*np.exp(p_ddag*np.log(PRyMini.ddag_expsigma))
        He3He3app_mu = PRyMini.He3He3app_median*np.exp(p_He3He3app*np.log(PRyMini.He3He3app_expsigma))
        Be7pB8g_mu = PRyMini.Be7pB8g_median*np.exp(p_Be7pB8g*np.log(PRyMini.Be7pB8g_expsigma))
        Li7daan_mu = PRyMini.Li7daan_median*np.exp(p_Li7daan*np.log(PRyMini.Li7daan_expsigma))
        dntg_mu = PRyMini.dntg_median*np.exp(p_dntg*np.log(PRyMini.dntg_expsigma))
        ttann_mu = PRyMini.ttann_median*np.exp(p_ttann*np.log(PRyMini.ttann_expsigma))
        He3nag_mu = PRyMini.He3nag_median*np.exp(p_He3nag*np.log(PRyMini.He3nag_expsigma))
        He3tad_mu = PRyMini.He3tad_median*np.exp(p_He3tad*np.log(PRyMini.He3tad_expsigma))
        He3tanp_mu = PRyMini.He3tanp_median*np.exp(p_He3tanp*np.log(PRyMini.He3tanp_expsigma))
        Li7taan_mu = PRyMini.Li7taan_median*np.exp(p_Li7taan*np.log(PRyMini.Li7taan_expsigma))
        Li7He3aanp_mu = PRyMini.Li7He3aanp_median*np.exp(p_Li7He3aanp*np.log(PRyMini.Li7He3aanp_expsigma))
        Li8dLi7t_mu = PRyMini.Li8dLi7t_median*np.exp(p_Li8dLi7t*np.log(PRyMini.Li8dLi7t_expsigma))
        Be7taanp_mu = PRyMini.Be7taanp_median*np.exp(p_Be7taanp*np.log(PRyMini.Be7taanp_expsigma))
        Be7He3aapp_mu = PRyMini.Be7He3aapp_median*np.exp(p_Be7He3aapp*np.log(PRyMini.Be7He3aapp_expsigma))
        Li6nta_mu = PRyMini.Li6nta_median*np.exp(p_Li6nta*np.log(PRyMini.Li6nta_expsigma))
        He3tLi6g_mu = PRyMini.He3tLi6g_median*np.exp(p_He3tLi6g*np.log(PRyMini.He3tLi6g_expsigma))
        anpLi6g_mu = PRyMini.anpLi6g_median*np.exp(p_anpLi6g*np.log(PRyMini.anpLi6g_expsigma))
        Li6nLi7g_mu = PRyMini.Li6nLi7g_median*np.exp(p_Li6nLi7g*np.log(PRyMini.Li6nLi7g_expsigma))
        Li6dLi7p_mu = PRyMini.Li6dLi7p_median*np.exp(p_Li6dLi7p*np.log(PRyMini.Li6dLi7p_expsigma))
        Li6dBe7n_mu = PRyMini.Li6dBe7n_median*np.exp(p_Li6dBe7n*np.log(PRyMini.Li6dBe7n_expsigma))
        Li7nLi8g_mu = PRyMini.Li7nLi8g_median*np.exp(p_Li7nLi8g*np.log(PRyMini.Li7nLi8g_expsigma))
        Li7dLi8p_mu = PRyMini.Li7dLi8p_median*np.exp(p_Li7dLi8p*np.log(PRyMini.Li7dLi8p_expsigma))
        Li8paan_mu = PRyMini.Li8paan_median*np.exp(p_Li8paan*np.log(PRyMini.Li8paan_expsigma))
        annHe6g_mu = PRyMini.annHe6g_median*np.exp(p_annHe6g*np.log(PRyMini.annHe6g_expsigma))
        ppndp_mu = PRyMini.ppndp_median*np.exp(p_ppndp*np.log(PRyMini.ppndp_expsigma))
        Li7taann_mu = PRyMini.Li7taann_median*np.exp(p_Li7taann*np.log(PRyMini.Li7taann_expsigma))
        if(PRyMini.NP_nuclear_flag):
            npdg_mu += PRyMini.delta_npdg*PRyMini.npdg_median
            dpHe3g_mu += PRyMini.delta_dpHe3g*PRyMini.dpHe3g_median
            ddHe3n_mu += PRyMini.delta_ddHe3n*PRyMini.ddHe3n_median
            ddtp_mu += PRyMini.delta_ddtp*PRyMini.ddtp_median
            tpag_mu += PRyMini.delta_tpag*PRyMini.tpag_median
            tdan_mu += PRyMini.delta_tdan*PRyMini.tdan_median
            taLi7g_mu += PRyMini.delta_taLi7g*PRyMini.taLi7g_median
            He3ntp_mu += PRyMini.delta_He3ntp*PRyMini.He3ntp_median
            He3dap_mu += PRyMini.delta_He3dap*PRyMini.He3dap_median
            He3aBe7g_mu += PRyMini.delta_He3aBe7g*PRyMini.He3aBe7g_median
            Be7nLi7p_mu += PRyMini.delta_Be7nLi7p*PRyMini.Be7nLi7p_median
            Li7paa_mu += PRyMini.delta_Li7paa*PRyMini.Li7paa_median
            Li7paag_mu += PRyMini.delta_Li7paag*PRyMini.Li7paag_median
            Be7naa_mu += PRyMini.delta_Be7naa*PRyMini.Be7naa_median
            Be7daap_mu += PRyMini.delta_Be7daap*PRyMini.Be7daap_median
            daLi6g_mu += PRyMini.delta_daLi6g*PRyMini.daLi6g_median
            Li6pBe7g_mu += PRyMini.delta_Li6pBe7g*PRyMini.Li6pBe7g_median
            Li6pHe3a_mu += PRyMini.delta_Li6pHe3a*PRyMini.Li6pHe3a_median
            B8naap_mu += PRyMini.delta_B8naap*PRyMini.B8naap_median
            Li6He3aap_mu += PRyMini.delta_Li6He3aap*PRyMini.Li6He3aap_median
            Li6taan_mu += PRyMini.delta_Li6taan*PRyMini.Li6taan_median
            Li6tLi8p_mu += PRyMini.delta_Li6tLi8p*PRyMini.Li6tLi8p_median
            Li7He3Li6a_mu += PRyMini.delta_Li7He3Li6a*PRyMini.Li7He3Li6a_median
            Li8He3Li7a_mu += PRyMini.delta_Li8He3Li7a*PRyMini.Li8He3Li7a_median
            Be7tLi6a_mu += PRyMini.delta_Be7tLi6a*PRyMini.Be7tLi6a_median
            B8tBe7a_mu += PRyMini.delta_B8tBe7a*PRyMini.B8tBe7a_median
            B8nLi6He3_mu += PRyMini.delta_B8nLi6He3*PRyMini.B8nLi6He3_median
            B8nBe7d_mu += PRyMini.delta_B8nBe7d*PRyMini.B8nBe7d_median
            Li6tLi7d_mu += PRyMini.delta_Li6tLi7d*PRyMini.Li6tLi7d_median
            Li6He3Be7d_mu += PRyMini.delta_Li6He3Be7d*PRyMini.Li6He3Be7d_median
            Li7He3aad_mu += PRyMini.delta_Li7He3aad*PRyMini.Li7He3aad_median
            Li8He3aat_mu += PRyMini.delta_Li8He3aat*PRyMini.Li8He3aat_median
            Be7taad_mu += PRyMini.delta_Be7taad*PRyMini.Be7taad_median
            Be7tLi7He3_mu += PRyMini.delta_Be7tLi7He3*PRyMini.Be7tLi7He3_median
            B8dBe7He3_mu += PRyMini.delta_B8dBe7He3*PRyMini.B8dBe7He3_median
            B8taaHe3_mu += PRyMini.delta_B8taaHe3*PRyMini.B8taaHe3_median
            Be7He3ppaa_mu += PRyMini.delta_Be7He3ppaa*PRyMini.Be7He3ppaa_median
            ddag_mu += PRyMini.delta_ddag*PRyMini.ddag_median
            He3He3app_mu += PRyMini.delta_He3He3app*PRyMini.He3He3app_median
            Be7pB8g_mu += PRyMini.delta_Be7pB8g*PRyMini.Be7pB8g_median
            Li7daan_mu += PRyMini.delta_Li7daan*PRyMini.Li7daan_median
            dntg_mu += PRyMini.delta_dntg*PRyMini.dntg_median
            ttann_mu += PRyMini.delta_ttann*PRyMini.ttann_median
            He3nag_mu += PRyMini.delta_He3nag*PRyMini.He3nag_median
            He3tad_mu += PRyMini.delta_He3tad*PRyMini.He3tad_median
            He3tanp_mu += PRyMini.delta_He3tanp*PRyMini.He3tanp_median
            Li7taan_mu += PRyMini.delta_Li7taan*PRyMini.Li7taan_median
            Li7He3aanp_mu += PRyMini.delta_Li7He3aanp*PRyMini.Li7He3aanp_median
            Li8dLi7t_mu += PRyMini.delta_Li8dLi7t*PRyMini.Li8dLi7t_median
            Be7taanp_mu += PRyMini.delta_Be7taanp*PRyMini.Be7taanp_median
            Be7He3aapp_mu += PRyMini.delta_Be7He3aapp*PRyMini.Be7He3aapp_median
            Li6nta_mu += PRyMini.delta_Li6nta*PRyMini.Li6nta_median
            He3tLi6g_mu += PRyMini.delta_He3tLi6g*PRyMini.He3tLi6g_median
            anpLi6g_mu += PRyMini.delta_anpLi6g*PRyMini.anpLi6g_median
            Li6nLi7g_mu += PRyMini.delta_Li6nLi7g*PRyMini.Li6nLi7g_median
            Li6dLi7p_mu += PRyMini.delta_Li6dLi7p*PRyMini.Li6dLi7p_median
            Li6dBe7n_mu += PRyMini.delta_Li6dBe7n*PRyMini.Li6dBe7n_median
            Li7nLi8g_mu += PRyMini.delta_Li7nLi8g*PRyMini.Li7nLi8g_median
            Li7dLi8p_mu += PRyMini.delta_Li7dLi8p*PRyMini.Li7dLi8p_median
            Li8paan_mu += PRyMini.delta_Li8paan*PRyMini.Li8paan_median
            annHe6g_mu += PRyMini.delta_annHe6g*PRyMini.annHe6g_median
            ppndp_mu += PRyMini.delta_ppndp*PRyMini.ppndp_median
            Li7taann_mu += PRyMini.delta_Li7taann*PRyMini.Li7taann_median
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
        self.Li7paag_spline = interp1d(PRyMini.Li7paag_T9,Li7paag_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7naa_spline = interp1d(PRyMini.Be7naa_T9,Be7naa_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7daap_spline = interp1d(PRyMini.Be7daap_T9,Be7daap_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.daLi6g_spline = interp1d(PRyMini.daLi6g_T9,daLi6g_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6pBe7g_spline = interp1d(PRyMini.Li6pBe7g_T9,Li6pBe7g_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6pHe3a_spline = interp1d(PRyMini.Li6pHe3a_T9,Li6pHe3a_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.B8naap_spline = interp1d(PRyMini.B8naap_T9,B8naap_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6He3aap_spline = interp1d(PRyMini.Li6He3aap_T9,Li6He3aap_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6taan_spline = interp1d(PRyMini.Li6taan_T9,Li6taan_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6tLi8p_spline = interp1d(PRyMini.Li6tLi8p_T9,Li6tLi8p_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li7He3Li6a_spline = interp1d(PRyMini.Li7He3Li6a_T9,Li7He3Li6a_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li8He3Li7a_spline = interp1d(PRyMini.Li8He3Li7a_T9,Li8He3Li7a_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7tLi6a_spline = interp1d(PRyMini.Be7tLi6a_T9,Be7tLi6a_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.B8tBe7a_spline = interp1d(PRyMini.B8tBe7a_T9,B8tBe7a_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.B8nLi6He3_spline = interp1d(PRyMini.B8nLi6He3_T9,B8nLi6He3_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.B8nBe7d_spline = interp1d(PRyMini.B8nBe7d_T9,B8nBe7d_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6tLi7d_spline = interp1d(PRyMini.Li6tLi7d_T9,Li6tLi7d_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li6He3Be7d_spline = interp1d(PRyMini.Li6He3Be7d_T9,Li6He3Be7d_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li7He3aad_spline = interp1d(PRyMini.Li7He3aad_T9,Li7He3aad_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li8He3aat_spline = interp1d(PRyMini.Li8He3aat_T9,Li8He3aat_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7taad_spline = interp1d(PRyMini.Be7taad_T9,Be7taad_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7tLi7He3_spline = interp1d(PRyMini.Be7tLi7He3_T9,Be7tLi7He3_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.B8dBe7He3_spline = interp1d(PRyMini.B8dBe7He3_T9,B8dBe7He3_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.B8taaHe3_spline = interp1d(PRyMini.B8taaHe3_T9,B8taaHe3_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7He3ppaa_spline = interp1d(PRyMini.Be7He3ppaa_T9,Be7He3ppaa_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.ddag_spline = interp1d(PRyMini.ddag_T9,ddag_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.He3He3app_spline = interp1d(PRyMini.He3He3app_T9,He3He3app_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Be7pB8g_spline = interp1d(PRyMini.Be7pB8g_T9,Be7pB8g_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.Li7daan_spline = interp1d(PRyMini.Li7daan_T9,Li7daan_mu,bounds_error=False,fill_value="extrapolate",kind='linear')
        self.dntg_spline = interp1d(PRyMini.dntg_T9,dntg_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.ttann_spline = interp1d(PRyMini.ttann_T9,ttann_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.He3nag_spline = interp1d(PRyMini.He3nag_T9,He3nag_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.He3tad_spline = interp1d(PRyMini.He3tad_T9,He3tad_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.He3tanp_spline = interp1d(PRyMini.He3tanp_T9,He3tanp_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li7taan_spline = interp1d(PRyMini.Li7taan_T9,Li7taan_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li7He3aanp_spline = interp1d(PRyMini.Li7He3aanp_T9,Li7He3aanp_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li8dLi7t_spline = interp1d(PRyMini.Li8dLi7t_T9,Li8dLi7t_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Be7taanp_spline = interp1d(PRyMini.Be7taanp_T9,Be7taanp_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Be7He3aapp_spline = interp1d(PRyMini.Be7He3aapp_T9,Be7He3aapp_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li6nta_spline = interp1d(PRyMini.Li6nta_T9,Li6nta_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.He3tLi6g_spline = interp1d(PRyMini.He3tLi6g_T9,He3tLi6g_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.anpLi6g_spline = interp1d(PRyMini.anpLi6g_T9,anpLi6g_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li6nLi7g_spline = interp1d(PRyMini.Li6nLi7g_T9,Li6nLi7g_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li6dLi7p_spline = interp1d(PRyMini.Li6dLi7p_T9,Li6dLi7p_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li6dBe7n_spline = interp1d(PRyMini.Li6dBe7n_T9,Li6dBe7n_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li7nLi8g_spline = interp1d(PRyMini.Li7nLi8g_T9,Li7nLi8g_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li7dLi8p_spline = interp1d(PRyMini.Li7dLi8p_T9,Li7dLi8p_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li8paan_spline = interp1d(PRyMini.Li8paan_T9,Li8paan_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.annHe6g_spline = interp1d(PRyMini.annHe6g_T9,annHe6g_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.ppndp_spline = interp1d(PRyMini.ppndp_T9,ppndp_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        self.Li7taann_spline = interp1d(PRyMini.Li7taann_T9,Li7taann_mu,bounds_error=False,fill_value="extrapolate",kind='quadratic')
    
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

    # Li7 p --> a a g
    def Li7paag_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7paag_spline(T9)
    def Li7paag_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7paag
        beta = PRyMini.beta_Li7paag
        gamma = PRyMini.gamma_Li7paag
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7paag_spline(T9)

    def Be7naa_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7naa_spline(T9)
    def Be7naa_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7naa
        beta = PRyMini.beta_Be7naa
        gamma = PRyMini.gamma_Be7naa
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7naa_spline(T9)

    def Be7daap_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7daap_spline(T9)
    def Be7daap_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7daap
        beta = PRyMini.beta_Be7daap
        gamma = PRyMini.gamma_Be7daap
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7daap_spline(T9)

    def daLi6g_frwrd(self,T):
        T9 = T*1.e-9
        return self.daLi6g_spline(T9)
    def daLi6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_daLi6g
        beta = PRyMini.beta_daLi6g
        gamma = PRyMini.gamma_daLi6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.daLi6g_spline(T9)

    def Li6pBe7g_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6pBe7g_spline(T9)
    def Li6pBe7g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6pBe7g
        beta = PRyMini.beta_Li6pBe7g
        gamma = PRyMini.gamma_Li6pBe7g
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6pBe7g_spline(T9)

    def Li6pHe3a_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6pHe3a_spline(T9)
    def Li6pHe3a_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6pHe3a
        beta = PRyMini.beta_Li6pHe3a
        gamma = PRyMini.gamma_Li6pHe3a
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6pHe3a_spline(T9)

    def B8naap_frwrd(self,T):
        T9 = T*1.e-9
        return self.B8naap_spline(T9)
    def B8naap_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_B8naap
        beta = PRyMini.beta_B8naap
        gamma = PRyMini.gamma_B8naap
        return alpha*T9**beta*np.exp(gamma/T9)*self.B8naap_spline(T9)

    def Li6He3aap_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6He3aap_spline(T9)
    def Li6He3aap_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6He3aap
        beta = PRyMini.beta_Li6He3aap
        gamma = PRyMini.gamma_Li6He3aap
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6He3aap_spline(T9)

    def Li6taan_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6taan_spline(T9)
    def Li6taan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6taan
        beta = PRyMini.beta_Li6taan
        gamma = PRyMini.gamma_Li6taan
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6taan_spline(T9)

    def Li6tLi8p_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6tLi8p_spline(T9)
    def Li6tLi8p_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6tLi8p
        beta = PRyMini.beta_Li6tLi8p
        gamma = PRyMini.gamma_Li6tLi8p
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6tLi8p_spline(T9)

    def Li7He3Li6a_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7He3Li6a_spline(T9)
    def Li7He3Li6a_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7He3Li6a
        beta = PRyMini.beta_Li7He3Li6a
        gamma = PRyMini.gamma_Li7He3Li6a
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7He3Li6a_spline(T9)

    def Li8He3Li7a_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li8He3Li7a_spline(T9)
    def Li8He3Li7a_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li8He3Li7a
        beta = PRyMini.beta_Li8He3Li7a
        gamma = PRyMini.gamma_Li8He3Li7a
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li8He3Li7a_spline(T9)

    def Be7tLi6a_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7tLi6a_spline(T9)
    def Be7tLi6a_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7tLi6a
        beta = PRyMini.beta_Be7tLi6a
        gamma = PRyMini.gamma_Be7tLi6a
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7tLi6a_spline(T9)

    def B8tBe7a_frwrd(self,T):
        T9 = T*1.e-9
        return self.B8tBe7a_spline(T9)
    def B8tBe7a_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_B8tBe7a
        beta = PRyMini.beta_B8tBe7a
        gamma = PRyMini.gamma_B8tBe7a
        return alpha*T9**beta*np.exp(gamma/T9)*self.B8tBe7a_spline(T9)

    def B8nLi6He3_frwrd(self,T):
        T9 = T*1.e-9
        return self.B8nLi6He3_spline(T9)
    def B8nLi6He3_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_B8nLi6He3
        beta = PRyMini.beta_B8nLi6He3
        gamma = PRyMini.gamma_B8nLi6He3
        return alpha*T9**beta*np.exp(gamma/T9)*self.B8nLi6He3_spline(T9)

    def B8nBe7d_frwrd(self,T):
        T9 = T*1.e-9
        return self.B8nBe7d_spline(T9)
    def B8nBe7d_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_B8nBe7d
        beta = PRyMini.beta_B8nBe7d
        gamma = PRyMini.gamma_B8nBe7d
        return alpha*T9**beta*np.exp(gamma/T9)*self.B8nBe7d_spline(T9)

    def Li6tLi7d_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6tLi7d_spline(T9)
    def Li6tLi7d_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6tLi7d
        beta = PRyMini.beta_Li6tLi7d
        gamma = PRyMini.gamma_Li6tLi7d
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6tLi7d_spline(T9)

    def Li6He3Be7d_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6He3Be7d_spline(T9)
    def Li6He3Be7d_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6He3Be7d
        beta = PRyMini.beta_Li6He3Be7d
        gamma = PRyMini.gamma_Li6He3Be7d
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6He3Be7d_spline(T9)

    def Li7He3aad_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7He3aad_spline(T9)
    def Li7He3aad_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7He3aad
        beta = PRyMini.beta_Li7He3aad
        gamma = PRyMini.gamma_Li7He3aad
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7He3aad_spline(T9)

    def Li8He3aat_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li8He3aat_spline(T9)
    def Li8He3aat_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li8He3aat
        beta = PRyMini.beta_Li8He3aat
        gamma = PRyMini.gamma_Li8He3aat
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li8He3aat_spline(T9)

    def Be7taad_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7taad_spline(T9)
    def Be7taad_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7taad
        beta = PRyMini.beta_Be7taad
        gamma = PRyMini.gamma_Be7taad
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7taad_spline(T9)

    def Be7tLi7He3_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7tLi7He3_spline(T9)
    def Be7tLi7He3_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7tLi7He3
        beta = PRyMini.beta_Be7tLi7He3
        gamma = PRyMini.gamma_Be7tLi7He3
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7tLi7He3_spline(T9)

    def B8dBe7He3_frwrd(self,T):
        T9 = T*1.e-9
        return self.B8dBe7He3_spline(T9)
    def B8dBe7He3_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_B8dBe7He3
        beta = PRyMini.beta_B8dBe7He3
        gamma = PRyMini.gamma_B8dBe7He3
        return alpha*T9**beta*np.exp(gamma/T9)*self.B8dBe7He3_spline(T9)

    def B8taaHe3_frwrd(self,T):
        T9 = T*1.e-9
        return self.B8taaHe3_spline(T9)
    def B8taaHe3_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_B8taaHe3
        beta = PRyMini.beta_B8taaHe3
        gamma = PRyMini.gamma_B8taaHe3
        return alpha*T9**beta*np.exp(gamma/T9)*self.B8taaHe3_spline(T9)

    def Be7He3ppaa_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7He3ppaa_spline(T9)
    def Be7He3ppaa_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7He3ppaa
        beta = PRyMini.beta_Be7He3ppaa
        gamma = PRyMini.gamma_Be7He3ppaa
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7He3ppaa_spline(T9)

    def ddag_frwrd(self,T):
        T9 = T*1.e-9
        return self.ddag_spline(T9)
    def ddag_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_ddag
        beta = PRyMini.beta_ddag
        gamma = PRyMini.gamma_ddag
        return alpha*T9**beta*np.exp(gamma/T9)*self.ddag_spline(T9)

    def He3He3app_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3He3app_spline(T9)
    def He3He3app_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3He3app
        beta = PRyMini.beta_He3He3app
        gamma = PRyMini.gamma_He3He3app
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3He3app_spline(T9)

    def Be7pB8g_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7pB8g_spline(T9)
    def Be7pB8g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7pB8g
        beta = PRyMini.beta_Be7pB8g
        gamma = PRyMini.gamma_Be7pB8g
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7pB8g_spline(T9)

    def Li7daan_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7daan_spline(T9)
    def Li7daan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7daan
        beta = PRyMini.beta_Li7daan
        gamma = PRyMini.gamma_Li7daan
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7daan_spline(T9)

    def dntg_frwrd(self,T):
        T9 = T*1.e-9
        return 214.*T9**0.075 + 7.42*T9
    def dntg_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_dntg = 1.6364262*10**10
        beta_dntg = 1.5
        gamma_dntg = -72.612132
        alpha = alpha_dntg
        beta = beta_dntg
        gamma = gamma_dntg
        return alpha*T9**beta*np.exp(gamma/T9)*self.dntg_frwrd(T)
        
    def ttann_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2./3.)
        T913 = T9**(1./3.)
        T943 = T9**(4./3.)
        T953 = T9**(5./3.)
        return 1/T923*1.67*10**9*np.exp(-4.872/T913)*(1. - 0.272*T9 + 0.086*T913 - 0.455*T923 + 0.148*T943 + 0.225*T953)
    def ttann_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_ttann = 3.3826187*10**-10
        beta_ttann = -1.5
        gamma_ttann = -131.50322
        alpha = alpha_ttann
        beta = beta_ttann
        gamma = gamma_ttann
        return alpha*T9**beta*np.exp(gamma/T9)*self.ttann_frwrd(T)

    def He3nag_frwrd(self,T):
        T9 = T*1.e-9
        return 6.62*(1 + 905*T9)
    def He3nag_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_He3nag = 2.6152351*10**10
        beta_He3nag = 1.5
        gamma_He3nag = -238.79338
        alpha = alpha_He3nag
        beta = beta_He3nag
        gamma = gamma_He3nag
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3nag_frwrd(T)
        
    def He3tad_frwrd(self,T):
        T9 = T*1.e-9
        T9A = T9/(1. + 0.128*T9)
        T932 = T9**(3/2)
        T9A13 = T9A**(1./3.)
        T9A56 = T9A**(5./6.)
        return 5.46*10**9*T9A56/T932*np.exp(-7.733/T9A13)
    def He3tad_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_He3tad = 1.5981381
        beta_He3tad = 0.
        gamma_He3tad = -166.18124
        alpha = alpha_He3tad
        beta = beta_He3tad
        gamma = gamma_He3tad
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3tad_frwrd(T)

    def He3tanp_frwrd(self,T):
        T9 = T*1.e-9
        T9A = T9/(1. + 0.115*T9)
        T932 = T9**(3/2)
        T9A13 = T9A**(1./3.)
        T9A56 = T9A**(5./6.)
        return 7.71*10**9*T9A56/T932*np.exp(-7.733/T9A13)
    def He3tanp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_He3tanp = 3.3886566*10**-10
        beta_He3tanp = -1.5
        gamma_He3tanp = -140.36623
        alpha = alpha_He3tanp
        beta = beta_He3tanp
        gamma = gamma_He3tanp
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3tanp_frwrd(T)

    def Li7taan_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 8.81*10**11/T923*np.exp(-11.333/T913)
    def Li7taan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li7taan = 1.2153497*10**-19
        beta_Li7taan = -3.
        gamma_Li7taan = -102.86767
        alpha = alpha_Li7taan
        beta = beta_Li7taan
        gamma = gamma_Li7taan
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7taan_frwrd(T)
        
    def Li7He3aanp_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 1.11*10**13/T923*np.exp(-17.989/T913)
    def Li7He3aanp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li7He3aanp = 6.0875952*10**-20
        beta_Li7He3aanp = -3.
        gamma_Li7He3aanp = -111.73068
        alpha = alpha_Li7He3aanp
        beta = beta_Li7He3aanp
        gamma = gamma_Li7He3aanp
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7He3aanp_frwrd(T)
        
    def Li8dLi7t_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 3.02*10**8/T9**0.624*np.exp(-3.51/T9) + 5.82*10**11/T923*np.exp(-19.72/T913)*(1.0 + 0.280*T913)
    def Li8dLi7t_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li8dLi7t = 1.2509926
        beta_Li8dLi7t = 0.
        gamma_Li8dLi7t = -49.02453
        alpha = alpha_Li8dLi7t
        beta = beta_Li8dLi7t
        gamma = gamma_Li8dLi7t
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li8dLi7t_frwrd(T)
        
    def Be7taanp_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 2.91*10**12/T923*np.exp(-13.729/T913)
    def Be7taanp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Be7taanp = 6.0898077*10**-20
        beta_Be7taanp = -3.
        gamma_Be7taanp =-121.9483
        alpha = alpha_Be7taanp
        beta = beta_Be7taanp
        gamma = gamma_Be7taanp
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7taanp_frwrd(T)
        
    def Be7He3aapp_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 6.11*10**13/T923*np.exp(-21.793/T913)
    def Be7He3aapp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Be7He3aapp = 1.2201356*10**-19
        beta_Be7He3aapp = -3.
        gamma_Be7He3aapp = -130.81131
        alpha = alpha_Be7He3aapp
        beta = beta_Be7He3aapp
        gamma = gamma_Be7He3aapp
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7He3aapp_frwrd(T)
        
    def Li6nta_frwrd(self,T):
        T9 = T*1.e-9
        T9A = T9/(1. + 49.18*T9)
        T9A32 = T9A**(3./2.)
        T932 = T9**(3/2)
        return 1.80*10**8*(1. - .261*T9A32/T932)*.935 + 2.72*10**9/T932*np.exp((55.494 - 57.884)/T9)*.935
    def Li6nta_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li6nta = 1.0691921
        beta_Li6nta = 0.
        gamma_Li6nta = -55.509875
        alpha = alpha_Li6nta
        beta = beta_Li6nta
        gamma = gamma_Li6nta
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6nta_frwrd(T)
        
    def He3tLi6g_frwrd(self,T):
        T9 = T*1.e-9
        T92 = T9**2
        T923 = T9**(2/3)
        T932 = T9**(3/2)
        T913 = T9**(1/3)
        T943 = T9**(4/3)
        T953 = T9**(5/3)
        return 2.21*10**5/T923*np.exp(-7.720/T913)*(1. + 2.68*T923 + 0.868*T9 + 0.192*T943 + 0.174*T953 + 0.044*T92)
    def He3tLi6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_He3tLi6g = 2.4459918*10**10
        beta_He3tLi6g = 1.5
        gamma_He3tLi6g = -183.2835
        alpha = alpha_He3tLi6g
        beta = beta_He3tLi6g
        gamma = gamma_He3tLi6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3tLi6g_frwrd(T)
        
    def anpLi6g_frwrd(self,T):
        T9 = T*1.e-9
        if (T9 > 1):
            return 4.62*10**-6/T9**2*(1. + 0.075*T9)*np.exp(-19.353/T9)
        else:
            return 0.
    def anpLi6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_anpLi6g = 7.2181753*10**19
        beta_anpLi6g = 3.
        gamma_anpLi6g = -42.917276
        alpha = alpha_anpLi6g
        beta = beta_anpLi6g
        gamma = gamma_anpLi6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.anpLi6g_frwrd(T)
        
    def Li6nLi7g_frwrd(self,T):
        T9 = T*1.e-9
        return 5.10*10**3
    def Li6nLi7g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li6nLi7g = 1.1903305*10**10
        beta_Li6nLi7g = 1.5
        gamma_Li6nLi7g = -84.145424
        alpha = alpha_Li6nLi7g
        beta = beta_Li6nLi7g
        gamma = gamma_Li6nLi7g
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6nLi7g_frwrd(T)
        
    def Li6dLi7p_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return  1.48*10**12/T923*np.exp(-10.135/T913)
    def Li6dLi7p_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li6dLi7p = 2.5239503
        beta_Li6dLi7p = 0.
        gamma_Li6dLi7p = -58.330405
        alpha = alpha_Li6dLi7p
        beta = beta_Li6dLi7p
        gamma = gamma_Li6dLi7p
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6dLi7p_frwrd(T)
        
    def Li6dBe7n_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 1.48*10**12/T923*np.exp(-10.135/T913)
    def Li6dBe7n_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li6dBe7n = 2.5185377
        beta_Li6dBe7n = 0.
        gamma_Li6dBe7n = -39.249773
        alpha = alpha_Li6dBe7n
        beta = beta_Li6dBe7n
        gamma = gamma_Li6dBe7n
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6dBe7n_frwrd(T)
        
    def Li7nLi8g_frwrd(self,T):
        T9 = T*1.e-9
        T932 = T9**(3/2)
        return 6.015*10**3 + 1.141*10**4/T932*np.exp(-2.576/T9)
    def Li7nLi8g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li7nLi8g = 1.3081022*10**10
        beta_Li7nLi8g = 1.5
        gamma_Li7nLi8g = -23.587602
        alpha = alpha_Li7nLi8g
        beta = beta_Li7nLi8g
        gamma = gamma_Li7nLi8g
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7nLi8g_frwrd(T)

    def Li7dLi8p_frwrd(self,T):
        T9 = T*1.e-9
        T932 = T9**(3/2)
        return 8.31*10**8/T932*np.exp(-6.998/T9)
    def Li7dLi8p_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li7dLi8p = 2.7736709
        beta_Li7dLi8p = 0.
        gamma_Li7dLi8p = 2.2274166
        alpha = alpha_Li7dLi8p
        beta = beta_Li7dLi8p
        gamma = gamma_Li7dLi8p
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7dLi8p_frwrd(T)

    def Li8paan_frwrd(self,T):
        T9 = T*1.e-9
        T932 = T9**(3/2)
        T913 = T9**(1/3)
        T923 = T9**(2/3)
        T92 = T9**2
        T93 = T9**3
        T94 = T9**4
        T95 = T9**5
        if (T9 < 5):
            return 5.36*10**8/T932*np.exp(-4.41/T9) + 1.99*10**8/T932*np.exp(-7.08/T9) + 5.85*10**10/T923*np.exp(-8.50/T913)* (1. - 1.70*T9 + 0.849*T92 - 0.175*T93 + 1.62*10**-2*T94 - 5.60*10**-4*T95)
        else:
            return 7.777*10**7
    def Li8paan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li8paan = 3.5851946*10**-10
        beta_Li8paan = -1.5
        gamma_Li8paan = -177.70722
        alpha = alpha_Li8paan
        beta = beta_Li8paan
        gamma = gamma_Li8paan
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li8paan_frwrd(T)

    def annHe6g_frwrd(self,T):
        T9 = T*1.e-9
        if (T9 < 2):
            return 2.65*10**-3*T9**2.555*np.exp(0.181/np.maximum(T9, .1))
        else:
            return 2.93*10**-1*T9**(-3.51*10**-1)*np.exp(-5.24/T9)
    def annHe6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_annHe6g = 1.0837999*10**20
        beta_annHe6g = 3.
        gamma_annHe6g = -11.319626
        alpha = alpha_annHe6g
        beta = beta_annHe6g
        gamma = gamma_annHe6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.annHe6g_frwrd(T)

    def ppndp_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 1.35*10**7*np.exp(-3.720/T913)*(1. + 0.784*T913 + 0.346*T923 + 0.690*T9)/(2.3590*10**9)
    def ppndp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_ppndp = 2.3580703*10**9
        beta_ppndp = 1.5
        gamma_ppndp = -25.815019
        alpha = alpha_ppndp
        beta = beta_ppndp
        gamma = gamma_ppndp
        return alpha*T9**beta*np.exp(gamma/T9)*self.ppndp_frwrd(T)

    def Li7taann_frwrd(self,T):
        T9 = T*1.e-9
        T923 = T9**(2/3)
        T913 = T9**(1/3)
        return 8.81*10**11/T923*np.exp(-11.333/T913)
    def Li7taann_bkwrd(self,T):
        T9 = T*1.e-9
        alpha_Li7taann = 1.2153497*10**-19
        beta_Li7taann = -3.
        gamma_Li7taann = -102.86767
        alpha = alpha_Li7taann
        beta = beta_Li7taann
        gamma = gamma_Li7taann
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7taann_frwrd(T)
        
    def dntg_frwrd(self,T):
        T9 = T*1.e-9
        return self.dntg_spline(T9)
    def dntg_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_dntg
        beta = PRyMini.beta_dntg
        gamma = PRyMini.gamma_dntg
        return alpha*T9**beta*np.exp(gamma/T9)*self.dntg_spline(T9)

    def ttann_frwrd(self,T):
        T9 = T*1.e-9
        return self.ttann_spline(T9)
    def ttann_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_ttann
        beta = PRyMini.beta_ttann
        gamma = PRyMini.gamma_ttann
        return alpha*T9**beta*np.exp(gamma/T9)*self.ttann_spline(T9)

    def He3nag_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3nag_spline(T9)
    def He3nag_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3nag
        beta = PRyMini.beta_He3nag
        gamma = PRyMini.gamma_He3nag
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3nag_spline(T9)

    def He3tad_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3tad_spline(T9)
    def He3tad_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3tad
        beta = PRyMini.beta_He3tad
        gamma = PRyMini.gamma_He3tad
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3tad_spline(T9)

    def He3tanp_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3tanp_spline(T9)
    def He3tanp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3tanp
        beta = PRyMini.beta_He3tanp
        gamma = PRyMini.gamma_He3tanp
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3tanp_spline(T9)

    def Li7taan_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7taan_spline(T9)
    def Li7taan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7taan
        beta = PRyMini.beta_Li7taan
        gamma = PRyMini.gamma_Li7taan
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7taan_spline(T9)

    def Li7He3aanp_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7He3aanp_spline(T9)
    def Li7He3aanp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7He3aanp
        beta = PRyMini.beta_Li7He3aanp
        gamma = PRyMini.gamma_Li7He3aanp
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7He3aanp_spline(T9)

    def Li8dLi7t_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li8dLi7t_spline(T9)
    def Li8dLi7t_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li8dLi7t
        beta = PRyMini.beta_Li8dLi7t
        gamma = PRyMini.gamma_Li8dLi7t
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li8dLi7t_spline(T9)

    def Be7taanp_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7taanp_spline(T9)
    def Be7taanp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7taanp
        beta = PRyMini.beta_Be7taanp
        gamma = PRyMini.gamma_Be7taanp
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7taanp_spline(T9)

    def Be7He3aapp_frwrd(self,T):
        T9 = T*1.e-9
        return self.Be7He3aapp_spline(T9)
    def Be7He3aapp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Be7He3aapp
        beta = PRyMini.beta_Be7He3aapp
        gamma = PRyMini.gamma_Be7He3aapp
        return alpha*T9**beta*np.exp(gamma/T9)*self.Be7He3aapp_spline(T9)

    def Li6nta_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6nta_spline(T9)
    def Li6nta_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6nta
        beta = PRyMini.beta_Li6nta
        gamma = PRyMini.gamma_Li6nta
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6nta_spline(T9)

    def He3tLi6g_frwrd(self,T):
        T9 = T*1.e-9
        return self.He3tLi6g_spline(T9)
    def He3tLi6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_He3tLi6g
        beta = PRyMini.beta_He3tLi6g
        gamma = PRyMini.gamma_He3tLi6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.He3tLi6g_spline(T9)

    def anpLi6g_frwrd(self,T):
        T9 = T*1.e-9
        return self.anpLi6g_spline(T9)
    def anpLi6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_anpLi6g
        beta = PRyMini.beta_anpLi6g
        gamma = PRyMini.gamma_anpLi6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.anpLi6g_spline(T9)

    def Li6nLi7g_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6nLi7g_spline(T9)
    def Li6nLi7g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6nLi7g
        beta = PRyMini.beta_Li6nLi7g
        gamma = PRyMini.gamma_Li6nLi7g
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6nLi7g_spline(T9)

    def Li6dLi7p_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6dLi7p_spline(T9)
    def Li6dLi7p_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6dLi7p
        beta = PRyMini.beta_Li6dLi7p
        gamma = PRyMini.gamma_Li6dLi7p
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6dLi7p_spline(T9)

    def Li6dBe7n_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li6dBe7n_spline(T9)
    def Li6dBe7n_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li6dBe7n
        beta = PRyMini.beta_Li6dBe7n
        gamma = PRyMini.gamma_Li6dBe7n
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li6dBe7n_spline(T9)

    def Li7nLi8g_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7nLi8g_spline(T9)
    def Li7nLi8g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7nLi8g
        beta = PRyMini.beta_Li7nLi8g
        gamma = PRyMini.gamma_Li7nLi8g
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7nLi8g_spline(T9)

    def Li7dLi8p_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7dLi8p_spline(T9)
    def Li7dLi8p_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7dLi8p
        beta = PRyMini.beta_Li7dLi8p
        gamma = PRyMini.gamma_Li7dLi8p
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7dLi8p_spline(T9)

    def Li8paan_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li8paan_spline(T9)
    def Li8paan_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li8paan
        beta = PRyMini.beta_Li8paan
        gamma = PRyMini.gamma_Li8paan
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li8paan_spline(T9)

    def annHe6g_frwrd(self,T):
        T9 = T*1.e-9
        return self.annHe6g_spline(T9)
    def annHe6g_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_annHe6g
        beta = PRyMini.beta_annHe6g
        gamma = PRyMini.gamma_annHe6g
        return alpha*T9**beta*np.exp(gamma/T9)*self.annHe6g_spline(T9)

    def ppndp_frwrd(self,T):
        T9 = T*1.e-9
        return self.ppndp_spline(T9)
    def ppndp_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_ppndp
        beta = PRyMini.beta_ppndp
        gamma = PRyMini.gamma_ppndp
        return alpha*T9**beta*np.exp(gamma/T9)*self.ppndp_spline(T9)

    def Li7taann_frwrd(self,T):
        T9 = T*1.e-9
        return self.Li7taann_spline(T9)
    def Li7taann_bkwrd(self,T):
        T9 = T*1.e-9
        alpha = PRyMini.alpha_Li7taann
        beta = PRyMini.beta_Li7taann
        gamma = PRyMini.gamma_Li7taann
        return alpha*T9**beta*np.exp(gamma/T9)*self.Li7taann_spline(T9)
        
    def dYndtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return -nTOp_frwrd(T_t)*Yn1p0 + nTOp_bkwrd(T_t)*Yn0p1 + rhoBBN*(0.5*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 - self.npdg_frwrd(T_t)*Yn0p1*Yn1p0 + self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 + self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 - (self.He3ntp_frwrd(T_t) + self.ddHe3n_bkwrd(T_t))*Yn1p0*Yn1p2 - self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 + self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4) + self.npdg_bkwrd(T_t)*Yn1p1 + rhoBBN*(-self.Be7naa_frwrd(T_t)*Yn1p0*Yn3p4) + rhoBBN*(0.5*self.Be7naa_bkwrd(T_t)*Yn2p2*Yn2p2)

    def dYpdtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return nTOp_frwrd(T_t)*Yn1p0 - nTOp_bkwrd(T_t)*Yn0p1 + rhoBBN*(0.5*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - self.npdg_frwrd(T_t)*Yn0p1*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 - (self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn0p1*Yn2p1 + self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 + self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 - self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 + 0.5*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 - (self.Li7paa_frwrd(T_t) + self.Be7nLi7p_bkwrd(T_t))*Yn0p1*Yn4p3 + self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4) + self.npdg_bkwrd(T_t)*Yn1p1 + self.dpHe3g_bkwrd(T_t)*Yn1p2 + self.tpag_bkwrd(T_t)*Yn2p2 + rhoBBN*(- 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 + self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4) + rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn0p1*Yn3p3) + self.Li6pBe7g_bkwrd(T_t)*Yn3p4 + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn0p1*Yn4p3) + 0.5*rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*Yn2p2

    def dYddtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*(self.npdg_frwrd(T_t)*Yn0p1*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 - (self.ddHe3n_frwrd(T_t) + self.ddtp_frwrd(T_t))*Yn1p1*Yn1p1 + 2.*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + 2.*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 - self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 + self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2) - self.npdg_bkwrd(T_t)*Yn1p1 + self.dpHe3g_bkwrd(T_t)*Yn1p2 + rhoBBN*(-self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4 + 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2) + rhoBBN*(-self.daLi6g_frwrd(T_t)*Yn1p1*Yn2p2) + self.daLi6g_bkwrd(T_t)*Yn3p3

    def dYtdtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*(0.5*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - (self.tpag_frwrd(T_t)+self.ddtp_bkwrd(T_t)+self.He3ntp_bkwrd(T_t))*Yn0p1*Yn2p1 - self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 + self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 - self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2) + self.tpag_bkwrd(T_t)*Yn2p2 + self.taLi7g_bkwrd(T_t)*Yn4p3

    def dYHe3dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*(self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 + 0.5*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 + self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 - self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - (self.He3ntp_frwrd(T_t)+self.ddHe3n_bkwrd(T_t))*Yn1p0*Yn1p2 - self.He3aBe7g_frwrd(T_t)*Yn1p2*Yn2p2) + self.He3aBe7g_bkwrd(T_t)*Yn3p4 - self.dpHe3g_bkwrd(T_t)*Yn1p2

    def dYadtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*(self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 + self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 - self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 - self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 - self.He3aBe7g_frwrd(T_t)*Yn1p2*Yn2p2 - self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 + 2.*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3) + self.He3aBe7g_bkwrd(T_t)*Yn3p4 - self.tpag_bkwrd(T_t)*Yn2p2 + self.taLi7g_bkwrd(T_t)*Yn4p3 + rhoBBN*(-self.Be7naa_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(2*self.Be7naa_frwrd(T_t)*Yn1p0*Yn3p4) + rhoBBN*(- rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*Yn2p2*Yn0p1 + 2*self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4) + rhoBBN*(-self.daLi6g_frwrd(T_t)*Yn2p2*Yn1p1) + self.daLi6g_bkwrd(T_t)*Yn3p3 + (-rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(2*self.Li7paag_frwrd(T_t)*Yn0p1*Yn4p3)

    def dYLi7dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*(self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 + 0.5*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 - (self.Li7paa_frwrd(T_t)+self.Be7nLi7p_bkwrd(T_t))*Yn0p1*Yn4p3 + self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4) - self.taLi7g_bkwrd(T_t)*Yn4p3 + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn4p3*Yn0p1) + 0.5*rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*Yn2p2

    def dYBe7dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*(self.He3aBe7g_frwrd(T_t)*Yn1p2*Yn2p2 + self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4) - self.He3aBe7g_bkwrd(T_t)*Yn3p4 + rhoBBN*(-self.Be7naa_frwrd(T_t)*Yn3p4*Yn1p0) + rhoBBN*(0.5*self.Be7naa_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(-self.Be7daap_frwrd(T_t)*Yn3p4*Yn1p1 + 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2) + (-self.Li6pBe7g_bkwrd(T_t)*Yn3p4) + rhoBBN*(self.Li6pBe7g_frwrd(T_t)*Yn0p1*Yn3p3)

    def dYHe6dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return 0.

    def dYLi8dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return 0.

    def dYLi6dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return (-self.daLi6g_bkwrd(T_t)*Yn3p3) + rhoBBN*(self.daLi6g_frwrd(T_t)*Yn1p1*Yn2p2) + rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn3p3*Yn0p1) + (self.Li6pBe7g_bkwrd(T_t)*Yn3p4)

    def dYB8dtMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return 0.

    def JacobianMT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4}
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        # Yn
        dYn_primeOdYn = rhoBBN*(-self.npdg_frwrd(T_t)*Yn0p1 - (self.He3ntp_frwrd(T_t) + self.ddHe3n_bkwrd(T_t))*Yn1p2 - self.tdan_bkwrd(T_t)*Yn2p2 - self.Be7nLi7p_frwrd(T_t)*Yn3p4) + rhoBBN*(-self.Be7naa_frwrd(T_t)*Yn3p4)
        dYn_primeOdYp = nTOp_bkwrd(T_t) + rhoBBN*(- self.npdg_frwrd(T_t)*Yn1p0 + self.He3ntp_bkwrd(T_t)*Yn2p1 + self.Be7nLi7p_bkwrd(T_t)*Yn4p3)
        dYn_primeOdYd = rhoBBN*(self.ddHe3n_frwrd(T_t)*Yn1p1 + self.tdan_frwrd(T_t)*Yn2p1) + self.npdg_bkwrd(T_t)
        dYn_primeOdYt = rhoBBN*(self.He3ntp_bkwrd(T_t)*Yn0p1 + self.tdan_frwrd(T_t)*Yn1p1)
        dYn_primeOdYHe3 = -rhoBBN*(self.He3ntp_frwrd(T_t) + self.ddHe3n_bkwrd(T_t))*Yn1p0
        dYn_primeOdYa = -rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0 + rhoBBN*(0.5*self.Be7naa_bkwrd(T_t)*Yn2p2*2)
        dYn_primeOdYLi7 = rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1
        dYn_primeOdYBe7 = -rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0 + rhoBBN*(-self.Be7naa_frwrd(T_t)*Yn1p0)
        dYn_primeOdYHe6 = 0.
        dYn_primeOdYLi8 = 0.
        dYn_primeOdYLi6 = 0.
        dYn_primeOdYB8 = 0.
        dYn_row = [dYn_primeOdYn,dYn_primeOdYp,dYn_primeOdYd,dYn_primeOdYt,dYn_primeOdYHe3,dYn_primeOdYa,dYn_primeOdYLi7,dYn_primeOdYBe7,dYn_primeOdYHe6,dYn_primeOdYLi8,dYn_primeOdYLi6,dYn_primeOdYB8]

        # Yp
        dYp_primeOdYn = nTOp_frwrd(T_t) + rhoBBN*(- self.npdg_frwrd(T_t)*Yn0p1 + self.He3ntp_frwrd(T_t)*Yn1p2 + self.Be7nLi7p_frwrd(T_t)*Yn3p4)
        dYp_primeOdYp = -nTOp_bkwrd(T_t)*Yn0p1 + rhoBBN*(- self.npdg_frwrd(T_t)*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn1p1 - (self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn2p1 - self.He3dap_bkwrd(T_t)*Yn2p2 - (self.Li7paa_frwrd(T_t) + self.Be7nLi7p_bkwrd(T_t))*Yn4p3) + rhoBBN*(- 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn3p3) + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn4p3)
        dYp_primeOdYd = rhoBBN*(self.ddtp_frwrd(T_t)*Yn1p1 - self.dpHe3g_frwrd(T_t)*Yn0p1 + self.He3dap_frwrd(T_t)*Yn1p2) + self.npdg_bkwrd(T_t) + rhoBBN*(self.Be7daap_frwrd(T_t)*Yn3p4)
        dYp_primeOdYt = -rhoBBN*(self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn0p1
        dYp_primeOdYHe3 = rhoBBN*(self.He3ntp_frwrd(T_t)*Yn1p0 + self.He3dap_frwrd(T_t)*Yn1p1) + self.dpHe3g_bkwrd(T_t)
        dYp_primeOdYa = rhoBBN*(-self.He3dap_bkwrd(T_t)*Yn0p1 + self.Li7paa_bkwrd(T_t)*Yn2p2) + self.tpag_bkwrd(T_t) + rhoBBN*(- 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*2) + 0.5*rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*2
        dYp_primeOdYLi7 = rhoBBN*(-(self.Li7paa_frwrd(T_t) + self.Be7nLi7p_bkwrd(T_t))*Yn0p1) + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn0p1)
        dYp_primeOdYBe7 = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0 + rhoBBN*(self.Be7daap_frwrd(T_t)*Yn1p1) + (self.Li6pBe7g_bkwrd(T_t))
        dYp_primeOdYHe6 = 0.
        dYp_primeOdYLi8 = 0.
        dYp_primeOdYLi6 = rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn0p1)
        dYp_primeOdYB8 = 0.
        dYp_row = [dYp_primeOdYn,dYp_primeOdYp,dYp_primeOdYd,dYp_primeOdYt,dYp_primeOdYHe3,dYp_primeOdYa,dYp_primeOdYLi7,dYp_primeOdYBe7,dYp_primeOdYHe6,dYp_primeOdYLi8,dYp_primeOdYLi6,dYp_primeOdYB8]

        # Yd
        dYd_primeOdYn = rhoBBN*(self.npdg_frwrd(T_t)*Yn0p1 + 2.*self.ddHe3n_bkwrd(T_t)*Yn1p2 + self.tdan_bkwrd(T_t)*Yn2p2)
        dYd_primeOdYp = rhoBBN*(self.npdg_frwrd(T_t)*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn1p1 + 2.*self.ddtp_bkwrd(T_t)*Yn2p1 + self.He3dap_bkwrd(T_t)*Yn2p2) + rhoBBN*(0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*Yn2p2)
        dYd_primeOdYd = rhoBBN*(- self.dpHe3g_frwrd(T_t)*Yn0p1 - (self.ddtp_frwrd(T_t))*Yn1p1*2 - self.tdan_frwrd(T_t)*Yn2p1 - self.He3dap_frwrd(T_t)*Yn1p2) - self.npdg_bkwrd(T_t) + rhoBBN*(-self.Be7daap_frwrd(T_t)*Yn3p4) + rhoBBN*(-self.daLi6g_frwrd(T_t)*Yn2p2)
        dYd_primeOdYt = rhoBBN*(2.*self.ddtp_bkwrd(T_t)*Yn0p1 - self.tdan_frwrd(T_t)*Yn1p1)
        dYd_primeOdYHe3 = rhoBBN*(2.*self.ddHe3n_bkwrd(T_t)*Yn1p0 - self.He3dap_frwrd(T_t)*Yn1p1) + self.dpHe3g_bkwrd(T_t)
        dYd_primeOdYa = rhoBBN*(self.tdan_bkwrd(T_t)*Yn1p0 + self.He3dap_bkwrd(T_t)*Yn0p1) + rhoBBN*(0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*2) + rhoBBN*(-self.daLi6g_frwrd(T_t)*Yn1p1)
        dYd_primeOdYLi7 = 0.
        dYd_primeOdYBe7 = rhoBBN*(-self.Be7daap_frwrd(T_t)*Yn1p1)
        dYd_primeOdYHe6 = 0.
        dYd_primeOdYLi8 = 0.
        dYd_primeOdYLi6 = self.daLi6g_bkwrd(T_t)
        dYd_primeOdYB8 = 0.
        dYd_row = [dYd_primeOdYn,dYd_primeOdYp,dYd_primeOdYd,dYd_primeOdYt,dYd_primeOdYHe3,dYd_primeOdYa,dYd_primeOdYLi7,dYd_primeOdYBe7,dYd_primeOdYHe6,dYd_primeOdYLi8,dYd_primeOdYLi6,dYd_primeOdYB8]

        # Yt
        dYt_primeOdYn = rhoBBN*(self.He3ntp_frwrd(T_t)*Yn1p2 + self.tdan_bkwrd(T_t)*Yn2p2)
        dYt_primeOdYp = -rhoBBN*(self.tpag_frwrd(T_t)+self.ddtp_bkwrd(T_t)+self.He3ntp_bkwrd(T_t))*Yn2p1
        dYt_primeOdYd = rhoBBN*(self.ddtp_frwrd(T_t)*Yn1p1 - self.tdan_frwrd(T_t)*Yn2p1)
        dYt_primeOdYt = -rhoBBN*((self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn0p1 + self.tdan_frwrd(T_t)*Yn1p1 + self.taLi7g_frwrd(T_t)*Yn2p2)
        dYt_primeOdYHe3 = rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0
        dYt_primeOdYa = rhoBBN*(self.tdan_bkwrd(T_t)*Yn1p0 - self.taLi7g_frwrd(T_t)*Yn2p1) + self.tpag_bkwrd(T_t)
        dYt_primeOdYLi7 = self.taLi7g_bkwrd(T_t)
        dYt_primeOdYBe7 = 0.
        dYt_primeOdYHe6 = 0.
        dYt_primeOdYLi8 = 0.
        dYt_primeOdYLi6 = 0.
        dYt_primeOdYB8 = 0.
        dYt_row = [dYt_primeOdYn,dYt_primeOdYp,dYt_primeOdYd,dYt_primeOdYt,dYt_primeOdYHe3,dYt_primeOdYa,dYt_primeOdYLi7,dYt_primeOdYBe7,dYt_primeOdYHe6,dYt_primeOdYLi8,dYt_primeOdYLi6,dYt_primeOdYB8]

        # YHe3
        dYHe3_primeOdYn = -rhoBBN*(self.He3ntp_frwrd(T_t)+self.ddHe3n_bkwrd(T_t))*Yn1p2
        dYHe3_primeOdYp = rhoBBN*(self.dpHe3g_frwrd(T_t)*Yn1p1 + self.He3ntp_bkwrd(T_t)*Yn2p1 + self.He3dap_bkwrd(T_t)*Yn2p2)
        dYHe3_primeOdYd = rhoBBN*(self.dpHe3g_frwrd(T_t)*Yn0p1 + self.ddHe3n_frwrd(T_t)*Yn1p1 - self.He3dap_frwrd(T_t)*Yn1p2)
        dYHe3_primeOdYt = rhoBBN*(self.He3ntp_bkwrd(T_t)*Yn0p1)
        dYHe3_primeOdYHe3 = rhoBBN*(- self.He3dap_frwrd(T_t)*Yn1p1 - (self.He3ntp_frwrd(T_t)+self.ddHe3n_bkwrd(T_t))*Yn1p0 - self.He3aBe7g_frwrd(T_t)*Yn2p2) - self.dpHe3g_bkwrd(T_t)
        dYHe3_primeOdYa = rhoBBN*(self.He3dap_bkwrd(T_t)*Yn0p1 - self.He3aBe7g_frwrd(T_t)*Yn1p2)
        dYHe3_primeOdYLi7 = 0.
        dYHe3_primeOdYBe7 = self.He3aBe7g_bkwrd(T_t)
        dYHe3_primeOdYHe6 = 0.
        dYHe3_primeOdYLi8 = 0
        dYHe3_primeOdYLi6 = 0
        dYHe3_primeOdYB8 = 0
        dYHe3_row = [dYHe3_primeOdYn,dYHe3_primeOdYp,dYHe3_primeOdYd,dYHe3_primeOdYt,dYHe3_primeOdYHe3,dYHe3_primeOdYa,dYHe3_primeOdYLi7,dYHe3_primeOdYBe7,dYHe3_primeOdYHe6,dYHe3_primeOdYLi8,dYHe3_primeOdYLi6,dYHe3_primeOdYB8]

        # Ya
        dYa_primeOdYn = -rhoBBN*self.tdan_bkwrd(T_t)*Yn2p2 + rhoBBN*(2*self.Be7naa_frwrd(T_t)*Yn3p4)
        dYa_primeOdYp = rhoBBN*(- self.He3dap_bkwrd(T_t)*Yn2p2 + 2.*self.Li7paa_frwrd(T_t)*Yn4p3 + self.tpag_frwrd(T_t)*Yn2p1) + rhoBBN*(- rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(2*self.Li7paag_frwrd(T_t)*Yn4p3)
        dYa_primeOdYd = rhoBBN*(self.He3dap_frwrd(T_t)*Yn1p2 + self.tdan_frwrd(T_t)*Yn2p1) + rhoBBN*(2*self.Be7daap_frwrd(T_t)*Yn3p4) + rhoBBN*(-self.daLi6g_frwrd(T_t)*Yn2p2)
        dYa_primeOdYt = rhoBBN*(-self.taLi7g_frwrd(T_t)*Yn2p2 + self.tdan_frwrd(T_t)*Yn1p1 + self.tpag_frwrd(T_t)*Yn0p1)
        dYa_primeOdYHe3 = rhoBBN*(- self.He3aBe7g_frwrd(T_t)*Yn2p2 + self.He3dap_frwrd(T_t)*Yn1p1)
        dYa_primeOdYa = -rhoBBN*(self.He3aBe7g_frwrd(T_t)*Yn1p2 + self.He3dap_bkwrd(T_t)*Yn0p1 + 2.*self.Li7paa_bkwrd(T_t)*Yn2p2 + self.taLi7g_frwrd(T_t)*Yn2p1 + self.tdan_bkwrd(T_t)*Yn1p0) - self.tpag_bkwrd(T_t) + rhoBBN*(-self.Be7naa_bkwrd(T_t)*Yn2p2*2) + rhoBBN*(-rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*2*Yn0p1) + rhoBBN*(-self.daLi6g_frwrd(T_t)*Yn1p1) + (-rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*2)
        dYa_primeOdYLi7 = 2.*rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1+self.taLi7g_bkwrd(T_t) + rhoBBN*(2*self.Li7paag_frwrd(T_t)*Yn0p1)
        dYa_primeOdYBe7 = self.He3aBe7g_bkwrd(T_t) + rhoBBN*(2*self.Be7naa_frwrd(T_t)*Yn1p0) + rhoBBN*(2*self.Be7daap_frwrd(T_t)*Yn1p1)
        dYa_primeOdYHe6 = 0.
        dYa_primeOdYLi8 = 0.
        dYa_primeOdYLi6 = self.daLi6g_bkwrd(T_t)
        dYa_primeOdYB8 = 0.
        dYa_row = [dYa_primeOdYn,dYa_primeOdYp,dYa_primeOdYd,dYa_primeOdYt,dYa_primeOdYHe3,dYa_primeOdYa,dYa_primeOdYLi7,dYa_primeOdYBe7,dYa_primeOdYHe6,dYa_primeOdYLi8,dYa_primeOdYLi6,dYa_primeOdYB8]

        # YLi7
        dYLi7_primeOdYn = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn3p4
        dYLi7_primeOdYp = -rhoBBN*(self.Be7nLi7p_bkwrd(T_t) + self.Li7paa_frwrd(T_t))*Yn4p3 + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn4p3)
        dYLi7_primeOdYd = 0.
        dYLi7_primeOdYt = rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p2
        dYLi7_primeOdYHe3 = 0.
        dYLi7_primeOdYa = rhoBBN*(self.Li7paa_bkwrd(T_t)*Yn2p2 + self.taLi7g_frwrd(T_t)*Yn2p1) + 0.5*rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*2
        dYLi7_primeOdYLi7 = -rhoBBN*(self.Be7nLi7p_bkwrd(T_t) + self.Li7paa_frwrd(T_t))*Yn0p1 - self.taLi7g_bkwrd(T_t) + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn0p1)
        dYLi7_primeOdYBe7 = rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0
        dYLi7_primeOdYHe6 = 0.
        dYLi7_primeOdYLi8 = 0.
        dYLi7_primeOdYLi6 = 0.
        dYLi7_primeOdYB8 = 0.
        dYLi7_row = [dYLi7_primeOdYn,dYLi7_primeOdYp,dYLi7_primeOdYd,dYLi7_primeOdYt,dYLi7_primeOdYHe3,dYLi7_primeOdYa,dYLi7_primeOdYLi7,dYLi7_primeOdYBe7,dYLi7_primeOdYHe6,dYLi7_primeOdYLi8,dYLi7_primeOdYLi6,dYLi7_primeOdYB8]

        # YBe7
        dYBe7_primeOdYn = -rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn3p4 + rhoBBN*(-self.Be7naa_frwrd(T_t)*Yn3p4)
        dYBe7_primeOdYp = rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn4p3 + rhoBBN*(0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(self.Li6pBe7g_frwrd(T_t)*Yn3p3)
        dYBe7_primeOdYd = rhoBBN*(-self.Be7daap_frwrd(T_t)*Yn3p4)
        dYBe7_primeOdYt = 0.
        dYBe7_primeOdYHe3 = rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn2p2
        dYBe7_primeOdYa = rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2 + rhoBBN*(0.5*self.Be7naa_bkwrd(T_t)*Yn2p2*2 + 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*2)
        dYBe7_primeOdYLi7 = rhoBBN* self.Be7nLi7p_bkwrd(T_t)*Yn0p1
        dYBe7_primeOdYBe7 = -rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0 - self.He3aBe7g_bkwrd(T_t) + rhoBBN*(-self.Be7naa_frwrd(T_t)*Yn1p0) + rhoBBN*(-self.Be7daap_frwrd(T_t)*Yn1p1) + (-self.Li6pBe7g_bkwrd(T_t))
        dYBe7_primeOdYHe6 = 0.
        dYBe7_primeOdYLi8 = 0.
        dYBe7_primeOdYLi6 = rhoBBN*(self.Li6pBe7g_frwrd(T_t)*Yn0p1)
        dYBe7_primeOdYB8 = 0.
        dYBe7_row = [dYBe7_primeOdYn,dYBe7_primeOdYp,dYBe7_primeOdYd,dYBe7_primeOdYt,dYBe7_primeOdYHe3,dYBe7_primeOdYa,dYBe7_primeOdYLi7,dYBe7_primeOdYBe7,dYBe7_primeOdYHe6,dYBe7_primeOdYLi8,dYBe7_primeOdYLi6,dYBe7_primeOdYB8]

        # YHe6
        dYHe6_primeOdYn = 0.
        dYHe6_primeOdYp = 0.
        dYHe6_primeOdYd = 0.
        dYHe6_primeOdYt = 0.
        dYHe6_primeOdYHe3 = 0.
        dYHe6_primeOdYa = 0.
        dYHe6_primeOdYLi7 = 0.
        dYHe6_primeOdYBe7 = 0.
        dYHe6_primeOdYHe6 = 0.
        dYHe6_primeOdYLi8 = 0.
        dYHe6_primeOdYLi6 = 0.
        dYHe6_primeOdYB8 = 0.
        dYHe6_row = [dYHe6_primeOdYn,dYHe6_primeOdYp,dYHe6_primeOdYd,dYHe6_primeOdYt,dYHe6_primeOdYHe3,dYHe6_primeOdYa,dYHe6_primeOdYLi7,dYHe6_primeOdYBe7,dYHe6_primeOdYHe6,dYHe6_primeOdYLi8,dYHe6_primeOdYLi6,dYHe6_primeOdYB8]

        # YLi8
        dYLi8_primeOdYn = 0.
        dYLi8_primeOdYp = 0.
        dYLi8_primeOdYd = 0.
        dYLi8_primeOdYt = 0.
        dYLi8_primeOdYHe3 = 0.
        dYLi8_primeOdYa = 0.
        dYLi8_primeOdYLi7 = 0.
        dYLi8_primeOdYBe7 = 0.
        dYLi8_primeOdYHe6 = 0.
        dYLi8_primeOdYLi8 = 0.
        dYLi8_primeOdYLi6 = 0.
        dYLi8_primeOdYB8 = 0.
        dYLi8_row = [dYLi8_primeOdYn,dYLi8_primeOdYp,dYLi8_primeOdYd,dYLi8_primeOdYt,dYLi8_primeOdYHe3,dYLi8_primeOdYa,dYLi8_primeOdYLi7,dYLi8_primeOdYBe7,dYLi8_primeOdYHe6,dYLi8_primeOdYLi8,dYLi8_primeOdYLi6,dYLi8_primeOdYB8]

        # YLi6
        dYLi6_primeOdYn = 0.
        dYLi6_primeOdYp = rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn3p3)
        dYLi6_primeOdYd = rhoBBN*(self.daLi6g_frwrd(T_t)*Yn2p2)
        dYLi6_primeOdYt = 0.
        dYLi6_primeOdYHe3 = 0.
        dYLi6_primeOdYa = rhoBBN*(self.daLi6g_frwrd(T_t)*Yn1p1)
        dYLi6_primeOdYLi7 = 0.
        dYLi6_primeOdYBe7 = self.Li6pBe7g_bkwrd(T_t)
        dYLi6_primeOdYHe6 = 0.
        dYLi6_primeOdYLi8 = 0.
        dYLi6_primeOdYLi6 = (-self.daLi6g_bkwrd(T_t)) + rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn0p1)
        dYLi6_primeOdYB8 = 0.
        dYLi6_row = [dYLi6_primeOdYn,dYLi6_primeOdYp,dYLi6_primeOdYd,dYLi6_primeOdYt,dYLi6_primeOdYHe3,dYLi6_primeOdYa,dYLi6_primeOdYLi7,dYLi6_primeOdYBe7,dYLi6_primeOdYHe6,dYLi6_primeOdYLi8,dYLi6_primeOdYLi6,dYLi6_primeOdYB8]

        # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4, YHe6 -> Yn4p2, Li8 -> Yn5p3, Li6 -> Yn3p3, B8 -> Yn3p5}
        # YB8
        dYB8_primeOdYn = 0.
        dYB8_primeOdYp = 0.
        dYB8_primeOdYd = 0.
        dYB8_primeOdYt = 0.
        dYB8_primeOdYHe3 = 0.
        dYB8_primeOdYa = 0.
        dYB8_primeOdYLi7 = 0.
        dYB8_primeOdYBe7 = 0.
        dYB8_primeOdYHe6 = 0.
        dYB8_primeOdYLi8 = 0.
        dYB8_primeOdYLi6 = 0.
        dYB8_primeOdYB8 = 0.
        dYB8_row = [dYB8_primeOdYn,dYB8_primeOdYp,dYB8_primeOdYd,dYB8_primeOdYt,dYB8_primeOdYHe3,dYB8_primeOdYa,dYB8_primeOdYLi7,dYB8_primeOdYBe7,dYB8_primeOdYHe6,dYB8_primeOdYLi8,dYB8_primeOdYLi6,dYB8_primeOdYB8]

        return [dYn_row,dYp_row,dYd_row,dYt_row,dYHe3_row,dYa_row,dYLi7_row,dYBe7_row, dYHe6_row, dYLi8_row, dYLi6_row, dYB8_row]

    def dYndtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YHe6 -> Yn4p2, YLi6 -> Yn3p3, YLi7 -> Yn4p3, YLi8 -> Yn5p3, YBe7 -> Yn3p4, YB8 -> Yn3p5}
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return -nTOp_frwrd(T_t)*Yn1p0 + nTOp_bkwrd(T_t)*Yn0p1 - rhoBBN*self.npdg_frwrd(T_t)*Yn1p0*Yn0p1 - 0.5*rhoBBN*rhoBBN* self.ppndp_frwrd(T_t)*Yn1p0*Yn0p1*Yn0p1 + self.npdg_bkwrd(T_t)*Yn1p1 - rhoBBN*self.dntg_frwrd(T_t)*Yn1p0*Yn1p1 + rhoBBN*self.ppndp_bkwrd(T_t)*Yn0p1*Yn1p1 + 0.5*rhoBBN*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 + self.dntg_bkwrd(T_t)*Yn2p1 + rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 + rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + rhoBBN*self.ttann_frwrd(T_t)*Yn2p1*Yn2p1 - rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3nag_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 + rhoBBN*self.He3tanp_frwrd(T_t)*Yn2p1*Yn1p2 + self.He3nag_bkwrd(T_t)*Yn2p2 - rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 - rhoBBN*rhoBBN*self.annHe6g_frwrd(T_t)*Yn1p0*Yn1p0*Yn2p2 - rhoBBN*rhoBBN*self.ttann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2 - rhoBBN*rhoBBN*self.anpLi6g_frwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*self.He3tanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*self.Li6nta_bkwrd(T_t)*Yn2p1*Yn2p2 + 0.5*rhoBBN*self.Be7naa_bkwrd(T_t)*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li6taan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li7daan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li8paan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Li7taann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2* Yn2p2 + 0.5*rhoBBN*rhoBBN*self.B8naap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Li7He3aanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7taanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 2*self.annHe6g_bkwrd(T_t)*Yn4p2 + self.anpLi6g_bkwrd(T_t)*Yn3p3 - rhoBBN*self.Li6nta_frwrd(T_t)*Yn1p0*Yn3p3 - rhoBBN*self.Li6nLi7g_frwrd(T_t)*Yn1p0*Yn3p3 + rhoBBN*self.Li6dBe7n_frwrd(T_t)*Yn1p1*Yn3p3 + rhoBBN*self.Li6taan_frwrd(T_t)*Yn2p1*Yn3p3 + rhoBBN*self.B8nLi6He3_bkwrd(T_t)*Yn1p2*Yn3p3 + self.Li6nLi7g_bkwrd(T_t)*Yn4p3 - rhoBBN*self.Li7nLi8g_frwrd(T_t)*Yn1p0*Yn4p3 + rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 + rhoBBN*self.Li7daan_frwrd(T_t)*Yn1p1*Yn4p3 + 2*rhoBBN*self.Li7taann_frwrd(T_t)*Yn2p1*Yn4p3 + rhoBBN*self.Li7He3aanp_frwrd(T_t)*Yn1p2*Yn4p3 + self.Li7nLi8g_bkwrd(T_t)*Yn5p3 + rhoBBN*self.Li8paan_frwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Be7naa_frwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Li6dBe7n_bkwrd(T_t)*Yn1p0*Yn3p4 + rhoBBN*self.B8nBe7d_bkwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.Be7taanp_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.B8naap_frwrd(T_t)*Yn1p0*Yn3p5 - rhoBBN*self.B8nLi6He3_frwrd(T_t)*Yn1p0*Yn3p5 - rhoBBN*self.B8nBe7d_frwrd(T_t)*Yn1p0*Yn3p5
        
    def dYpdtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return nTOp_frwrd(T_t)*Yn1p0 - nTOp_bkwrd(T_t)*Yn0p1 - rhoBBN*self.npdg_frwrd(T_t)*Yn1p0*Yn0p1 - 0.5*rhoBBN*rhoBBN*self.ppndp_frwrd(T_t)*Yn1p0*Yn0p1*Yn0p1 + self.npdg_bkwrd(T_t)*Yn1p1 - rhoBBN*self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 + rhoBBN*self.ppndp_bkwrd(T_t)*Yn0p1*Yn1p1 + 0.5*rhoBBN*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - rhoBBN*self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 + self.dpHe3g_bkwrd(T_t)*Yn1p2 + rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 + rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + rhoBBN*self.He3tanp_frwrd(T_t)*Yn2p1*Yn1p2 + rhoBBN*self.He3He3app_frwrd(T_t)*Yn1p2*Yn1p2 + self.tpag_bkwrd(T_t)*Yn2p2 - rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*self.anpLi6g_frwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*self.He3tanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*self.He3He3app_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2 + rhoBBN*self.Li6pHe3a_bkwrd(T_t)*Yn1p2*Yn2p2 + 0.5*rhoBBN*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 + 0.5*rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li8paan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.B8naap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li6He3aap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Li7He3aanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7taanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7He3ppaa_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7He3aapp_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + self.anpLi6g_bkwrd(T_t)*Yn3p3 - rhoBBN*self.Li6pBe7g_frwrd(T_t)*Yn0p1*Yn3p3 - rhoBBN*self.Li6pHe3a_frwrd(T_t)*Yn0p1*Yn3p3 + rhoBBN*self.Li6dLi7p_frwrd(T_t)*Yn1p1*Yn3p3 + rhoBBN*self.Li6tLi8p_frwrd(T_t)*Yn2p1*Yn3p3 + rhoBBN*self.Li6He3aap_frwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Li7paag_frwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Li6dLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 + rhoBBN*self.Li7dLi8p_frwrd(T_t)*Yn1p1*Yn4p3 + rhoBBN*self.Li7He3aanp_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li8paan_frwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Li6tLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Li7dLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 + self.Li6pBe7g_bkwrd(T_t)*Yn3p4 + rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Be7pB8g_frwrd(T_t)*Yn0p1*Yn3p4 + rhoBBN*self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.Be7taanp_frwrd(T_t)*Yn2p1*Yn3p4 + 2.*rhoBBN*self.Be7He3ppaa_frwrd(T_t)*Yn1p2*Yn3p4 + 2.*rhoBBN*self.Be7He3aapp_frwrd(T_t)*Yn1p2*Yn3p4 + self.Be7pB8g_bkwrd(T_t)*Yn3p5 + rhoBBN*self.B8naap_frwrd(T_t)*Yn1p0*Yn3p5
        
    def dYddtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*self.npdg_frwrd(T_t)*Yn1p0*Yn0p1 + 0.5*rhoBBN*rhoBBN*self.ppndp_frwrd(T_t)*Yn1p0*Yn0p1*Yn0p1 - self.npdg_bkwrd(T_t)*Yn1p1 - rhoBBN*self.dntg_frwrd(T_t)*Yn1p0*Yn1p1 - rhoBBN*self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 - rhoBBN*self.ppndp_bkwrd(T_t)*Yn0p1*Yn1p1 - rhoBBN*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 - rhoBBN*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - rhoBBN*self.ddag_frwrd(T_t)*Yn1p1*Yn1p1 + self.dntg_bkwrd(T_t)*Yn2p1 + 2*rhoBBN*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + self.dpHe3g_bkwrd(T_t)*Yn1p2 + 2*rhoBBN*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + rhoBBN*self.He3tad_frwrd(T_t)*Yn2p1*Yn1p2 + 2*self.ddag_bkwrd(T_t)*Yn2p2 + rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 + rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - rhoBBN*self.daLi6g_frwrd(T_t)*Yn1p1*Yn2p2 - rhoBBN*self.He3tad_bkwrd(T_t)*Yn1p1*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li7daan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li7He3aad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Be7taad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 + self.daLi6g_bkwrd(T_t)*Yn3p3 - rhoBBN*self.Li6dLi7p_frwrd(T_t)*Yn1p1*Yn3p3 - rhoBBN*self.Li6dBe7n_frwrd(T_t)*Yn1p1*Yn3p3 + rhoBBN*self.Li6tLi7d_frwrd(T_t)*Yn2p1*Yn3p3 + rhoBBN*self.Li6He3Be7d_frwrd(T_t)*Yn1p2*Yn3p3 + rhoBBN*self.Li6dLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Li7daan_frwrd(T_t)*Yn1p1*Yn4p3 - rhoBBN*self.Li7dLi8p_frwrd(T_t)*Yn1p1*Yn4p3 - rhoBBN*self.Li6tLi7d_bkwrd(T_t)*Yn1p1*Yn4p3 + rhoBBN*self.Li8dLi7t_bkwrd(T_t)*Yn2p1*Yn4p3 + rhoBBN*self.Li7He3aad_frwrd(T_t)*Yn1p2*Yn4p3 + rhoBBN*self.Li7dLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Li8dLi7t_frwrd(T_t)*Yn1p1*Yn5p3 + rhoBBN*self.Li6dBe7n_bkwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4 - rhoBBN*self.B8nBe7d_bkwrd(T_t)*Yn1p1*Yn3p4 - rhoBBN*self.Li6He3Be7d_bkwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.Be7taad_frwrd(T_t)*Yn2p1*Yn3p4 + rhoBBN*self.B8dBe7He3_bkwrd(T_t)*Yn1p2*Yn3p4 + rhoBBN*self.B8nBe7d_frwrd(T_t)*Yn1p0*Yn3p5 - rhoBBN*self.B8dBe7He3_frwrd(T_t)*Yn1p1*Yn3p5
        
    def dYtdtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*self.dntg_frwrd(T_t)*Yn1p0*Yn1p1 + 0.5*rhoBBN*self.ddtp_frwrd(T_t)*Yn1p1*Yn1p1 - self.dntg_bkwrd(T_t)*Yn2p1 - rhoBBN*self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.ddtp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 - rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 - rhoBBN*self.ttann_frwrd(T_t)*Yn2p1*Yn2p1 + rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3tad_frwrd(T_t)*Yn2p1*Yn1p2 - rhoBBN*self.He3tanp_frwrd(T_t)*Yn2p1*Yn1p2 - rhoBBN*self.He3tLi6g_frwrd(T_t)*Yn2p1*Yn1p2 + self.tpag_bkwrd(T_t)*Yn2p2 + rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 + rhoBBN*rhoBBN*self.ttann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2 + rhoBBN*rhoBBN*self.He3tanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*self.He3tad_bkwrd(T_t)*Yn1p1*Yn2p2 - rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 - rhoBBN*self.Li6nta_bkwrd(T_t)*Yn2p1*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li6taan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*self.Li7taann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7taanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Be7taad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li8He3aat_bkwrd(T_t)*Yn2p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.B8taaHe3_bkwrd(T_t)*Yn1p2*Yn2p2*Yn2p2 + self.He3tLi6g_bkwrd(T_t)*Yn3p3 + rhoBBN*self.Li6nta_frwrd(T_t)*Yn1p0*Yn3p3 - rhoBBN*self.Li6taan_frwrd(T_t)*Yn2p1*Yn3p3 - rhoBBN*self.Li6tLi8p_frwrd(T_t)*Yn2p1*Yn3p3 - rhoBBN*self.Li6tLi7d_frwrd(T_t)*Yn2p1*Yn3p3 + rhoBBN*self.Be7tLi6a_bkwrd(T_t)*Yn2p2*Yn3p3 + self.taLi7g_bkwrd(T_t)*Yn4p3 + rhoBBN*self.Li6tLi7d_bkwrd(T_t)*Yn1p1*Yn4p3 - rhoBBN*self.Li7taann_frwrd(T_t)*Yn2p1*Yn4p3 - rhoBBN*self.Li8dLi7t_bkwrd(T_t)*Yn2p1*Yn4p3 + rhoBBN*self.Be7tLi7He3_bkwrd(T_t)*Yn1p2*Yn4p3 + rhoBBN*self.Li6tLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 + rhoBBN*self.Li8dLi7t_frwrd(T_t)*Yn1p1*Yn5p3 + rhoBBN*self.Li8He3aat_frwrd(T_t)*Yn1p2*Yn5p3 - rhoBBN*self.Be7tLi6a_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7taad_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7tLi7He3_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7taanp_frwrd(T_t)*Yn2p1*Yn3p4 + rhoBBN*self.B8tBe7a_bkwrd(T_t)*Yn2p2*Yn3p4 - rhoBBN*self.B8tBe7a_frwrd(T_t)*Yn2p1*Yn3p5 - rhoBBN*self.B8taaHe3_frwrd(T_t)*Yn2p1*Yn3p5
        
    def dYHe3dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*self.dpHe3g_frwrd(T_t)*Yn0p1*Yn1p1 + 0.5*rhoBBN*self.ddHe3n_frwrd(T_t)*Yn1p1*Yn1p1 + rhoBBN*self.He3ntp_bkwrd(T_t)*Yn0p1*Yn2p1 - self.dpHe3g_bkwrd(T_t)*Yn1p2 - rhoBBN*self.He3ntp_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3nag_frwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.ddHe3n_bkwrd(T_t)*Yn1p0*Yn1p2 - rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 - rhoBBN*self.He3tad_frwrd(T_t)*Yn2p1*Yn1p2 - rhoBBN*self.He3tanp_frwrd(T_t)*Yn2p1*Yn1p2 - rhoBBN*self.He3tLi6g_frwrd(T_t)*Yn2p1*Yn1p2 - rhoBBN*self.He3He3app_frwrd(T_t)*Yn1p2*Yn1p2 + self.He3nag_bkwrd(T_t)*Yn2p2 + rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 + rhoBBN*rhoBBN*self.He3tanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*rhoBBN*self.He3He3app_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2 + rhoBBN*self.He3tad_bkwrd(T_t)*Yn1p1*Yn2p2 - rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2*Yn2p2 - rhoBBN*self.Li6pHe3a_bkwrd(T_t)*Yn1p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li6He3aap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*self.Li7He3aanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*self.Be7He3ppaa_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*self.Be7He3aapp_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li7He3aad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li8He3aat_bkwrd(T_t)*Yn2p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.B8taaHe3_bkwrd(T_t)*Yn1p2*Yn2p2*Yn2p2 + self.He3tLi6g_bkwrd(T_t)*Yn3p3 + rhoBBN*self.Li6pHe3a_frwrd(T_t)*Yn0p1*Yn3p3 - rhoBBN*self.Li6He3aap_frwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.Li6He3Be7d_frwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.B8nLi6He3_bkwrd(T_t)*Yn1p2*Yn3p3 + rhoBBN*self.Li7He3Li6a_bkwrd(T_t)*Yn2p2*Yn3p3 - rhoBBN*self.Li7He3Li6a_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li7He3aad_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li7He3aanp_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Be7tLi7He3_bkwrd(T_t)*Yn1p2*Yn4p3 + rhoBBN*self.Li8He3Li7a_bkwrd(T_t)*Yn2p2*Yn4p3 - rhoBBN*self.Li8He3Li7a_frwrd(T_t)*Yn1p2*Yn5p3 - rhoBBN*self.Li8He3aat_frwrd(T_t)*Yn1p2*Yn5p3 + self.He3aBe7g_bkwrd(T_t)*Yn3p4 + rhoBBN*self.Li6He3Be7d_bkwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.Be7tLi7He3_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7He3ppaa_frwrd(T_t)*Yn1p2*Yn3p4 - rhoBBN*self.Be7He3aapp_frwrd(T_t)*Yn1p2*Yn3p4 - rhoBBN*self.B8dBe7He3_bkwrd(T_t)*Yn1p2*Yn3p4 + rhoBBN*self.B8nLi6He3_frwrd(T_t)*Yn1p0*Yn3p5 + rhoBBN*self.B8dBe7He3_frwrd(T_t)*Yn1p1*Yn3p5 + rhoBBN*self.B8taaHe3_frwrd(T_t)*Yn2p1*Yn3p5
        
    def dYadtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return 0.5*rhoBBN*self.ddag_frwrd(T_t)*Yn1p1*Yn1p1 + rhoBBN*self.tpag_frwrd(T_t)*Yn0p1*Yn2p1 + rhoBBN*self.tdan_frwrd(T_t)*Yn1p1*Yn2p1 + 0.5*rhoBBN*self.ttann_frwrd(T_t)*Yn2p1*Yn2p1 + rhoBBN*self.He3nag_frwrd(T_t)*Yn1p0*Yn1p2 + rhoBBN*self.He3dap_frwrd(T_t)*Yn1p1*Yn1p2 + rhoBBN*self.He3tad_frwrd(T_t)*Yn2p1*Yn1p2 + rhoBBN*self.He3tanp_frwrd(T_t)*Yn2p1*Yn1p2 + 0.5*rhoBBN*self.He3He3app_frwrd(T_t)*Yn1p2*Yn1p2 - self.tpag_bkwrd(T_t)*Yn2p2 - self.ddag_bkwrd(T_t)*Yn2p2 - self.He3nag_bkwrd(T_t)*Yn2p2 - rhoBBN*self.tdan_bkwrd(T_t)*Yn1p0*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.annHe6g_frwrd(T_t)*Yn1p0*Yn1p0*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.ttann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2 - rhoBBN*self.He3dap_bkwrd(T_t)*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*self.anpLi6g_frwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*self.He3tanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.He3He3app_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2 - rhoBBN*self.daLi6g_frwrd(T_t)*Yn1p1*Yn2p2 - rhoBBN*self.He3tad_bkwrd(T_t)*Yn1p1*Yn2p2 - rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 - rhoBBN*self.Li6nta_bkwrd(T_t)*Yn2p1*Yn2p2 - rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2*Yn2p2 - rhoBBN*self.Li6pHe3a_bkwrd(T_t)*Yn1p2*Yn2p2 - rhoBBN*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 - rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*Yn2p2 - rhoBBN*self.Be7naa_bkwrd(T_t)*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Li6taan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Li7daan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Li8paan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Li7taann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.B8naap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Li6He3aap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*rhoBBN*self.Li7He3aanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*rhoBBN*self.Be7taanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7He3ppaa_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7He3aapp_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Li7He3aad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Be7taad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Li8He3aat_bkwrd(T_t)*Yn2p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.B8taaHe3_bkwrd(T_t)*Yn1p2*Yn2p2*Yn2p2 + self.annHe6g_bkwrd(T_t)*Yn4p2 + self.daLi6g_bkwrd(T_t)*Yn3p3 + self.anpLi6g_bkwrd(T_t)*Yn3p3 + rhoBBN*self.Li6nta_frwrd(T_t)*Yn1p0*Yn3p3 + rhoBBN*self.Li6pHe3a_frwrd(T_t)*Yn0p1*Yn3p3 + 2*rhoBBN*self.Li6taan_frwrd(T_t)*Yn2p1*Yn3p3 + 2*rhoBBN*self.Li6He3aap_frwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.Li7He3Li6a_bkwrd(T_t)*Yn2p2*Yn3p3 - rhoBBN*self.Be7tLi6a_bkwrd(T_t)*Yn2p2*Yn3p3 + self.taLi7g_bkwrd(T_t)*Yn4p3 + 2*rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3 + 2*rhoBBN*self.Li7paag_frwrd(T_t)*Yn0p1*Yn4p3 + 2*rhoBBN*self.Li7daan_frwrd(T_t)*Yn1p1*Yn4p3 + 2*rhoBBN*self.Li7taann_frwrd(T_t)*Yn2p1*Yn4p3 + rhoBBN*self.Li7He3Li6a_frwrd(T_t)*Yn1p2*Yn4p3 + 2*rhoBBN*self.Li7He3aad_frwrd(T_t)*Yn1p2*Yn4p3 + 2*rhoBBN*self.Li7He3aanp_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li8He3Li7a_bkwrd(T_t)*Yn2p2*Yn4p3 + 2*rhoBBN*self.Li8paan_frwrd(T_t)*Yn0p1*Yn5p3 + rhoBBN*self.Li8He3Li7a_frwrd(T_t)*Yn1p2*Yn5p3 + 2*rhoBBN*self.Li8He3aat_frwrd(T_t)*Yn1p2*Yn5p3 + self.He3aBe7g_bkwrd(T_t)*Yn3p4 + 2*rhoBBN*self.Be7naa_frwrd(T_t)*Yn1p0*Yn3p4 + 2*rhoBBN*self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.Be7tLi6a_frwrd(T_t)*Yn2p1*Yn3p4 + 2*rhoBBN*self.Be7taad_frwrd(T_t)*Yn2p1*Yn3p4 + 2*rhoBBN*self.Be7taanp_frwrd(T_t)*Yn2p1*Yn3p4 + 2*rhoBBN*self.Be7He3ppaa_frwrd(T_t)*Yn1p2*Yn3p4 + 2*rhoBBN*self.Be7He3aapp_frwrd(T_t)*Yn1p2*Yn3p4 - rhoBBN*self.B8tBe7a_bkwrd(T_t)*Yn2p2*Yn3p4 + 2*rhoBBN*self.B8naap_frwrd(T_t)*Yn1p0*Yn3p5 + rhoBBN*self.B8tBe7a_frwrd(T_t)*Yn2p1*Yn3p5 + 2*rhoBBN*self.B8taaHe3_frwrd(T_t)*Yn2p1*Yn3p5

    def dYHe6dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return 0.5*rhoBBN*rhoBBN*self.annHe6g_frwrd(T_t)*Yn1p0*Yn1p0*Yn2p2 - self.annHe6g_bkwrd(T_t)*Yn4p2
        
    def dYLi6dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*self.He3tLi6g_frwrd(T_t)*Yn2p1*Yn1p2 + rhoBBN*rhoBBN*self.anpLi6g_frwrd(T_t)*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*self.daLi6g_frwrd(T_t)*Yn1p1*Yn2p2 + rhoBBN*self.Li6nta_bkwrd(T_t)*Yn2p1*Yn2p2 + rhoBBN*self.Li6pHe3a_bkwrd(T_t)*Yn1p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li6taan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li6He3aap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - self.daLi6g_bkwrd(T_t)*Yn3p3 - self.He3tLi6g_bkwrd(T_t)*Yn3p3 - self.anpLi6g_bkwrd(T_t)*Yn3p3 - rhoBBN*self.Li6nta_frwrd(T_t)*Yn1p0*Yn3p3 - rhoBBN*self.Li6nLi7g_frwrd(T_t)*Yn1p0*Yn3p3 - rhoBBN*self.Li6pBe7g_frwrd(T_t)*Yn0p1*Yn3p3 - rhoBBN*self.Li6pHe3a_frwrd(T_t)*Yn0p1*Yn3p3 - rhoBBN*self.Li6dLi7p_frwrd(T_t)*Yn1p1*Yn3p3 - rhoBBN*self.Li6dBe7n_frwrd(T_t)*Yn1p1*Yn3p3 - rhoBBN*self.Li6taan_frwrd(T_t)*Yn2p1*Yn3p3 - rhoBBN*self.Li6tLi8p_frwrd(T_t)*Yn2p1*Yn3p3 - rhoBBN*self.Li6tLi7d_frwrd(T_t)*Yn2p1*Yn3p3 - rhoBBN*self.Li6He3aap_frwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.Li6He3Be7d_frwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.B8nLi6He3_bkwrd(T_t)*Yn1p2*Yn3p3 - rhoBBN*self.Li7He3Li6a_bkwrd(T_t)*Yn2p2*Yn3p3 - rhoBBN*self.Be7tLi6a_bkwrd(T_t)*Yn2p2*Yn3p3 + self.Li6nLi7g_bkwrd(T_t)*Yn4p3 + rhoBBN*self.Li6dLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 + rhoBBN*self.Li6tLi7d_bkwrd(T_t)*Yn1p1*Yn4p3 + rhoBBN*self.Li7He3Li6a_frwrd(T_t)*Yn1p2*Yn4p3 + rhoBBN*self.Li6tLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 + self.Li6pBe7g_bkwrd(T_t)*Yn3p4 + rhoBBN*self.Li6dBe7n_bkwrd(T_t)*Yn1p0*Yn3p4 + rhoBBN*self.Li6He3Be7d_bkwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.Be7tLi6a_frwrd(T_t)*Yn2p1*Yn3p4 + rhoBBN*self.B8nLi6He3_frwrd(T_t)*Yn1p0*Yn3p5
        
    def dYLi7dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*self.taLi7g_frwrd(T_t)*Yn2p1*Yn2p2 + 0.5*rhoBBN*self.Li7paa_bkwrd(T_t)*Yn2p2*Yn2p2 + 0.5*rhoBBN*self.Li7paag_bkwrd(T_t)*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li7daan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*self.Li7taann_bkwrd(T_t)*Yn1p0*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*self.Li7He3aanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li7He3aad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 + rhoBBN*self.Li6nLi7g_frwrd(T_t)*Yn1p0*Yn3p3 + rhoBBN*self.Li6dLi7p_frwrd(T_t)*Yn1p1*Yn3p3 + rhoBBN*self.Li6tLi7d_frwrd(T_t)*Yn2p1*Yn3p3 + rhoBBN*self.Li7He3Li6a_bkwrd(T_t)*Yn2p2*Yn3p3 - self.taLi7g_bkwrd(T_t)*Yn4p3 - self.Li6nLi7g_bkwrd(T_t)*Yn4p3 - rhoBBN*self.Li7nLi8g_frwrd(T_t)*Yn1p0*Yn4p3 - rhoBBN*self.Li7paa_frwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Li7paag_frwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Li6dLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 - rhoBBN*self.Li7daan_frwrd(T_t)*Yn1p1*Yn4p3 - rhoBBN*self.Li7dLi8p_frwrd(T_t)*Yn1p1*Yn4p3 - rhoBBN*self.Li6tLi7d_bkwrd(T_t)*Yn1p1*Yn4p3 - rhoBBN*self.Li7taann_frwrd(T_t)*Yn2p1*Yn4p3 - rhoBBN*self.Li8dLi7t_bkwrd(T_t)*Yn2p1*Yn4p3 - rhoBBN*self.Li7He3Li6a_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li7He3aad_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li7He3aanp_frwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Be7tLi7He3_bkwrd(T_t)*Yn1p2*Yn4p3 - rhoBBN*self.Li8He3Li7a_bkwrd(T_t)*Yn2p2*Yn4p3 + self.Li7nLi8g_bkwrd(T_t)*Yn5p3 + rhoBBN*self.Li7dLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 + rhoBBN*self.Li8dLi7t_frwrd(T_t)*Yn1p1*Yn5p3 + rhoBBN*self.Li8He3Li7a_frwrd(T_t)*Yn1p2*Yn5p3 + rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4 + rhoBBN*self.Be7tLi7He3_frwrd(T_t)*Yn2p1*Yn3p4
        
    def dYLi8dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return 0.5*rhoBBN*rhoBBN*self.Li8paan_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Li8He3aat_bkwrd(T_t)*Yn2p1*Yn2p2*Yn2p2 + rhoBBN*self.Li6tLi8p_frwrd(T_t)*Yn2p1*Yn3p3 + rhoBBN*self.Li7nLi8g_frwrd(T_t)*Yn1p0*Yn4p3 + rhoBBN*self.Li7dLi8p_frwrd(T_t)*Yn1p1*Yn4p3 + rhoBBN*self.Li8dLi7t_bkwrd(T_t)*Yn2p1*Yn4p3 + rhoBBN*self.Li8He3Li7a_bkwrd(T_t)*Yn2p2*Yn4p3 - self.Li7nLi8g_bkwrd(T_t)*Yn5p3 - rhoBBN*self.Li8paan_frwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Li6tLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Li7dLi8p_bkwrd(T_t)*Yn0p1*Yn5p3 - rhoBBN*self.Li8dLi7t_frwrd(T_t)*Yn1p1*Yn5p3 - rhoBBN*self.Li8He3Li7a_frwrd(T_t)*Yn1p2*Yn5p3 - rhoBBN*self.Li8He3aat_frwrd(T_t)*Yn1p2*Yn5p3
        
    def dYBe7dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        # Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        return rhoBBN*self.He3aBe7g_frwrd(T_t)*Yn1p2*Yn2p2 + 0.5*rhoBBN*self.Be7naa_bkwrd(T_t)*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*self.Be7taanp_bkwrd(T_t)*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*self.Be7He3ppaa_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*self.Be7He3aapp_bkwrd(T_t)*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.Be7taad_bkwrd(T_t)*Yn1p1*Yn2p2*Yn2p2 + rhoBBN*self.Li6pBe7g_frwrd(T_t)*Yn0p1*Yn3p3 + rhoBBN*self.Li6dBe7n_frwrd(T_t)*Yn1p1*Yn3p3 + rhoBBN*self.Li6He3Be7d_frwrd(T_t)*Yn1p2*Yn3p3 + rhoBBN*self.Be7tLi6a_bkwrd(T_t)*Yn2p2*Yn3p3 + rhoBBN*self.Be7nLi7p_bkwrd(T_t)*Yn0p1*Yn4p3 + rhoBBN*self.Be7tLi7He3_bkwrd(T_t)*Yn1p2*Yn4p3 - self.He3aBe7g_bkwrd(T_t)*Yn3p4 - self.Li6pBe7g_bkwrd(T_t)*Yn3p4 - rhoBBN*self.Be7nLi7p_frwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Be7naa_frwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Li6dBe7n_bkwrd(T_t)*Yn1p0*Yn3p4 - rhoBBN*self.Be7pB8g_frwrd(T_t)*Yn0p1*Yn3p4 - rhoBBN*self.Be7daap_frwrd(T_t)*Yn1p1*Yn3p4 - rhoBBN*self.B8nBe7d_bkwrd(T_t)*Yn1p1*Yn3p4 - rhoBBN*self.Li6He3Be7d_bkwrd(T_t)*Yn1p1*Yn3p4 - rhoBBN*self.Be7tLi6a_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7taad_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7tLi7He3_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7taanp_frwrd(T_t)*Yn2p1*Yn3p4 - rhoBBN*self.Be7He3ppaa_frwrd(T_t)*Yn1p2*Yn3p4 - rhoBBN*self.Be7He3aapp_frwrd(T_t)*Yn1p2*Yn3p4 - rhoBBN*self.B8dBe7He3_bkwrd(T_t)*Yn1p2*Yn3p4 - rhoBBN*self.B8tBe7a_bkwrd(T_t)*Yn2p2*Yn3p4 + self.Be7pB8g_bkwrd(T_t)*Yn3p5 + rhoBBN*self.B8nBe7d_frwrd(T_t)*Yn1p0*Yn3p5 + rhoBBN*self.B8dBe7He3_frwrd(T_t)*Yn1p1*Yn3p5 + rhoBBN*self.B8tBe7a_frwrd(T_t)*Yn2p1*Yn3p5

    def dYB8dtLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p2, Yn3p3, Yn4p3, Yn5p3, Yn3p4, Yn3p5 = Y
        return 0.5*rhoBBN*rhoBBN*self.B8naap_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*self.B8taaHe3_bkwrd(T_t)*Yn1p2*Yn2p2*Yn2p2 + rhoBBN*self.B8nLi6He3_bkwrd(T_t)*Yn1p2*Yn3p3 + rhoBBN*self.Be7pB8g_frwrd(T_t)*Yn0p1*Yn3p4 + rhoBBN*self.B8nBe7d_bkwrd(T_t)*Yn1p1*Yn3p4 + rhoBBN*self.B8dBe7He3_bkwrd(T_t)*Yn1p2*Yn3p4 + rhoBBN*self.B8tBe7a_bkwrd(T_t)*Yn2p2*Yn3p4 - self.Be7pB8g_bkwrd(T_t)*Yn3p5 - rhoBBN*self.B8naap_frwrd(T_t)*Yn1p0*Yn3p5 - rhoBBN*self.B8nLi6He3_frwrd(T_t)*Yn1p0*Yn3p5 - rhoBBN*self.B8nBe7d_frwrd(T_t)*Yn1p0*Yn3p5 - rhoBBN*self.B8dBe7He3_frwrd(T_t)*Yn1p1*Yn3p5 - rhoBBN*self.B8tBe7a_frwrd(T_t)*Yn2p1*Yn3p5 - rhoBBN*self.B8taaHe3_frwrd(T_t)*Yn2p1*Yn3p5
        
    def JacobianLT(self,Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd):
    # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4}
        Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y
        # Yn
        dYn_primeOdYn = -2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.annHe6g_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.anpLi6g_frwrd(T_t) - rhoBBN*Yn3p5*self.B8naap_frwrd(T_t) - rhoBBN*Yn3p5*self.B8nBe7d_frwrd(T_t) - rhoBBN*Yn3p5*self.B8nLi6He3_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7naa_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7nLi7p_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) - rhoBBN*Yn1p2*self.ddHe3n_bkwrd(T_t) - rhoBBN*Yn1p1*self.dntg_frwrd(T_t) - rhoBBN*Yn1p2*self.He3nag_frwrd(T_t) - rhoBBN*Yn1p2*self.He3ntp_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3tanp_bkwrd(T_t) - rhoBBN*Yn3p4*self.Li6dBe7n_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li6nLi7g_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6nta_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6taan_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7daan_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7nLi8g_frwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7taann_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8paan_bkwrd(T_t) - rhoBBN*Yn0p1*self.npdg_frwrd(T_t) - nTOp_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*self.ppndp_frwrd(T_t) - rhoBBN*Yn2p2*self.tdan_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.ttann_bkwrd(T_t)
        dYn_primeOdYp = -rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.anpLi6g_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8naap_bkwrd(T_t) + rhoBBN*Yn4p3*self.Be7nLi7p_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn2p1*self.He3ntp_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.He3tanp_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) + rhoBBN*Yn5p3*self.Li8paan_frwrd(T_t) - rhoBBN*Yn1p0*self.npdg_frwrd(T_t) + nTOp_bkwrd(T_t) + rhoBBN*Yn1p1*self.ppndp_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.ppndp_frwrd(T_t)
        dYn_primeOdYd = rhoBBN*Yn3p4*self.B8nBe7d_bkwrd(T_t) + rhoBBN*Yn1p1*self.ddHe3n_frwrd(T_t) - rhoBBN*Yn1p0*self.dntg_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6dBe7n_frwrd(T_t) + rhoBBN*Yn4p3*self.Li7daan_frwrd(T_t) + self.npdg_bkwrd(T_t) + rhoBBN*Yn0p1*self.ppndp_bkwrd(T_t) + rhoBBN*Yn2p1*self.tdan_frwrd(T_t)
        dYn_primeOdYt = rhoBBN*Yn3p4*self.Be7taanp_frwrd(T_t) + self.dntg_bkwrd(T_t) + rhoBBN*Yn0p1*self.He3ntp_bkwrd(T_t) + rhoBBN*Yn1p2*self.He3tanp_frwrd(T_t) + rhoBBN*Yn2p2*self.Li6nta_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6taan_frwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7taann_frwrd(T_t) + rhoBBN*Yn1p1*self.tdan_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.ttann_frwrd(T_t)
        dYn_primeOdYHe3 = rhoBBN*Yn3p3*self.B8nLi6He3_bkwrd(T_t) - rhoBBN*Yn1p0*self.ddHe3n_bkwrd(T_t) - rhoBBN*Yn1p0*self.He3nag_frwrd(T_t) - rhoBBN*Yn1p0*self.He3ntp_frwrd(T_t) + rhoBBN*Yn2p1*self.He3tanp_frwrd(T_t) + rhoBBN*Yn4p3*self.Li7He3aanp_frwrd(T_t)
        dYn_primeOdYa = -rhoBBN*rhoBBN*Yn1p0*Yn1p0*self.annHe6g_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.anpLi6g_frwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.B8naap_bkwrd(T_t) + rhoBBN*Yn2p2*self.Be7naa_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Be7taanp_bkwrd(T_t) + self.He3nag_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.He3tanp_bkwrd(T_t) + rhoBBN*Yn2p1*self.Li6nta_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li6taan_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li7daan_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Li7He3aanp_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*self.Li7taann_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li8paan_bkwrd(T_t) - rhoBBN*Yn1p0*self.tdan_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn1p0*self.ttann_bkwrd(T_t)
        dYn_primeOdYLi7 = rhoBBN*Yn0p1*self.Be7nLi7p_bkwrd(T_t) + self.Li6nLi7g_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li7daan_frwrd(T_t) + rhoBBN*Yn1p2*self.Li7He3aanp_frwrd(T_t) - rhoBBN*Yn1p0*self.Li7nLi8g_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.Li7taann_frwrd(T_t)
        dYn_primeOdYBe7 = rhoBBN*Yn1p1*self.B8nBe7d_bkwrd(T_t) - rhoBBN*Yn1p0*self.Be7naa_frwrd(T_t) - rhoBBN*Yn1p0*self.Be7nLi7p_frwrd(T_t) + rhoBBN*Yn2p1*self.Be7taanp_frwrd(T_t) - rhoBBN*Yn1p0*self.Li6dBe7n_bkwrd(T_t)
        dYn_primeOdYHe6 = 2.*self.annHe6g_bkwrd(T_t)
        dYn_primeOdYLi8 = self.Li7nLi8g_bkwrd(T_t) + rhoBBN*Yn0p1*self.Li8paan_frwrd(T_t)
        dYn_primeOdYLi6 = self.anpLi6g_bkwrd(T_t) + rhoBBN*Yn1p2*self.B8nLi6He3_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6dBe7n_frwrd(T_t) - rhoBBN*Yn1p0*self.Li6nLi7g_frwrd(T_t) - rhoBBN*Yn1p0*self.Li6nta_frwrd(T_t) + rhoBBN*Yn2p1*self.Li6taan_frwrd(T_t)
        dYn_primeOdYB8 = -rhoBBN*Yn1p0*self.B8naap_frwrd(T_t) - rhoBBN*Yn1p0*self.B8nBe7d_frwrd(T_t) - rhoBBN*Yn1p0*self.B8nLi6He3_frwrd(T_t)
        dYn_row = [dYn_primeOdYn,dYn_primeOdYp,dYn_primeOdYd,dYn_primeOdYt,dYn_primeOdYHe3,dYn_primeOdYa,dYn_primeOdYLi7,dYn_primeOdYBe7,dYn_primeOdYHe6,dYn_primeOdYLi8,dYn_primeOdYLi6,dYn_primeOdYB8]

        # Yp
        dYp_primeOdYn = -rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.anpLi6g_frwrd(T_t) + rhoBBN*Yn3p5*self.B8naap_frwrd(T_t) + rhoBBN*Yn3p4*self.Be7nLi7p_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn1p2*self.He3ntp_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3tanp_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8paan_bkwrd(T_t) - rhoBBN*Yn0p1*self.npdg_frwrd(T_t) + nTOp_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*self.ppndp_frwrd(T_t)
        dYp_primeOdYp = -nTOp_bkwrd(T_t) + rhoBBN*(- self.npdg_frwrd(T_t)*Yn1p0 - self.dpHe3g_frwrd(T_t)*Yn1p1 - (self.tpag_frwrd(T_t) + self.ddtp_bkwrd(T_t) + self.He3ntp_bkwrd(T_t))*Yn2p1 - self.He3dap_bkwrd(T_t)*Yn2p2 - (self.Li7paa_frwrd(T_t) + self.Be7nLi7p_bkwrd(T_t))*Yn4p3) + rhoBBN*(-self.Li7paag_frwrd(T_t)*Yn4p3) + rhoBBN*(-self.Li6pBe7g_frwrd(T_t)*Yn3p3) + rhoBBN*(-self.Li6pHe3a_frwrd(T_t)*Yn3p3) + rhoBBN*(-0.5*rhoBBN*self.B8naap_bkwrd(T_t)*Yn2p2*Yn2p2) + rhoBBN*(-0.5*rhoBBN*self.Li6He3aap_bkwrd(T_t)*Yn2p2*Yn2p2 - self.Li6tLi8p_bkwrd(T_t)*Yn5p3) + rhoBBN*(-rhoBBN*rhoBBN*self.Be7He3ppaa_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - 2.*rhoBBN*self.He3He3app_bkwrd(T_t)*Yn0p1*Yn2p2) + rhoBBN*(-rhoBBN*self.He3tanp_bkwrd(T_t)*Yn1p0*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Li7He3aanp_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*self.Be7daap_bkwrd(T_t)*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*self.Be7taanp_bkwrd(T_t)*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*self.Be7He3aapp_bkwrd(T_t)*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*self.anpLi6g_frwrd(T_t)*Yn1p0*Yn2p2 - self.Li6dLi7p_bkwrd(T_t)*Yn4p3) + rhoBBN*(-self.Li7dLi8p_bkwrd(T_t)*Yn5p3) + rhoBBN*(-self.Li8paan_frwrd(T_t)*Yn5p3) + rhoBBN*(-rhoBBN*self.ppndp_frwrd(T_t)*Yn0p1*Yn1p0) + rhoBBN*(self.ppndp_bkwrd(T_t)*Yn1p1) + rhoBBN*(-self.Be7pB8g_frwrd(T_t)*Yn3p4)
        dYp_primeOdYd = rhoBBN*Yn3p4*self.Be7daap_frwrd(T_t) + rhoBBN*Yn1p1*self.ddtp_frwrd(T_t) - rhoBBN*Yn0p1*self.dpHe3g_frwrd(T_t) + rhoBBN*Yn1p2*self.He3dap_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6dLi7p_frwrd(T_t) + rhoBBN*Yn4p3*self.Li7dLi8p_frwrd(T_t) + self.npdg_bkwrd(T_t) + rhoBBN*Yn0p1*self.ppndp_bkwrd(T_t)
        dYp_primeOdYt = rhoBBN*Yn3p4*self.Be7taanp_frwrd(T_t) - rhoBBN*Yn0p1*self.ddtp_bkwrd(T_t) - rhoBBN*Yn0p1*self.He3ntp_bkwrd(T_t) + rhoBBN*Yn1p2*self.He3tanp_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6tLi8p_frwrd(T_t) - rhoBBN*Yn0p1*self.tpag_frwrd(T_t)
        dYp_primeOdYHe3 = 2.*rhoBBN*Yn3p4*self.Be7He3aapp_frwrd(T_t) + 2.*rhoBBN*Yn3p4*self.Be7He3ppaa_frwrd(T_t) + self.dpHe3g_bkwrd(T_t) + rhoBBN*Yn1p1*self.He3dap_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.He3He3app_frwrd(T_t) + rhoBBN*Yn1p0*self.He3ntp_frwrd(T_t) + rhoBBN*Yn2p1*self.He3tanp_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6He3aap_frwrd(T_t) + rhoBBN*Yn2p2*self.Li6pHe3a_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li7He3aanp_frwrd(T_t)
        dYp_primeOdYa = -rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.anpLi6g_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.B8naap_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Be7daap_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3aapp_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Be7taanp_bkwrd(T_t) - rhoBBN*Yn0p1*self.He3dap_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn0p1*self.He3He3app_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.He3tanp_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Li6He3aap_bkwrd(T_t) + rhoBBN*Yn1p2*self.Li6pHe3a_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Li7He3aanp_bkwrd(T_t) + rhoBBN*Yn2p2*self.Li7paa_bkwrd(T_t) + rhoBBN*Yn2p2*self.Li7paag_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li8paan_bkwrd(T_t) + self.tpag_bkwrd(T_t)
        dYp_primeOdYLi7 = -rhoBBN*Yn0p1*self.Be7nLi7p_bkwrd(T_t) - rhoBBN*Yn0p1*self.Li6dLi7p_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li7dLi8p_frwrd(T_t) + rhoBBN*Yn1p2*self.Li7He3aanp_frwrd(T_t) - rhoBBN*Yn0p1*self.Li7paa_frwrd(T_t) - rhoBBN*Yn0p1*self.Li7paag_frwrd(T_t)
        dYp_primeOdYBe7 = rhoBBN*Yn1p1*self.Be7daap_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Be7He3aapp_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Be7He3ppaa_frwrd(T_t) + rhoBBN*Yn1p0*self.Be7nLi7p_frwrd(T_t) - rhoBBN*Yn0p1*self.Be7pB8g_frwrd(T_t) + rhoBBN*Yn2p1*self.Be7taanp_frwrd(T_t) + self.Li6pBe7g_bkwrd(T_t)
        dYp_primeOdYHe6 = 0.
        dYp_primeOdYLi8 =  -rhoBBN*Yn0p1*self.Li6tLi8p_bkwrd(T_t) - rhoBBN*Yn0p1*self.Li7dLi8p_bkwrd(T_t) - rhoBBN*Yn0p1*self.Li8paan_frwrd(T_t)
        dYp_primeOdYLi6 = self.anpLi6g_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6dLi7p_frwrd(T_t) + rhoBBN*Yn1p2*self.Li6He3aap_frwrd(T_t) - rhoBBN*Yn0p1*self.Li6pBe7g_frwrd(T_t) - rhoBBN*Yn0p1*self.Li6pHe3a_frwrd(T_t) + rhoBBN*Yn2p1*self.Li6tLi8p_frwrd(T_t)
        dYp_primeOdYB8 = rhoBBN*Yn1p0*self.B8naap_frwrd(T_t) + self.Be7pB8g_bkwrd(T_t)
        dYp_row = [dYp_primeOdYn,dYp_primeOdYp,dYp_primeOdYd,dYp_primeOdYt,dYp_primeOdYHe3,dYp_primeOdYa,dYp_primeOdYLi7,dYp_primeOdYBe7,dYp_primeOdYHe6,dYp_primeOdYLi8,dYp_primeOdYLi6,dYp_primeOdYB8]

        # Yd
        dYd_primeOdYn = rhoBBN*Yn3p5*self.B8nBe7d_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.ddHe3n_bkwrd(T_t) - rhoBBN*Yn1p1*self.dntg_frwrd(T_t) + rhoBBN*Yn3p4*self.Li6dBe7n_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7daan_bkwrd(T_t) + rhoBBN*Yn0p1*self.npdg_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*self.ppndp_frwrd(T_t) + rhoBBN*Yn2p2*self.tdan_bkwrd(T_t)
        dYd_primeOdYp = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7daap_bkwrd(T_t) + 2.*rhoBBN*Yn2p1*self.ddtp_bkwrd(T_t) - rhoBBN*Yn1p1*self.dpHe3g_frwrd(T_t) + rhoBBN*Yn2p2*self.He3dap_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li6dLi7p_bkwrd(T_t) + rhoBBN*Yn5p3*self.Li7dLi8p_bkwrd(T_t) + rhoBBN*Yn1p0*self.npdg_frwrd(T_t) - rhoBBN*Yn1p1*self.ppndp_bkwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.ppndp_frwrd(T_t)
        dYd_primeOdYd = -rhoBBN*Yn3p5*self.B8dBe7He3_frwrd(T_t) - rhoBBN*Yn3p4*self.B8nBe7d_bkwrd(T_t) - rhoBBN*Yn3p4*self.Be7daap_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7taad_bkwrd(T_t) - rhoBBN*Yn2p2*self.daLi6g_frwrd(T_t) - 2.*rhoBBN*Yn1p1*self.ddag_frwrd(T_t) - 2.*rhoBBN*Yn1p1*self.ddHe3n_frwrd(T_t) - 2.*rhoBBN*Yn1p1*self.ddtp_frwrd(T_t) - rhoBBN*Yn1p0*self.dntg_frwrd(T_t) - rhoBBN*Yn0p1*self.dpHe3g_frwrd(T_t) - rhoBBN*Yn1p2*self.He3dap_frwrd(T_t) - rhoBBN*Yn2p2*self.He3tad_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li6dBe7n_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6dLi7p_frwrd(T_t) - rhoBBN*Yn3p4*self.Li6He3Be7d_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li6tLi7d_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7daan_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7dLi8p_frwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7He3aad_bkwrd(T_t) - rhoBBN*Yn5p3*self.Li8dLi7t_frwrd(T_t) - self.npdg_bkwrd(T_t) - rhoBBN*Yn0p1*self.ppndp_bkwrd(T_t) - rhoBBN*Yn2p1*self.tdan_frwrd(T_t)
        dYd_primeOdYt = rhoBBN*Yn3p4*self.Be7taad_frwrd(T_t) + 2.*rhoBBN*Yn0p1*self.ddtp_bkwrd(T_t) + self.dntg_bkwrd(T_t) + rhoBBN*Yn1p2*self.He3tad_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6tLi7d_frwrd(T_t) + rhoBBN*Yn4p3*self.Li8dLi7t_bkwrd(T_t) - rhoBBN*Yn1p1*self.tdan_frwrd(T_t)
        dYd_primeOdYHe3 = rhoBBN*Yn3p4*self.B8dBe7He3_bkwrd(T_t) + 2.*rhoBBN*Yn1p0*self.ddHe3n_bkwrd(T_t) + self.dpHe3g_bkwrd(T_t) - rhoBBN*Yn1p1*self.He3dap_frwrd(T_t) + rhoBBN*Yn2p1*self.He3tad_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6He3Be7d_frwrd(T_t) + rhoBBN*Yn4p3*self.Li7He3aad_frwrd(T_t)
        dYd_primeOdYa = rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Be7daap_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Be7taad_bkwrd(T_t) - rhoBBN*Yn1p1*self.daLi6g_frwrd(T_t) + 2.*self.ddag_bkwrd(T_t) + rhoBBN*Yn0p1*self.He3dap_bkwrd(T_t) - rhoBBN*Yn1p1*self.He3tad_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li7daan_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Li7He3aad_bkwrd(T_t) + rhoBBN*Yn1p0*self.tdan_bkwrd(T_t)
        dYd_primeOdYLi7 = rhoBBN*Yn0p1*self.Li6dLi7p_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li6tLi7d_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li7daan_frwrd(T_t) - rhoBBN*Yn1p1*self.Li7dLi8p_frwrd(T_t) + rhoBBN*Yn1p2*self.Li7He3aad_frwrd(T_t) + rhoBBN*Yn2p1*self.Li8dLi7t_bkwrd(T_t)
        dYd_primeOdYBe7 = rhoBBN*Yn1p2*self.B8dBe7He3_bkwrd(T_t) - rhoBBN*Yn1p1*self.B8nBe7d_bkwrd(T_t) - rhoBBN*Yn1p1*self.Be7daap_frwrd(T_t) + rhoBBN*Yn2p1*self.Be7taad_frwrd(T_t) + rhoBBN*Yn1p0*self.Li6dBe7n_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li6He3Be7d_bkwrd(T_t)
        dYd_primeOdYHe6 = 0.
        dYd_primeOdYLi8 = rhoBBN*Yn0p1*self.Li7dLi8p_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li8dLi7t_frwrd(T_t)
        dYd_primeOdYLi6 = self.daLi6g_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li6dBe7n_frwrd(T_t) - rhoBBN*Yn1p1*self.Li6dLi7p_frwrd(T_t) + rhoBBN*Yn1p2*self.Li6He3Be7d_frwrd(T_t) + rhoBBN*Yn2p1*self.Li6tLi7d_frwrd(T_t)
        dYd_primeOdYB8 = -rhoBBN*Yn1p1*self.B8dBe7He3_frwrd(T_t) + rhoBBN*Yn1p0*self.B8nBe7d_frwrd(T_t)
        dYd_row = [dYd_primeOdYn,dYd_primeOdYp,dYd_primeOdYd,dYd_primeOdYt,dYd_primeOdYHe3,dYd_primeOdYa,dYd_primeOdYLi7,dYd_primeOdYBe7,dYd_primeOdYHe6,dYd_primeOdYLi8,dYd_primeOdYLi6,dYd_primeOdYB8]

        # Yt
        dYt_primeOdYn = 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn1p1*self.dntg_frwrd(T_t) + rhoBBN*Yn1p2*self.He3ntp_frwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3tanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6nta_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6taan_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7taann_bkwrd(T_t) + rhoBBN*Yn2p2*self.tdan_bkwrd(T_t) + 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.ttann_bkwrd(T_t)
        dYt_primeOdYp = 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) - rhoBBN*Yn2p1*self.ddtp_bkwrd(T_t) - rhoBBN*Yn2p1*self.He3ntp_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.He3tanp_bkwrd(T_t) + rhoBBN*Yn5p3*self.Li6tLi8p_bkwrd(T_t) - rhoBBN*Yn2p1*self.tpag_frwrd(T_t)
        dYt_primeOdYd = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7taad_bkwrd(T_t) + rhoBBN*Yn1p1*self.ddtp_frwrd(T_t) + rhoBBN*Yn1p0*self.dntg_frwrd(T_t) + rhoBBN*Yn2p2*self.He3tad_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li6tLi7d_bkwrd(T_t) + rhoBBN*Yn5p3*self.Li8dLi7t_frwrd(T_t) - rhoBBN*Yn2p1*self.tdan_frwrd(T_t)
        dYt_primeOdYt = -rhoBBN*Yn3p5*self.B8taaHe3_frwrd(T_t) - rhoBBN*Yn3p5*self.B8tBe7a_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7taad_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7taanp_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7tLi6a_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7tLi7He3_frwrd(T_t) - rhoBBN*Yn0p1*self.ddtp_bkwrd(T_t) - self.dntg_bkwrd(T_t) - rhoBBN*Yn0p1*self.He3ntp_bkwrd(T_t) - rhoBBN*Yn1p2*self.He3tad_frwrd(T_t) - rhoBBN*Yn1p2*self.He3tanp_frwrd(T_t) - rhoBBN*Yn1p2*self.He3tLi6g_frwrd(T_t) - rhoBBN*Yn2p2*self.Li6nta_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li6taan_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6tLi7d_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6tLi8p_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7taann_frwrd(T_t) - rhoBBN*Yn4p3*self.Li8dLi7t_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8He3aat_bkwrd(T_t) - rhoBBN*Yn2p2*self.taLi7g_frwrd(T_t) - rhoBBN*Yn0p1*self.tpag_frwrd(T_t) - rhoBBN*Yn1p1*self.tdan_frwrd(T_t) - 2.*rhoBBN*Yn2p1*self.ttann_frwrd(T_t)
        dYt_primeOdYHe3 = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) + rhoBBN*Yn4p3*self.Be7tLi7He3_bkwrd(T_t) + rhoBBN*Yn1p0*self.He3ntp_frwrd(T_t) - rhoBBN*Yn2p1*self.He3tad_frwrd(T_t) - rhoBBN*Yn2p1*self.He3tanp_frwrd(T_t) - rhoBBN*Yn2p1*self.He3tLi6g_frwrd(T_t) + rhoBBN*Yn5p3*self.Li8He3aat_frwrd(T_t)
        dYt_primeOdYa = rhoBBN*rhoBBN*Yn1p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) + rhoBBN*Yn3p4*self.B8tBe7a_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Be7taad_bkwrd(T_t) + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Be7tLi6a_bkwrd(T_t) + rhoBBN*Yn1p1*self.He3tad_bkwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.He3tanp_bkwrd(T_t) - rhoBBN*Yn2p1*self.Li6nta_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li6taan_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*self.Li7taann_bkwrd(T_t) - rhoBBN*rhoBBN*Yn2p1*Yn2p2*self.Li8He3aat_bkwrd(T_t) - rhoBBN*Yn2p1*self.taLi7g_frwrd(T_t) + self.tpag_bkwrd(T_t) + rhoBBN*Yn1p0*self.tdan_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn1p0*self.ttann_bkwrd(T_t)
        dYt_primeOdYLi7 = rhoBBN*Yn1p2*self.Be7tLi7He3_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6tLi7d_bkwrd(T_t) - rhoBBN*Yn2p1*self.Li7taann_frwrd(T_t) - rhoBBN*Yn2p1*self.Li8dLi7t_bkwrd(T_t) + self.taLi7g_bkwrd(T_t)
        dYt_primeOdYBe7 = rhoBBN*Yn2p2*self.B8tBe7a_bkwrd(T_t) - rhoBBN*Yn2p1*self.Be7taad_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7taanp_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7tLi6a_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7tLi7He3_frwrd(T_t)
        dYt_primeOdYHe6 = 0.
        dYt_primeOdYLi8 = rhoBBN*Yn0p1*self.Li6tLi8p_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li8dLi7t_frwrd(T_t) + rhoBBN*Yn1p2*self.Li8He3aat_frwrd(T_t)
        dYt_primeOdYLi6 = rhoBBN*Yn2p2*self.Be7tLi6a_bkwrd(T_t) + self.He3tLi6g_bkwrd(T_t) + rhoBBN*Yn1p0*self.Li6nta_frwrd(T_t) - rhoBBN*Yn2p1*self.Li6taan_frwrd(T_t) - rhoBBN*Yn2p1*self.Li6tLi7d_frwrd(T_t) - rhoBBN*Yn2p1*self.Li6tLi8p_frwrd(T_t)
        dYt_primeOdYB8 = -rhoBBN*Yn2p1*self.B8taaHe3_frwrd(T_t) - rhoBBN*Yn2p1*self.B8tBe7a_frwrd(T_t)
        dYt_row = [dYt_primeOdYn,dYt_primeOdYp,dYt_primeOdYd,dYt_primeOdYt,dYt_primeOdYHe3,dYt_primeOdYa,dYt_primeOdYLi7,dYt_primeOdYBe7,dYt_primeOdYHe6,dYt_primeOdYLi8,dYt_primeOdYLi6,dYt_primeOdYB8]

        # YHe3
        dYHe3_primeOdYn = rhoBBN*Yn3p5*self.B8nLi6He3_frwrd(T_t) - rhoBBN*Yn1p2*self.ddHe3n_bkwrd(T_t) - rhoBBN*Yn1p2*self.He3nag_frwrd(T_t) - rhoBBN*Yn1p2*self.He3ntp_frwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3tanp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t)
        dYHe3_primeOdYp = 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7He3aapp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) + rhoBBN*Yn1p1*self.dpHe3g_frwrd(T_t) + rhoBBN*Yn2p2*self.He3dap_bkwrd(T_t) + 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3He3app_bkwrd(T_t) + rhoBBN*Yn2p1*self.He3ntp_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.He3tanp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6He3aap_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6pHe3a_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t)
        dYHe3_primeOdYd = rhoBBN*Yn3p5*self.B8dBe7He3_frwrd(T_t) + rhoBBN*Yn1p1*self.ddHe3n_frwrd(T_t) + rhoBBN*Yn0p1*self.dpHe3g_frwrd(T_t) - rhoBBN*Yn1p2*self.He3dap_frwrd(T_t) + rhoBBN*Yn2p2*self.He3tad_bkwrd(T_t) + rhoBBN*Yn3p4*self.Li6He3Be7d_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7He3aad_bkwrd(T_t)
        dYHe3_primeOdYt = rhoBBN*Yn3p5*self.B8taaHe3_frwrd(T_t) + rhoBBN*Yn3p4*self.Be7tLi7He3_frwrd(T_t) + rhoBBN*Yn0p1*self.He3ntp_bkwrd(T_t) - rhoBBN*Yn1p2*self.He3tad_frwrd(T_t) - rhoBBN*Yn1p2*self.He3tanp_frwrd(T_t) - rhoBBN*Yn1p2*self.He3tLi6g_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8He3aat_bkwrd(T_t)
        dYHe3_primeOdYHe3 = -rhoBBN*Yn3p4*self.B8dBe7He3_bkwrd(T_t) - rhoBBN*Yn3p3*self.B8nLi6He3_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) - rhoBBN*Yn3p4*self.Be7He3aapp_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7He3ppaa_frwrd(T_t) - rhoBBN*Yn4p3*self.Be7tLi7He3_bkwrd(T_t) - rhoBBN*Yn1p0*self.ddHe3n_bkwrd(T_t) - self.dpHe3g_bkwrd(T_t) - rhoBBN*Yn2p2*self.He3aBe7g_frwrd(T_t) - rhoBBN*Yn1p1*self.He3dap_frwrd(T_t) - 2.*rhoBBN*Yn1p2*self.He3He3app_frwrd(T_t) - rhoBBN*Yn1p0*self.He3nag_frwrd(T_t) - rhoBBN*Yn1p0*self.He3ntp_frwrd(T_t) - rhoBBN*Yn2p1*self.He3tad_frwrd(T_t) - rhoBBN*Yn2p1*self.He3tanp_frwrd(T_t) - rhoBBN*Yn2p1*self.He3tLi6g_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6He3aap_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6He3Be7d_frwrd(T_t) - rhoBBN*Yn2p2*self.Li6pHe3a_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7He3aad_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7He3aanp_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7He3Li6a_frwrd(T_t) - rhoBBN*Yn5p3*self.Li8He3aat_frwrd(T_t) - rhoBBN*Yn5p3*self.Li8He3Li7a_frwrd(T_t)
        dYHe3_primeOdYa = -rhoBBN*rhoBBN*Yn1p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3aapp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) - rhoBBN*Yn1p2*self.He3aBe7g_frwrd(T_t) + rhoBBN*Yn0p1*self.He3dap_bkwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn0p1*self.He3He3app_bkwrd(T_t) + self.He3nag_bkwrd(T_t) + rhoBBN*Yn1p1*self.He3tad_bkwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.He3tanp_bkwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Li6He3aap_bkwrd(T_t) - rhoBBN*Yn1p2*self.Li6pHe3a_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Li7He3aad_bkwrd(T_t) + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Li7He3aanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li7He3Li6a_bkwrd(T_t) + rhoBBN*rhoBBN*Yn2p1*Yn2p2*self.Li8He3aat_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li8He3Li7a_bkwrd(T_t)
        dYHe3_primeOdYLi7 = -rhoBBN*Yn1p2*self.Be7tLi7He3_bkwrd(T_t) - rhoBBN*Yn1p2*self.Li7He3aad_frwrd(T_t) - rhoBBN*Yn1p2*self.Li7He3aanp_frwrd(T_t) - rhoBBN*Yn1p2*self.Li7He3Li6a_frwrd(T_t) + rhoBBN*Yn2p2*self.Li8He3Li7a_bkwrd(T_t)
        dYHe3_primeOdYBe7 = -rhoBBN*Yn1p2*self.B8dBe7He3_bkwrd(T_t) - rhoBBN*Yn1p2*self.Be7He3aapp_frwrd(T_t) - rhoBBN*Yn1p2*self.Be7He3ppaa_frwrd(T_t) + rhoBBN*Yn2p1*self.Be7tLi7He3_frwrd(T_t) + self.He3aBe7g_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6He3Be7d_bkwrd(T_t)
        dYHe3_primeOdYHe6 = 0.
        dYHe3_primeOdYLi8 = -rhoBBN*Yn1p2*self.Li8He3aat_frwrd(T_t) - rhoBBN*Yn1p2*self.Li8He3Li7a_frwrd(T_t)
        dYHe3_primeOdYLi6 = -rhoBBN*Yn1p2*self.B8nLi6He3_bkwrd(T_t) + self.He3tLi6g_bkwrd(T_t) - rhoBBN*Yn1p2*self.Li6He3aap_frwrd(T_t) - rhoBBN*Yn1p2*self.Li6He3Be7d_frwrd(T_t) + rhoBBN*Yn0p1*self.Li6pHe3a_frwrd(T_t) + rhoBBN*Yn2p2*self.Li7He3Li6a_bkwrd(T_t)
        dYHe3_primeOdYB8 = rhoBBN*Yn1p1*self.B8dBe7He3_frwrd(T_t) + rhoBBN*Yn1p0*self.B8nLi6He3_frwrd(T_t) + rhoBBN*Yn2p1*self.B8taaHe3_frwrd(T_t)
        dYHe3_row = [dYHe3_primeOdYn,dYHe3_primeOdYp,dYHe3_primeOdYd,dYHe3_primeOdYt,dYHe3_primeOdYHe3,dYHe3_primeOdYa,dYHe3_primeOdYLi7,dYHe3_primeOdYBe7,dYHe3_primeOdYHe6,dYHe3_primeOdYLi8,dYHe3_primeOdYLi6,dYHe3_primeOdYB8]

        # Ya
        dYa_primeOdYn = -rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.annHe6g_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.anpLi6g_frwrd(T_t) + 2.*rhoBBN*Yn3p5*self.B8naap_frwrd(T_t) + 2.*rhoBBN*Yn3p4*self.Be7naa_frwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn1p2*self.He3nag_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3tanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6nta_frwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6taan_bkwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7daan_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7taann_bkwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8paan_bkwrd(T_t) - rhoBBN*Yn2p2*self.tdan_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.ttann_bkwrd(T_t)
        dYa_primeOdYp = -rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.anpLi6g_frwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8naap_bkwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7daap_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7He3aapp_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) - rhoBBN*Yn2p2*self.He3dap_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.He3He3app_bkwrd(T_t) - rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.He3tanp_bkwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6He3aap_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6pHe3a_frwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7paa_frwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7paag_frwrd(T_t) + 2.*rhoBBN*Yn5p3*self.Li8paan_frwrd(T_t) + rhoBBN*Yn2p1*self.tpag_frwrd(T_t)
        dYa_primeOdYd = 2.*rhoBBN*Yn3p4*self.Be7daap_frwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7taad_bkwrd(T_t) - rhoBBN*Yn2p2*self.daLi6g_frwrd(T_t) + rhoBBN*Yn1p1*self.ddag_frwrd(T_t) + rhoBBN*Yn1p2*self.He3dap_frwrd(T_t) - rhoBBN*Yn2p2*self.He3tad_bkwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7daan_frwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7He3aad_bkwrd(T_t) + rhoBBN*Yn2p1*self.tdan_frwrd(T_t)
        dYa_primeOdYt = 2.*rhoBBN*Yn3p5*self.B8taaHe3_frwrd(T_t) + rhoBBN*Yn3p5*self.B8tBe7a_frwrd(T_t) + 2.*rhoBBN*Yn3p4*self.Be7taad_frwrd(T_t) + 2.*rhoBBN*Yn3p4*self.Be7taanp_frwrd(T_t) + rhoBBN*Yn3p4*self.Be7tLi6a_frwrd(T_t) + rhoBBN*Yn1p2*self.He3tad_frwrd(T_t) + rhoBBN*Yn1p2*self.He3tanp_frwrd(T_t) - rhoBBN*Yn2p2*self.Li6nta_bkwrd(T_t) + 2.*rhoBBN*Yn3p3*self.Li6taan_frwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7taann_frwrd(T_t) - rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8He3aat_bkwrd(T_t) - rhoBBN*Yn2p2*self.taLi7g_frwrd(T_t) + rhoBBN*Yn0p1*self.tpag_frwrd(T_t) + rhoBBN*Yn1p1*self.tdan_frwrd(T_t) + rhoBBN*Yn2p1*self.ttann_frwrd(T_t)
        dYa_primeOdYHe3 = -rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) + 2.*rhoBBN*Yn3p4*self.Be7He3aapp_frwrd(T_t) + 2.*rhoBBN*Yn3p4*self.Be7He3ppaa_frwrd(T_t) - rhoBBN*Yn2p2*self.He3aBe7g_frwrd(T_t) + rhoBBN*Yn1p1*self.He3dap_frwrd(T_t) + rhoBBN*Yn1p2*self.He3He3app_frwrd(T_t) + rhoBBN*Yn1p0*self.He3nag_frwrd(T_t) + rhoBBN*Yn2p1*self.He3tad_frwrd(T_t) + rhoBBN*Yn2p1*self.He3tanp_frwrd(T_t) + 2.*rhoBBN*Yn3p3*self.Li6He3aap_frwrd(T_t) - rhoBBN*Yn2p2*self.Li6pHe3a_bkwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7He3aad_frwrd(T_t) + 2.*rhoBBN*Yn4p3*self.Li7He3aanp_frwrd(T_t) + rhoBBN*Yn4p3*self.Li7He3Li6a_frwrd(T_t) + 2.*rhoBBN*Yn5p3*self.Li8He3aat_frwrd(T_t) + rhoBBN*Yn5p3*self.Li8He3Li7a_frwrd(T_t)
        dYa_primeOdYa = -0.5*rhoBBN*rhoBBN*Yn1p0*Yn1p0*self.annHe6g_frwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.anpLi6g_frwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.B8naap_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) - rhoBBN*Yn3p4*self.B8tBe7a_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Be7daap_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3aapp_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) - 2.*rhoBBN*Yn2p2*self.Be7naa_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Be7taad_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Be7taanp_bkwrd(T_t) - rhoBBN*Yn3p3*self.Be7tLi6a_bkwrd(T_t) - rhoBBN*Yn1p1*self.daLi6g_frwrd(T_t) - self.ddag_bkwrd(T_t) - rhoBBN*Yn1p2*self.He3aBe7g_frwrd(T_t) - rhoBBN*Yn0p1*self.He3dap_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*self.He3He3app_bkwrd(T_t) - self.He3nag_bkwrd(T_t) - rhoBBN*Yn1p1*self.He3tad_bkwrd(T_t) - rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.He3tanp_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Li6He3aap_bkwrd(T_t) - rhoBBN*Yn2p1*self.Li6nta_bkwrd(T_t) - rhoBBN*Yn1p2*self.Li6pHe3a_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li6taan_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li7daan_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Li7He3aad_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Li7He3aanp_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li7He3Li6a_bkwrd(T_t) - 2.*rhoBBN*Yn2p2*self.Li7paa_bkwrd(T_t) - 2.*rhoBBN*Yn2p2*self.Li7paag_bkwrd(T_t) - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*self.Li7taann_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn2p1*Yn2p2*self.Li8He3aat_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li8He3Li7a_bkwrd(T_t) - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li8paan_bkwrd(T_t) - rhoBBN*Yn2p1*self.taLi7g_frwrd(T_t) - self.tpag_bkwrd(T_t) - rhoBBN*Yn1p0*self.tdan_bkwrd(T_t) - 0.5*rhoBBN*rhoBBN*Yn1p0*Yn1p0*self.ttann_bkwrd(T_t)
        dYa_primeOdYLi7 = 2.*rhoBBN*Yn1p1*self.Li7daan_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Li7He3aad_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Li7He3aanp_frwrd(T_t) + rhoBBN*Yn1p2*self.Li7He3Li6a_frwrd(T_t) + 2.*rhoBBN*Yn0p1*self.Li7paa_frwrd(T_t) + 2.*rhoBBN*Yn0p1*self.Li7paag_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.Li7taann_frwrd(T_t) - rhoBBN*Yn2p2*self.Li8He3Li7a_bkwrd(T_t) + self.taLi7g_bkwrd(T_t)
        dYa_primeOdYBe7 = -rhoBBN*Yn2p2*self.B8tBe7a_bkwrd(T_t) + 2.*rhoBBN*Yn1p1*self.Be7daap_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Be7He3aapp_frwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Be7He3ppaa_frwrd(T_t) + 2.*rhoBBN*Yn1p0*self.Be7naa_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.Be7taad_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.Be7taanp_frwrd(T_t) + rhoBBN*Yn2p1*self.Be7tLi6a_frwrd(T_t) + self.He3aBe7g_bkwrd(T_t)
        dYa_primeOdYHe6 = self.annHe6g_bkwrd(T_t)
        dYa_primeOdYLi8 = 2.*rhoBBN*Yn1p2*self.Li8He3aat_frwrd(T_t) + rhoBBN*Yn1p2*self.Li8He3Li7a_frwrd(T_t) + 2.*rhoBBN*Yn0p1*self.Li8paan_frwrd(T_t)
        dYa_primeOdYLi6 = self.anpLi6g_bkwrd(T_t) - rhoBBN*Yn2p2*self.Be7tLi6a_bkwrd(T_t) + self.daLi6g_bkwrd(T_t) + 2.*rhoBBN*Yn1p2*self.Li6He3aap_frwrd(T_t) + rhoBBN*Yn1p0*self.Li6nta_frwrd(T_t) + rhoBBN*Yn0p1*self.Li6pHe3a_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.Li6taan_frwrd(T_t) - rhoBBN*Yn2p2*self.Li7He3Li6a_bkwrd(T_t)
        dYa_primeOdYB8 = 2.*rhoBBN*Yn1p0*self.B8naap_frwrd(T_t) + 2.*rhoBBN*Yn2p1*self.B8taaHe3_frwrd(T_t) + rhoBBN*Yn2p1*self.B8tBe7a_frwrd(T_t)
        dYa_row = [dYa_primeOdYn,dYa_primeOdYp,dYa_primeOdYd,dYa_primeOdYt,dYa_primeOdYHe3,dYa_primeOdYa,dYa_primeOdYLi7,dYa_primeOdYBe7,dYa_primeOdYHe6,dYa_primeOdYLi8,dYa_primeOdYLi6,dYa_primeOdYB8]

        # YLi7
        dYLi7_primeOdYn = rhoBBN*Yn3p4*self.Be7nLi7p_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6nLi7g_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7daan_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7nLi8g_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7taann_bkwrd(T_t)
        dYLi7_primeOdYp = -rhoBBN*Yn4p3*self.Be7nLi7p_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li6dLi7p_bkwrd(T_t) + rhoBBN*Yn5p3*self.Li7dLi8p_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Li7He3aanp_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7paa_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7paag_frwrd(T_t)
        dYLi7_primeOdYd = rhoBBN*Yn3p3*self.Li6dLi7p_frwrd(T_t) - rhoBBN*Yn4p3*self.Li6tLi7d_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7daan_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7dLi8p_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li7He3aad_bkwrd(T_t) + rhoBBN*Yn5p3*self.Li8dLi7t_frwrd(T_t)
        dYLi7_primeOdYt = rhoBBN*Yn3p4*self.Be7tLi7He3_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6tLi7d_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7taann_frwrd(T_t) - rhoBBN*Yn4p3*self.Li8dLi7t_bkwrd(T_t) + rhoBBN*Yn2p2*self.taLi7g_frwrd(T_t)
        dYLi7_primeOdYHe3 = -rhoBBN*Yn4p3*self.Be7tLi7He3_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li7He3aad_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7He3aanp_frwrd(T_t) - rhoBBN*Yn4p3*self.Li7He3Li6a_frwrd(T_t) + rhoBBN*Yn5p3*self.Li8He3Li7a_frwrd(T_t)
        dYLi7_primeOdYa = rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li7daan_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Li7He3aad_bkwrd(T_t) + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Li7He3aanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li7He3Li6a_bkwrd(T_t) + rhoBBN*Yn2p2*self.Li7paa_bkwrd(T_t) + rhoBBN*Yn2p2*self.Li7paag_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*self.Li7taann_bkwrd(T_t) - rhoBBN*Yn4p3*self.Li8He3Li7a_bkwrd(T_t) + rhoBBN*Yn2p1*self.taLi7g_frwrd(T_t)
        dYLi7_primeOdYLi7 = -rhoBBN*Yn0p1*self.Be7nLi7p_bkwrd(T_t) - rhoBBN*Yn1p2*self.Be7tLi7He3_bkwrd(T_t) - rhoBBN*Yn0p1*self.Li6dLi7p_bkwrd(T_t) - self.Li6nLi7g_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li6tLi7d_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li7daan_frwrd(T_t) - rhoBBN*Yn1p1*self.Li7dLi8p_frwrd(T_t) - rhoBBN*Yn1p2*self.Li7He3aad_frwrd(T_t) - rhoBBN*Yn1p2*self.Li7He3aanp_frwrd(T_t) - rhoBBN*Yn1p2*self.Li7He3Li6a_frwrd(T_t) - rhoBBN*Yn1p0*self.Li7nLi8g_frwrd(T_t) - rhoBBN*Yn0p1*self.Li7paa_frwrd(T_t) - rhoBBN*Yn0p1*self.Li7paag_frwrd(T_t) - rhoBBN*Yn2p1*self.Li7taann_frwrd(T_t) - rhoBBN*Yn2p1*self.Li8dLi7t_bkwrd(T_t) - rhoBBN*Yn2p2*self.Li8He3Li7a_bkwrd(T_t) - self.taLi7g_bkwrd(T_t)
        dYLi7_primeOdYBe7 = rhoBBN*Yn1p0*self.Be7nLi7p_frwrd(T_t) + rhoBBN*Yn2p1*self.Be7tLi7He3_frwrd(T_t)
        dYLi7_primeOdYHe6 = 0.
        dYLi7_primeOdYLi8 = rhoBBN*Yn0p1*self.Li7dLi8p_bkwrd(T_t) + self.Li7nLi8g_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li8dLi7t_frwrd(T_t) + rhoBBN*Yn1p2*self.Li8He3Li7a_frwrd(T_t)
        dYLi7_primeOdYLi6 = rhoBBN*Yn1p1*self.Li6dLi7p_frwrd(T_t) + rhoBBN*Yn1p0*self.Li6nLi7g_frwrd(T_t) + rhoBBN*Yn2p1*self.Li6tLi7d_frwrd(T_t) + rhoBBN*Yn2p2*self.Li7He3Li6a_bkwrd(T_t)
        dYLi7_primeOdYB8 = 0.
        dYLi7_row = [dYLi7_primeOdYn,dYLi7_primeOdYp,dYLi7_primeOdYd,dYLi7_primeOdYt,dYLi7_primeOdYHe3,dYLi7_primeOdYa,dYLi7_primeOdYLi7,dYLi7_primeOdYBe7,dYLi7_primeOdYHe6,dYLi7_primeOdYLi8,dYLi7_primeOdYLi6,dYLi7_primeOdYB8]

        # YBe7
        dYBe7_primeOdYn = rhoBBN*Yn3p5*self.B8nBe7d_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7naa_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7nLi7p_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) - rhoBBN*Yn3p4*self.Li6dBe7n_bkwrd(T_t)
        dYBe7_primeOdYp = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7daap_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7He3aapp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) + rhoBBN*Yn4p3*self.Be7nLi7p_bkwrd(T_t) - rhoBBN*Yn3p4*self.Be7pB8g_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6pBe7g_frwrd(T_t)
        dYBe7_primeOdYd = rhoBBN*Yn3p5*self.B8dBe7He3_frwrd(T_t) - rhoBBN*Yn3p4*self.B8nBe7d_bkwrd(T_t) - rhoBBN*Yn3p4*self.Be7daap_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Be7taad_bkwrd(T_t) + rhoBBN*Yn3p3*self.Li6dBe7n_frwrd(T_t) - rhoBBN*Yn3p4*self.Li6He3Be7d_bkwrd(T_t)
        dYBe7_primeOdYt = rhoBBN*Yn3p5*self.B8tBe7a_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7taad_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7taanp_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7tLi6a_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7tLi7He3_frwrd(T_t)
        dYBe7_primeOdYHe3 = -rhoBBN*Yn3p4*self.B8dBe7He3_bkwrd(T_t) - rhoBBN*Yn3p4*self.Be7He3aapp_frwrd(T_t) - rhoBBN*Yn3p4*self.Be7He3ppaa_frwrd(T_t) + rhoBBN*Yn4p3*self.Be7tLi7He3_bkwrd(T_t) + rhoBBN*Yn2p2*self.He3aBe7g_frwrd(T_t) + rhoBBN*Yn3p3*self.Li6He3Be7d_frwrd(T_t)
        dYBe7_primeOdYa = -rhoBBN*Yn3p4*self.B8tBe7a_bkwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Be7daap_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3aapp_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*self.Be7He3ppaa_bkwrd(T_t) + rhoBBN*Yn2p2*self.Be7naa_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p1*Yn2p2*self.Be7taad_bkwrd(T_t) + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*self.Be7taanp_bkwrd(T_t) + rhoBBN*Yn3p3*self.Be7tLi6a_bkwrd(T_t) + rhoBBN*Yn1p2*self.He3aBe7g_frwrd(T_t)
        dYBe7_primeOdYLi7 = rhoBBN*Yn0p1*self.Be7nLi7p_bkwrd(T_t) + rhoBBN*Yn1p2*self.Be7tLi7He3_bkwrd(T_t)
        dYBe7_primeOdYBe7 = -rhoBBN*Yn1p2*self.B8dBe7He3_bkwrd(T_t) - rhoBBN*Yn1p1*self.B8nBe7d_bkwrd(T_t) - rhoBBN*Yn2p2*self.B8tBe7a_bkwrd(T_t) - rhoBBN*Yn1p1*self.Be7daap_frwrd(T_t) - rhoBBN*Yn1p2*self.Be7He3aapp_frwrd(T_t) - rhoBBN*Yn1p2*self.Be7He3ppaa_frwrd(T_t) - rhoBBN*Yn1p0*self.Be7naa_frwrd(T_t) - rhoBBN*Yn1p0*self.Be7nLi7p_frwrd(T_t) - rhoBBN*Yn0p1*self.Be7pB8g_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7taad_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7taanp_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7tLi6a_frwrd(T_t) - rhoBBN*Yn2p1*self.Be7tLi7He3_frwrd(T_t) - self.He3aBe7g_bkwrd(T_t) - rhoBBN*Yn1p0*self.Li6dBe7n_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li6He3Be7d_bkwrd(T_t) - self.Li6pBe7g_bkwrd(T_t)
        dYBe7_primeOdYHe6 = 0.
        dYBe7_primeOdYLi8 = 0.
        dYBe7_primeOdYLi6 = rhoBBN*Yn2p2*self.Be7tLi6a_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6dBe7n_frwrd(T_t) + rhoBBN*Yn1p2*self.Li6He3Be7d_frwrd(T_t) + rhoBBN*Yn0p1*self.Li6pBe7g_frwrd(T_t)
        dYBe7_primeOdYB8 = rhoBBN*Yn1p1*self.B8dBe7He3_frwrd(T_t) + rhoBBN*Yn1p0*self.B8nBe7d_frwrd(T_t) + rhoBBN*Yn2p1*self.B8tBe7a_frwrd(T_t) + self.Be7pB8g_bkwrd(T_t)
        dYBe7_row = [dYBe7_primeOdYn,dYBe7_primeOdYp,dYBe7_primeOdYd,dYBe7_primeOdYt,dYBe7_primeOdYHe3,dYBe7_primeOdYa,dYBe7_primeOdYLi7,dYBe7_primeOdYBe7,dYBe7_primeOdYHe6,dYBe7_primeOdYLi8,dYBe7_primeOdYLi6,dYBe7_primeOdYB8]

        # YHe6
        dYHe6_primeOdYn = rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.annHe6g_frwrd(T_t)
        dYHe6_primeOdYp = 0.
        dYHe6_primeOdYd = 0.
        dYHe6_primeOdYt = 0.
        dYHe6_primeOdYHe3 = 0.
        dYHe6_primeOdYa = 0.5*rhoBBN*rhoBBN*Yn1p0*Yn1p0*self.annHe6g_frwrd(T_t)
        dYHe6_primeOdYLi7 = 0.
        dYHe6_primeOdYBe7 = 0.
        dYHe6_primeOdYHe6 = -self.annHe6g_bkwrd(T_t)
        dYHe6_primeOdYLi8 = 0.
        dYHe6_primeOdYLi6 = 0.
        dYHe6_primeOdYB8 = 0.
        dYHe6_row = [dYHe6_primeOdYn,dYHe6_primeOdYp,dYHe6_primeOdYd,dYHe6_primeOdYt,dYHe6_primeOdYHe3,dYHe6_primeOdYa,dYHe6_primeOdYLi7,dYHe6_primeOdYBe7,dYHe6_primeOdYHe6,dYHe6_primeOdYLi8,dYHe6_primeOdYLi6,dYHe6_primeOdYB8]

        # YLi8
        dYLi8_primeOdYn = rhoBBN*Yn4p3*self.Li7nLi8g_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8paan_bkwrd(T_t)
        dYLi8_primeOdYp = -rhoBBN*Yn5p3*self.Li6tLi8p_bkwrd(T_t) - rhoBBN*Yn5p3*self.Li7dLi8p_bkwrd(T_t) - rhoBBN*Yn5p3*self.Li8paan_frwrd(T_t)
        dYLi8_primeOdYd = rhoBBN*Yn4p3*self.Li7dLi8p_frwrd(T_t) - rhoBBN*Yn5p3*self.Li8dLi7t_frwrd(T_t)
        dYLi8_primeOdYt = rhoBBN*Yn3p3*self.Li6tLi8p_frwrd(T_t) + rhoBBN*Yn4p3*self.Li8dLi7t_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li8He3aat_bkwrd(T_t)
        dYLi8_primeOdYHe3 = -rhoBBN*Yn5p3*self.Li8He3aat_frwrd(T_t) - rhoBBN*Yn5p3*self.Li8He3Li7a_frwrd(T_t)
        dYLi8_primeOdYa = rhoBBN*rhoBBN*Yn2p1*Yn2p2*self.Li8He3aat_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li8He3Li7a_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li8paan_bkwrd(T_t)
        dYLi8_primeOdYLi7 = rhoBBN*Yn1p1*self.Li7dLi8p_frwrd(T_t) + rhoBBN*Yn1p0*self.Li7nLi8g_frwrd(T_t) + rhoBBN*Yn2p1*self.Li8dLi7t_bkwrd(T_t) + rhoBBN*Yn2p2*self.Li8He3Li7a_bkwrd(T_t)
        dYLi8_primeOdYBe7 = 0.
        dYLi8_primeOdYHe6 = 0.
        dYLi8_primeOdYLi8 = -rhoBBN*Yn0p1*self.Li6tLi8p_bkwrd(T_t) - rhoBBN*Yn0p1*self.Li7dLi8p_bkwrd(T_t) - self.Li7nLi8g_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li8dLi7t_frwrd(T_t) - rhoBBN*Yn1p2*self.Li8He3aat_frwrd(T_t) - rhoBBN*Yn1p2*self.Li8He3Li7a_frwrd(T_t) - rhoBBN*Yn0p1*self.Li8paan_frwrd(T_t)
        dYLi8_primeOdYLi6 = rhoBBN*Yn2p1*self.Li6tLi8p_frwrd(T_t)
        dYLi8_primeOdYB8 = 0.
        dYLi8_row = [dYLi8_primeOdYn,dYLi8_primeOdYp,dYLi8_primeOdYd,dYLi8_primeOdYt,dYLi8_primeOdYHe3,dYLi8_primeOdYa,dYLi8_primeOdYLi7,dYLi8_primeOdYBe7,dYLi8_primeOdYHe6,dYLi8_primeOdYLi8,dYLi8_primeOdYLi6,dYLi8_primeOdYB8]

        # YLi6
        dYLi6_primeOdYn = rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.anpLi6g_frwrd(T_t) + rhoBBN*Yn3p5*self.B8nLi6He3_frwrd(T_t) + rhoBBN*Yn3p4*self.Li6dBe7n_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li6nLi7g_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6nta_frwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6taan_bkwrd(T_t)
        dYLi6_primeOdYp = rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.anpLi6g_frwrd(T_t) + rhoBBN*Yn4p3*self.Li6dLi7p_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.Li6He3aap_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li6pBe7g_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6pHe3a_frwrd(T_t) + rhoBBN*Yn5p3*self.Li6tLi8p_bkwrd(T_t)
        dYLi6_primeOdYd = rhoBBN*Yn2p2*self.daLi6g_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6dBe7n_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6dLi7p_frwrd(T_t) + rhoBBN*Yn3p4*self.Li6He3Be7d_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li6tLi7d_bkwrd(T_t)
        dYLi6_primeOdYt = rhoBBN*Yn3p4*self.Be7tLi6a_frwrd(T_t) + rhoBBN*Yn1p2*self.He3tLi6g_frwrd(T_t) + rhoBBN*Yn2p2*self.Li6nta_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li6taan_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6tLi7d_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6tLi8p_frwrd(T_t)
        dYLi6_primeOdYHe3 = -rhoBBN*Yn3p3*self.B8nLi6He3_bkwrd(T_t) + rhoBBN*Yn2p1*self.He3tLi6g_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6He3aap_frwrd(T_t) - rhoBBN*Yn3p3*self.Li6He3Be7d_frwrd(T_t) + rhoBBN*Yn2p2*self.Li6pHe3a_bkwrd(T_t) + rhoBBN*Yn4p3*self.Li7He3Li6a_frwrd(T_t)
        dYLi6_primeOdYa = rhoBBN*rhoBBN*Yn0p1*Yn1p0*self.anpLi6g_frwrd(T_t) - rhoBBN*Yn3p3*self.Be7tLi6a_bkwrd(T_t) + rhoBBN*Yn1p1*self.daLi6g_frwrd(T_t) + rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.Li6He3aap_bkwrd(T_t) + rhoBBN*Yn2p1*self.Li6nta_bkwrd(T_t) + rhoBBN*Yn1p2*self.Li6pHe3a_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p0*Yn2p2*self.Li6taan_bkwrd(T_t) - rhoBBN*Yn3p3*self.Li7He3Li6a_bkwrd(T_t)
        dYLi6_primeOdYLi7 = rhoBBN*Yn0p1*self.Li6dLi7p_bkwrd(T_t) + self.Li6nLi7g_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6tLi7d_bkwrd(T_t) + rhoBBN*Yn1p2*self.Li7He3Li6a_frwrd(T_t)
        dYLi6_primeOdYBe7 = rhoBBN*Yn2p1*self.Be7tLi6a_frwrd(T_t) + rhoBBN*Yn1p0*self.Li6dBe7n_bkwrd(T_t) + rhoBBN*Yn1p1*self.Li6He3Be7d_bkwrd(T_t) + self.Li6pBe7g_bkwrd(T_t)
        dYLi6_primeOdYHe6 = 0.
        dYLi6_primeOdYLi8 = rhoBBN*Yn0p1*self.Li6tLi8p_bkwrd(T_t)
        dYLi6_primeOdYLi6 = -self.anpLi6g_bkwrd(T_t) - rhoBBN*Yn1p2*self.B8nLi6He3_bkwrd(T_t) - rhoBBN*Yn2p2*self.Be7tLi6a_bkwrd(T_t) - self.daLi6g_bkwrd(T_t) - self.He3tLi6g_bkwrd(T_t) - rhoBBN*Yn1p1*self.Li6dBe7n_frwrd(T_t) - rhoBBN*Yn1p1*self.Li6dLi7p_frwrd(T_t) - rhoBBN*Yn1p2*self.Li6He3aap_frwrd(T_t) - rhoBBN*Yn1p2*self.Li6He3Be7d_frwrd(T_t) - rhoBBN*Yn1p0*self.Li6nLi7g_frwrd(T_t) - rhoBBN*Yn1p0*self.Li6nta_frwrd(T_t) - rhoBBN*Yn0p1*self.Li6pBe7g_frwrd(T_t) - rhoBBN*Yn0p1*self.Li6pHe3a_frwrd(T_t) - rhoBBN*Yn2p1*self.Li6taan_frwrd(T_t) - rhoBBN*Yn2p1*self.Li6tLi7d_frwrd(T_t) - rhoBBN*Yn2p1*self.Li6tLi8p_frwrd(T_t) - rhoBBN*Yn2p2*self.Li7He3Li6a_bkwrd(T_t)
        dYLi6_primeOdYB8 = rhoBBN*Yn1p0*self.B8nLi6He3_frwrd(T_t)
        dYLi6_row = [dYLi6_primeOdYn,dYLi6_primeOdYp,dYLi6_primeOdYd,dYLi6_primeOdYt,dYLi6_primeOdYHe3,dYLi6_primeOdYa,dYLi6_primeOdYLi7,dYLi6_primeOdYBe7,dYLi6_primeOdYHe6,dYLi6_primeOdYLi8,dYLi6_primeOdYLi6,dYLi6_primeOdYB8]

        # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4, YHe6 -> Yn4p2, Li8 -> Yn5p3, Li6 -> Yn3p3, B8 -> Yn3p5}
        # YB8
        dYB8_primeOdYn = -rhoBBN*Yn3p5*self.B8naap_frwrd(T_t) - rhoBBN*Yn3p5*self.B8nBe7d_frwrd(T_t) - rhoBBN*Yn3p5*self.B8nLi6He3_frwrd(T_t)
        dYB8_primeOdYp = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8naap_bkwrd(T_t) + rhoBBN*Yn3p4*self.Be7pB8g_frwrd(T_t)
        dYB8_primeOdYd = -rhoBBN*Yn3p5*self.B8dBe7He3_frwrd(T_t) + rhoBBN*Yn3p4*self.B8nBe7d_bkwrd(T_t)
        dYB8_primeOdYt = -rhoBBN*Yn3p5*self.B8taaHe3_frwrd(T_t) - rhoBBN*Yn3p5*self.B8tBe7a_frwrd(T_t)
        dYB8_primeOdYHe3 = rhoBBN*Yn3p4*self.B8dBe7He3_bkwrd(T_t) + rhoBBN*Yn3p3*self.B8nLi6He3_bkwrd(T_t) + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*self.B8taaHe3_bkwrd(T_t)
        dYB8_primeOdYa = rhoBBN*rhoBBN*Yn0p1*Yn2p2*self.B8naap_bkwrd(T_t) + rhoBBN*rhoBBN*Yn1p2*Yn2p2*self.B8taaHe3_bkwrd(T_t) + rhoBBN*Yn3p4*self.B8tBe7a_bkwrd(T_t)
        dYB8_primeOdYLi7 = 0.
        dYB8_primeOdYBe7 = rhoBBN*Yn1p2*self.B8dBe7He3_bkwrd(T_t) + rhoBBN*Yn1p1*self.B8nBe7d_bkwrd(T_t) + rhoBBN*Yn2p2*self.B8tBe7a_bkwrd(T_t) + rhoBBN*Yn0p1*self.Be7pB8g_frwrd(T_t)
        dYB8_primeOdYHe6 = 0.
        dYB8_primeOdYLi8 = 0.
        dYB8_primeOdYLi6 = rhoBBN*Yn1p2*self.B8nLi6He3_bkwrd(T_t)
        dYB8_primeOdYB8 = -rhoBBN*Yn1p1*self.B8dBe7He3_frwrd(T_t) - rhoBBN*Yn1p0*self.B8naap_frwrd(T_t) - rhoBBN*Yn1p0*self.B8nBe7d_frwrd(T_t) - rhoBBN*Yn1p0*self.B8nLi6He3_frwrd(T_t) - rhoBBN*Yn2p1*self.B8taaHe3_frwrd(T_t) - rhoBBN*Yn2p1*self.B8tBe7a_frwrd(T_t) - self.Be7pB8g_bkwrd(T_t)
        dYB8_row = [dYB8_primeOdYn,dYB8_primeOdYp,dYB8_primeOdYd,dYB8_primeOdYt,dYB8_primeOdYHe3,dYB8_primeOdYa,dYB8_primeOdYLi7,dYB8_primeOdYBe7,dYB8_primeOdYHe6,dYB8_primeOdYLi8,dYB8_primeOdYLi6,dYB8_primeOdYB8]

        return [dYn_row,dYp_row,dYd_row,dYt_row,dYHe3_row,dYa_row,dYLi7_row,dYBe7_row, dYHe6_row, dYLi8_row, dYLi6_row, dYB8_row]
