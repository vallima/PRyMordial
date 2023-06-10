# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini

my_dir = PRyMini.working_dir

def InterpolateWeakRates():
    # Upload p,n weak rates
    # HT interval: T_start --> T_weak
    nTOp_frwrd_HT_tab = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_HT.txt")
    nTOp_frwrd_HT = interp1d(nTOp_frwrd_HT_tab[:,0],nTOp_frwrd_HT_tab[:,1], bounds_error=False,fill_value="extrapolate",kind='quadratic')
    nTOp_bkwrd_HT_tab = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_HT.txt")
    nTOp_bkwrd_HT = interp1d(nTOp_bkwrd_HT_tab[:,0],nTOp_bkwrd_HT_tab[:,1], bounds_error=False,fill_value="extrapolate",kind='quadratic')
    # MT interval: T_weak --> T_nucl
    nTOp_frwrd_MT_tab = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_MT.txt")
    nTOp_frwrd_MT = interp1d(nTOp_frwrd_MT_tab[:,0],nTOp_frwrd_MT_tab[:,1], bounds_error=False,fill_value="extrapolate",kind='quadratic')
    nTOp_bkwrd_MT_tab = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_MT.txt")
    nTOp_bkwrd_MT = interp1d(nTOp_bkwrd_MT_tab[:,0],nTOp_bkwrd_MT_tab[:,1], bounds_error=False,fill_value="extrapolate",kind='quadratic')
    # LT interval: T_nucl --> T_end
    nTOp_frwrd_LT_tab = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_frwrd_LT.txt")
    nTOp_frwrd_LT = interp1d(nTOp_frwrd_LT_tab[:,0],nTOp_frwrd_LT_tab[:,1], bounds_error=False,fill_value="extrapolate",kind='quadratic')
    nTOp_bkwrd_LT_tab = np.loadtxt(my_dir+"/PRyMrates/nTOp/"+"nTOp_bkwrd_LT.txt")
    nTOp_bkwrd_LT = interp1d(nTOp_bkwrd_LT_tab[:,0],nTOp_bkwrd_LT_tab[:,1], bounds_error=False,fill_value="extrapolate",kind='quadratic')
    return [nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT]

def RecomputeWeakRates(Tvec):
    ##########################
    # Change of units to CGS #
    ##########################
    if(PRyMini.verbose_flag):
        print("Switch from natural units to CGS.")
    if(PRyMini.compute_nTOp_flag):
        if(PRyMini.verbose_flag):
            print(" ")
            print("Computing n <--> p weak rates from scratch.")
            print("Change flags in PRyM_init.py otherwise!")
        import PRyM.PRyM_eval_nTOp as PRyMevalnTOp
        T_interval_HT,nTOp_frwrdvec_HT,nTOp_bkwrdvec_HT,T_interval_MT,nTOp_frwrdvec_MT,nTOp_bkwrdvec_MT,T_interval_LT,nTOp_frwrdvec_LT,nTOp_bkwrdvec_LT = PRyMevalnTOp.ComputeWeakRates(Tvec)
        # HT interval: T_start --> T_weak
        nTOp_frwrd_HT = interp1d(T_interval_HT,nTOp_frwrdvec_HT,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        nTOp_bkwrd_HT = interp1d(T_interval_HT,nTOp_bkwrdvec_HT,bounds_error=False,fill_value="extrapolate",kind='quadratic')
            # MT interval: T_weak --> T_nucl
        nTOp_frwrd_MT = interp1d(T_interval_MT,nTOp_frwrdvec_MT,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        nTOp_bkwrd_MT = interp1d(T_interval_MT,nTOp_bkwrdvec_MT,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        # LT interval: T_nucl --> T_end
        nTOp_frwrd_LT = interp1d(T_interval_LT,nTOp_frwrdvec_LT,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        nTOp_bkwrd_LT = interp1d(T_interval_LT,nTOp_bkwrdvec_LT,bounds_error=False,fill_value="extrapolate",kind='quadratic')
        return [nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT]
    else:
        nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT = InterpolateWeakRates()
        return [nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT]
