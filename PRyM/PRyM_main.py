# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta
import numdifftools as nd

def PRyMresults(my_rho_NP=0.,my_p_NP=0.,my_drho_NP_dT=0.,my_delta_rho_NP=0.):

    #############################
    # PRyMordial initialization #
    #############################
    import PRyM.PRyM_init as PRyMini
    if(PRyMini.julia_flag):
        from julia import Main
        from diffeqpy import de
        import PRyM.PRyM_jl_sys as PRyMjl
    import PRyM.PRyM_thermo as PRyMthermo
    # Loading New Physics species (constructor default: none)
    PRyMthermo.rho_NP,PRyMthermo.p_NP,PRyMthermo.drho_NP_dT,PRyMthermo.delta_rho_NP=my_rho_NP,my_p_NP,my_drho_NP_dT,my_delta_rho_NP
    
    if(PRyMini.verbose_flag):
        print(" ")
        print("###########################################################")
        print("################## Welcome to PRyMordial ##################")
        print("###########################################################")
        start_time = time.time()

    ################################
    # PRyMordial working directory #
    ################################
    my_dir = PRyMini.working_dir
    
    ##############################
    # Definition of temperatures #
    ##############################
    Tstart_MeV = PRyMini.T_start/PRyMini.MeV_to_Kelvin
    Tend_MeV = PRyMini.T_end/PRyMini.MeV_to_Kelvin

    ##################
    # Thermodynamics #
    ##################
    # Units adopted for background:
    # - Time in [s]
    # - Energy, temperature in [MeV]

    # Expansion rate from Friedmann equation
    def Hubble(Tg,Tnue,Tnumu,T_NP=0.):
        rho_pl = PRyMthermo.rho_g(Tg)+PRyMthermo.rho_e(Tg)-PRyMthermo.PofT(Tg)+Tg*PRyMthermo.dPdT(Tg)
        rho_3nu = PRyMthermo.rho_nu(Tnue)+2.*PRyMthermo.rho_nu(Tnumu)
        rho_tot = rho_pl+rho_3nu
        # Extra radiation density due to Delta Neff
        rho_tot += PRyMini.normDeltaNeff*PRyMini.DeltaNeff*PRyMthermo.rho_g(Tg)
        if(PRyMini.NP_thermo_flag):
            rho_tot += PRyMthermo.rho_NP(T_NP)
        if(PRyMini.NP_nu_flag):
            rho_tot += PRyMthermo.rho_NP(Tnue)
        if(PRyMini.NP_e_flag):
            rho_tot += PRyMthermo.rho_NP(Tg)
        return PRyMini.MeV_to_secm1*(rho_tot*8.*np.pi/(3.*PRyMini.Mpl**2))**0.5
    # Computing the background (if not pre-stored)
    if(PRyMini.compute_bckg_flag):
        # Integrated Boltzmann equations for temperature of species
        # Neutrino temperature evolution
        def dTnudt(Tg,Tnue,Tnumu,T_NP=0.):
            Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
            num = -12.*Hubble_T*PRyMthermo.rho_nu(Tnue)
            delta_rho_nu = (PRyMthermo.delta_rho_nue(Tg,Tnue,Tnumu)+2.*PRyMthermo.delta_rho_numu(Tg,Tnue,Tnumu))
            if(PRyMini.NP_thermo_flag):
                delta_rho_nu += PRyMthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP)
            num += delta_rho_nu
            den = 3.*PRyMthermo.drho_nu_dT(Tnue)
            if(PRyMini.NP_nu_flag):
                num -= 3.*Hubble_T*(PRyMthermo.rho_NP(Tnue)+PRyMthermo.p_NP(Tnue))
                den += PRyMthermo.drho_NP_dT(Tnue)
            return num/den
        # Plasma temperature evolution
        def dTgdt(Tg,Tnue,Tnumu,T_NP=0.):
            Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
            num = -(Hubble_T*(4.*PRyMthermo.rho_g(Tg)+3.*(PRyMthermo.rho_e(Tg)+PRyMthermo.p_e(Tg))+3.*Tg*PRyMthermo.dPdT(Tg)))
            # Sum of collision terms must vanish
            delta_rho_g = -(PRyMthermo.delta_rho_nue(Tg,Tnue,Tnumu)+2.*PRyMthermo.delta_rho_numu(Tg,Tnue,Tnumu))
            if(PRyMini.NP_thermo_flag):
                delta_rho_g -= PRyMthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP) # traceless collision operator
            num += delta_rho_g
            den = PRyMthermo.drho_g_dT(Tg)+PRyMthermo.drho_e_dT(Tg)+Tg*PRyMthermo.d2PdT2(Tg)
            if(PRyMini.NP_e_flag):
                num -= 3.*Hubble_T*(PRyMthermo.rho_NP(Tg)+PRyMthermo.p_NP(Tg))
                den += PRyMthermo.drho_NP_dT(Tg)
            return num/den
        # NP temperature evolution
        def dTNPdt(Tg,Tnue,Tnumu,T_NP):
            Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
            rho_NP = PRyMthermo.rho_NP(T_NP)
            p_NP = PRyMthermo.p_NP(T_NP)
            num = -3.*Hubble_T*(rho_NP+p_NP)
            delta_rho_NP = PRyMthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP)
            num += delta_rho_NP
            den = PRyMthermo.drho_NP_dT(T_NP)
            return num/den
        def dTtotdt(t,T_vec):
            if(PRyMini.NP_thermo_flag):
                Tg,Tnu,T_NP = T_vec
                y_vec = dTgdt(Tg,Tnu,Tnu,T_NP),dTnudt(Tg,Tnu,Tnu,T_NP),dTNPdt(Tg,Tnue,Tnumu,T_NP)
                return y_vec
            else:
                Tg,Tnu = T_vec
                y_vec = dTgdt(Tg,Tnu,Tnu),dTnudt(Tg,Tnu,Tnu)
                return y_vec
        # Solution of Boltzmann equations for background thermodynamics
        tfin = PRyMini.t_end # [s]
        if(PRyMini.NP_thermo_flag):
            tini = 1./(2.*Hubble(Tstart_MeV,Tstart_MeV,Tstart_MeV,PRyMini.Tstart_NP)) # [s]
            sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PRyMini.n_sampling)
            sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
            Tini_vec = [Tstart_MeV,Tstart_MeV,PRyMini.Tstart_NP]
            if(PRyMini.julia_flag):
                T0 = np.float64(Tini_vec)
                tspan = (np.float64(tini),np.float64(tfin))
                p0 = [lambda w,x,y,z: np.float64(dTgdt(w,x,y,z)),lambda w,x,y,z: np.float64(dTnudt(w,x,y,z)),lambda w,x,y,z: np.float64(dTNPdt(w,x,y,z))]
                prob = de.ODEProblem(PRyMjl.dTtotdtNPjl,T0,tspan,p0)
                sol_thermo = de.solve(prob,de.Tsit5(),saveat=sol_thermo_sampling,reltol=1.e-6,abstol=1.e-9)
                t_vec = sol_thermo.t
                sol_thermo = np.array(sol_thermo.u)
                Tg_vec = sol_thermo[:,0]
                Tnu_vec = sol_thermo[:,1]
                TNP_vec = sol_thermo[:,2]
            else:
                sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                t_vec = sol_thermo.t
                Tg_vec = sol_thermo.y[0][:]
                Tnu_vec = sol_thermo.y[1][:]
                TNP_vec = sol_thermo.y[2][:]
        else:
            tini = 1./(2.*Hubble(Tstart_MeV,Tstart_MeV,Tstart_MeV)) # s
            sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PRyMini.n_sampling)
            sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
            Tini_vec = [Tstart_MeV,Tstart_MeV]
            if(PRyMini.julia_flag):
                T0 = np.float64(Tini_vec)
                tspan = (np.float64(tini),np.float64(tfin))
                p0 = [lambda x,y,z: np.float64(dTgdt(x,y,z)),lambda x,y,z: np.float64(dTnudt(x,y,z))]
                prob = de.ODEProblem(PRyMjl.dTtotdtSMjl,T0,tspan,p0)
                sol_thermo = de.solve(prob,de.Tsit5(),saveat=sol_thermo_sampling,reltol=1.e-6,abstol=1.e-9)
                t_vec = sol_thermo.t
                sol_thermo = np.array(sol_thermo.u)
                Tg_vec = sol_thermo[:,0]
                Tnu_vec = sol_thermo[:,1]
            else:
                sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                t_vec = sol_thermo.t
                Tg_vec = sol_thermo.y[0][:]
                Tnu_vec = sol_thermo.y[1][:]
        # Save results for background thermodynamics
        if(PRyMini.save_bckg_flag):
            if(PRyMini.NP_thermo_flag):
                np.savetxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu_TNP.txt",np.c_[t_vec,Tg_vec,Tnu_vec,TNP_vec])
            else:
                np.savetxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu.txt",np.c_[t_vec,Tg_vec,Tnu_vec])
    else:
        if(PRyMini.NP_thermo_flag):
            t_vec,Tg_vec,Tnu_vec,TNP_vec = np.loadtxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu_TNP.txt",unpack=True)
        else:
            t_vec,Tg_vec,Tnu_vec = np.loadtxt(my_dir+"/PRyMrates/"+"thermo/Tgamma_Tnu.txt",unpack=True)
            
    # Interpolation of Tnu(T) (and NP) for non-instantaneous decoupling effecs in a(T)
    if(PRyMini.aTid_flag):
        TnuofT = interp1d(Tg_vec[:],Tnu_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        if(PRyMini.NP_thermo_flag):
            TNPofT = interp1d(Tg_vec[:],TNP_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
    
    ################
    # N effective  #
    ################
    # Definition as extra radiation density relative to photons in units of 8/7 x (11/4)^(4/3)
    def N_eff(Tg,Tnue,Tnumu,T_NP=0.):
        rho_gamma = PRyMthermo.rho_g(Tg)
        rho_rad_tot = PRyMthermo.rho_nu(Tnue)+2.*PRyMthermo.rho_nu(Tnumu)+rho_gamma
        rho_rad_tot += PRyMini.normDeltaNeff*PRyMini.DeltaNeff*PRyMthermo.rho_g(Tg)
        if(PRyMini.NP_thermo_flag):
            rho_rad_tot += PRyMthermo.rho_NP(T_NP)
        elif(PRyMini.NP_nu_flag):
            rho_rad_tot += PRyMthermo.rho_NP(Tnue)
        elif(PRyMini.NP_e_flag):
            rho_rad_tot += PRyMthermo.rho_NP(Tg)
        return (rho_rad_tot-rho_gamma)/rho_gamma/PRyMini.normDeltaNeff
        
    ################################
    # Relic abundance of neutrinos #
    ################################
    # Cosmic abundance of single species of relativistic nu
    def Omeganuh2_relnu():
        Tnu0 = Tnu_vec[-1]/Tg_vec[-1]*PRyMini.T0CMB/PRyMini.MeV_to_Kelvin
        return (7.*np.pi**2/120.*Tnu0**4)/PRyMini.rhocOverh2 # dimensionless
    # Cosmic abundance of non-relativistic nu over sum of nu masses
    def Omeganuh2_nrnu():
        Tnu0 = Tnu_vec[-1]/Tg_vec[-1]*PRyMini.T0CMB/PRyMini.MeV_to_Kelvin
        return (3./2.*zeta(3)/np.pi**2*Tnu0**3)/PRyMini.rhocOverh2 # MeV
    
    ######################################################
    # FRW cosmological backround in radiation domination #
    ######################################################
    # Relation between time and temperature of the thermal bath
    t_of_T = interp1d(Tg_vec[:],t_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
    t_of_T_vec = np.vectorize(t_of_T)
    T_of_t = interp1d(t_vec[:],Tg_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
    T_of_t_vec = np.vectorize(T_of_t)
    
    ######################################################
    # Relation of scale factor with temperature and time #
    ######################################################
    # Non-instantaneous decoupling effects on the entropy of the plasma
    if(PRyMini.aTid_flag):
        # Heat rate due to neutrino (and NP) interactions with the plasma
        def N_nu_rate(T):
            Tnu = TnuofT(T)
            qdot_pl = -(PRyMthermo.delta_rho_nue(T,Tnu,Tnu)+2.*PRyMthermo.delta_rho_numu(T,Tnu,Tnu))
            Hubble_T = Hubble(T,Tnu,Tnu)
            if(PRyMini.NP_thermo_flag):
                TNP = TNPofT(T)
                qdot_pl -= PRyMthermo.delta_rho_NP(T,Tnu,Tnu,TNP)
                Hubble_T = Hubble(T,Tnu,Tnu,TNP)
            res = -qdot_pl/Hubble_T/T**4
            return res
        # Plasma entropy density normalized to T^3 (constant after e+- annihilation)
        def sbar(T):
            return PRyMthermo.spl(T)/T**3
        # Numerical derivative of the above wrt to temperature
        dsbardT = nd.Derivative(sbar,n=1)
        # dlog(a*T)/dlog(T)
        def dlnadlnT(lnT):
            T = np.exp(lnT)
            sbar_T = sbar(T)
            N_nu_T = N_nu_rate(T)
            return -(3.*sbar_T+T*dsbardT(T))/(3.*sbar_T+N_nu_T)
        # Log of scale factor as a function of log of temperature of thermal bath
        Tini_vec = [np.log(Tend_MeV),np.log(Tstart_MeV)]
        # Initial conditions using z = a*T and entropy conservation
        z0 = PRyMini.T0CMB/PRyMini.MeV_to_Kelvin # a0 = 1 --> z0 = T0
        # Assuming no change in plasma entropy per comoving volume after end of BBN
        zend = (z0/(sbar(Tend_MeV)/PRyMini.s0bar)**(1/3)) # iff d(spl*a^3) = 0
        # aend conveniently allows to sample from end of BBN instead of today
        T_sol_vec = np.logspace(np.log10(Tend_MeV),np.log10(Tstart_MeV),PRyMini.n_sampling)
        if(PRyMini.julia_flag):
            logaend_vec = [np.log(zend/Tend_MeV)]
            logaend = np.float64(logaend_vec)
            Tspan = (np.float64(np.log(Tend_MeV)),np.float64(np.log(Tstart_MeV)))
            p0 = [lambda x: np.float64(dlnadlnT(x))]
            prob = de.ODEProblem(PRyMjl.dlnajl,logaend,Tspan,p0)
            sol_lnalnT = de.solve(prob,de.Tsit5(),saveat=np.log(T_sol_vec),reltol=1.e-6,abstol=1.e-9)
            sol_lnT = sol_lnalnT.t
            sol_lnalnT = np.array(sol_lnalnT.u)
            sol_lna = sol_lnalnT[:,0]
        else:
            def dlna(lnT,y):
                return dlnadlnT(lnT)
            sol_lnalnT = solve_ivp(dlna,Tini_vec,[np.log(zend/Tend_MeV)],t_eval=np.log(T_sol_vec),method='LSODA',rtol=1.e-6,atol=1.e-9)
            sol_lnT = np.array(sol_lnalnT.t[:]).flatten()
            sol_lna = np.array(sol_lnalnT.y[:]).flatten()
        # log(a) as a function of log(T)
        lnalnT = interp1d(sol_lnT,sol_lna,bounds_error=False,fill_value="extrapolate")
    
    # Scale factor as a function of temperature of thermal bath
    def a_of_T(T):
        # Including non-instantaneous decoupling effects
        if(PRyMini.aTid_flag):
            return np.exp(lnalnT(np.log(T)))
        # Using instantaneous approximation
        else:
            spl_T = PRyMthermo.spl(T)
            return (PRyMini.s0CMB/spl_T)**(1./3.)
    a_of_T_vec = np.vectorize(a_of_T)
    # Scale factor as a function of time
    a_in = a_of_T(Tg_vec[0])
    a_fin = a_of_T(Tg_vec[-1])
    a_of_t = interp1d(t_vec[:],a_of_T_vec(Tg_vec),bounds_error=False,fill_value=(a_in,a_fin))
    
    ##########################################
    # Baryon density for the nuclear network #
    ##########################################
    # Baryon number density obtained as n0B = rho0B/mB
    # mB = averaged baryon mass in MeV: assumes helium fraction of 24.7%
    # rho0B = atomic density (it includes electron mass + binding energies)
    def nB(a):
        n0B = PRyMini.n0CMB*PRyMini.eta0 # baryon number density of today MeV^3
        return n0B/a**3 # MeV^3
    # Baryon-to-photon ratio as a function of temperature given in [K]
    def eta_of_T(T_K):
        T_MeV = T_K/PRyMini.MeV_to_Kelvin
        ngCMB = (2.*zeta(3))/(np.pi**2)*T_MeV**3
        return nB(a_of_T(T_MeV))/ngCMB
    # Baryon energy density adopted in the nuclear network
    # rhoB = nucleonic density (i.e. rho0B measured by CMB x ma/mB)
    def rhoB_BBN(a):
        n0B = PRyMini.n0CMB*PRyMini.eta0
        rho0BmaOvermB = PRyMini.ma*n0B
        return rho0BmaOvermB*PRyMini.MeV4_to_gcmm3/a**3 # CGS, a0 = 1
        
    ######################################################
    # Definition of temperature eras for nuclear network #
    ######################################################
    t_start = t_of_T(PRyMini.T_start/PRyMini.MeV_to_Kelvin)
    t_weak = t_of_T(PRyMini.T_weak/PRyMini.MeV_to_Kelvin)
    t_nucl = t_of_T(PRyMini.T_nucl/PRyMini.MeV_to_Kelvin)
    t_end = t_of_T(PRyMini.T_end/PRyMini.MeV_to_Kelvin)

    ##############################
    # Import n <--> p weak rates #
    ###############################
    import PRyM.PRyM_eval_nTOp as PRyMevalnTOp
    import PRyM.PRyM_nTOp as PRyMnTOp
    nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT = PRyMnTOp.RecomputeWeakRates([Tg_vec,Tnu_vec])
    
    ############################
    # Weak rates normalization #
    ############################
    if(PRyMini.tau_n_flag):
        Fn = PRyMevalnTOp.ComputeFn()
        NormWeakRates = 1./(Fn*PRyMini.tau_n) # normalization in [s-1]
    else:
        GFtilde2 = (PRyMini.GF*PRyMini.Vud)**2*(1+3.*PRyMini.gA**2)/(2.*np.pi**3)
        NormWeakRates = PRyMini.MeV_to_secm1*(GFtilde2*PRyMini.me**5) # normalization in [s-1]
    
    ##################################
    # High temperature era: only p,n #
    ##################################
    # Initial conditions from detailed balance
    def Yni(T):
        return nTOp_bkwrd_HT(T)/(nTOp_bkwrd_HT(T) + nTOp_frwrd_HT(T))
    def Ypi(T):
        return (1.-Yni(T))

    # Weak rates at HT
    def nTOp_frwrd(T):
        return NormWeakRates*nTOp_frwrd_HT(T)
    def nTOp_bkwrd(T):
        return NormWeakRates*nTOp_bkwrd_HT(T)
        
    def Yn_prime_HT(t,Y):
        T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
        return nTOp_bkwrd(T_t)*Y[1]-nTOp_frwrd(T_t)*Y[0]
        
    def Yp_prime_HT(t,Y):
        T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
        return nTOp_frwrd(T_t)*Y[0]-nTOp_bkwrd(T_t)*Y[1]
        
    def Y_prime_HT(t,Y):
        dY = Yn_prime_HT(t,Y),Yp_prime_HT(t,Y)
        return dY

    #############################
    # High temperature solution #
    #############################
    if(PRyMini.verbose_flag):
        print(" ")
        print("Solving neutron decoupling at high temperature era")
        
    # HT era definition
    t_init = t_start
    t_fin = t_weak

    # HT initial conditions
    Yn_i = Yni(PRyMini.T_start)
    Yp_i = Ypi(PRyMini.T_start)
    
    # Solving HT network
    Yi_vec = [Yn_i,Yp_i]
    if(PRyMini.julia_flag):
        Y0 = np.float64(Yi_vec)
        tspan = (np.float64(t_init),np.float64(t_fin))
        p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(nTOp_frwrd(x)),lambda x: np.float64(nTOp_bkwrd(x))]
        f_Y_prime_HT_jl = de.ODEFunction(PRyMjl.Y_prime_HT_jl,jac = PRyMjl.Jacobian_HT_jl)
        prob = de.ODEProblem(f_Y_prime_HT_jl,Y0,tspan,p0,reltol=1.e-6,abstol=1.e-9)
        sol_at_HT = de.solve(prob,de.RadauIIA5())
        sol_at_HT = np.array(sol_at_HT.u)
        Yn_HT_f,Yp_HT_f = sol_at_HT[-1,:]
    else:
        sol_at_HT = solve_ivp(Y_prime_HT,[t_init,t_fin],Yi_vec,method='LSODA',rtol=1.e-6,atol=1.e-9)
        Yn_HT_f,Yp_HT_f = sol_at_HT.y[0][-1],sol_at_HT.y[1][-1]
    
    if(PRyMini.verbose_flag):
        print("--- running time: %s seconds ---" % (time.time() - start_time))
        print(" ")
    
    ########################
    # Import nuclear rates #
    ########################
    if(PRyMini.smallnet_flag):
        import PRyM.PRyM_nuclear_net12 as PRyMnuclear
        PRyMnucl = PRyMnuclear.UpdateNuclearRates(PRyMini.p_npdg,PRyMini.p_dpHe3g,PRyMini.p_ddHe3n,PRyMini.p_ddtp,PRyMini.p_tpag,PRyMini.p_tdan,PRyMini.p_taLi7g,PRyMini.p_He3ntp,PRyMini.p_He3dap,PRyMini.p_He3aBe7g,PRyMini.p_Be7nLi7p,PRyMini.p_Li7paa)
        if(PRyMini.julia_flag):
            pMLT = [lambda x: np.float64(PRyMnucl.npdg_frwrd(x)),lambda x: np.float64(PRyMnucl.npdg_bkwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_frwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_frwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_frwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_bkwrd(x)),lambda x: np.float64(PRyMnucl.tpag_frwrd(x)),lambda x: np.float64(PRyMnucl.tpag_bkwrd(x)),lambda x: np.float64(PRyMnucl.tdan_frwrd(x)),lambda x: np.float64(PRyMnucl.tdan_bkwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_frwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_frwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_frwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_frwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_bkwrd(x))]
    else:
        import PRyM.PRyM_nuclear_net63 as PRyMnuclear
        PRyMnucl = PRyMnuclear.UpdateNuclearRates(PRyMini.p_npdg,PRyMini.p_dpHe3g,PRyMini.p_ddHe3n,PRyMini.p_ddtp,PRyMini.p_tpag,PRyMini.p_tdan,PRyMini.p_taLi7g,PRyMini.p_He3ntp,PRyMini.p_He3dap,PRyMini.p_He3aBe7g,PRyMini.p_Be7nLi7p,PRyMini.p_Li7paa,PRyMini.p_Li7paag,PRyMini.p_Be7naa,PRyMini.p_Be7daap,PRyMini.p_daLi6g,PRyMini.p_Li6pBe7g,PRyMini.p_Li6pHe3a,PRyMini.p_B8naap,PRyMini.p_Li6He3aap,PRyMini.p_Li6taan,PRyMini.p_Li6tLi8p,PRyMini.p_Li7He3Li6a,PRyMini.p_Li8He3Li7a,PRyMini.p_Be7tLi6a,PRyMini.p_B8tBe7a,PRyMini.p_B8nLi6He3,PRyMini.p_B8nBe7d,PRyMini.p_Li6tLi7d,PRyMini.p_Li6He3Be7d,PRyMini.p_Li7He3aad,PRyMini.p_Li8He3aat,PRyMini.p_Be7taad,PRyMini.p_Be7tLi7He3,PRyMini.p_B8dBe7He3,PRyMini.p_B8taaHe3,PRyMini.p_Be7He3ppaa,PRyMini.p_ddag,PRyMini.p_He3He3app,PRyMini.p_Be7pB8g,PRyMini.p_Li7daan,PRyMini.p_dntg,PRyMini.p_ttann,PRyMini.p_He3nag,PRyMini.p_He3tad,PRyMini.p_He3tanp,PRyMini.p_Li7taan,PRyMini.p_Li7He3aanp,PRyMini.p_Li8dLi7t,PRyMini.p_Be7taanp,PRyMini.p_Be7He3aapp,PRyMini.p_Li6nta,PRyMini.p_He3tLi6g,PRyMini.p_anpLi6g,PRyMini.p_Li6nLi7g,PRyMini.p_Li6dLi7p,PRyMini.p_Li6dBe7n,PRyMini.p_Li7nLi8g,PRyMini.p_Li7dLi8p,PRyMini.p_Li8paan,PRyMini.p_annHe6g,PRyMini.p_ppndp,PRyMini.p_Li7taann)
        if(PRyMini.julia_flag):
            pMT = [lambda x: np.float64(PRyMnucl.npdg_frwrd(x)),lambda x: np.float64(PRyMnucl.npdg_bkwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_frwrd(x)),lambda x: np.float64(PRyMnucl.dpHe3g_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_frwrd(x)),lambda x: np.float64(PRyMnucl.ddHe3n_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_frwrd(x)),lambda x: np.float64(PRyMnucl.ddtp_bkwrd(x)),lambda x: np.float64(PRyMnucl.tpag_frwrd(x)),lambda x: np.float64(PRyMnucl.tpag_bkwrd(x)),lambda x: np.float64(PRyMnucl.tdan_frwrd(x)),lambda x: np.float64(PRyMnucl.tdan_bkwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_frwrd(x)),lambda x: np.float64(PRyMnucl.taLi7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_frwrd(x)),lambda x: np.float64(PRyMnucl.He3ntp_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_frwrd(x)),lambda x: np.float64(PRyMnucl.He3dap_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_frwrd(x)),lambda x: np.float64(PRyMnucl.He3aBe7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7nLi7p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7paa_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7paag_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7paag_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7naa_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7naa_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7daap_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7daap_bkwrd(x)),lambda x: np.float64(PRyMnucl.daLi6g_frwrd(x)),lambda x: np.float64(PRyMnucl.daLi6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6pBe7g_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6pBe7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6pHe3a_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6pHe3a_bkwrd(x))]
            pLT = pMT+[lambda x: np.float64(PRyMnucl.B8naap_frwrd(x)),lambda x: np.float64(PRyMnucl.B8naap_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3aap_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3aap_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6taan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6taan_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi8p_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi8p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3Li6a_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3Li6a_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3Li7a_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3Li7a_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi6a_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi6a_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8tBe7a_frwrd(x)),lambda x: np.float64(PRyMnucl.B8tBe7a_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8nLi6He3_frwrd(x)),lambda x: np.float64(PRyMnucl.B8nLi6He3_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8nBe7d_frwrd(x)),lambda x: np.float64(PRyMnucl.B8nBe7d_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi7d_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6tLi7d_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3Be7d_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6He3Be7d_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aad_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aad_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3aat_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8He3aat_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7taad_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7taad_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi7He3_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7tLi7He3_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8dBe7He3_frwrd(x)),lambda x: np.float64(PRyMnucl.B8dBe7He3_bkwrd(x)),lambda x: np.float64(PRyMnucl.B8taaHe3_frwrd(x)),lambda x: np.float64(PRyMnucl.B8taaHe3_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3ppaa_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3ppaa_bkwrd(x)),lambda x: np.float64(PRyMnucl.ddag_frwrd(x)),lambda x: np.float64(PRyMnucl.ddag_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3He3app_frwrd(x)),lambda x: np.float64(PRyMnucl.He3He3app_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7pB8g_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7pB8g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7daan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7daan_bkwrd(x)),lambda x: np.float64(PRyMnucl.dntg_frwrd(x)),lambda x: np.float64(PRyMnucl.dntg_bkwrd(x)),lambda x: np.float64(PRyMnucl.ttann_frwrd(x)),lambda x: np.float64(PRyMnucl.ttann_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3nag_frwrd(x)),lambda x: np.float64(PRyMnucl.He3nag_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3tad_frwrd(x)),lambda x: np.float64(PRyMnucl.He3tad_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3tanp_frwrd(x)),lambda x: np.float64(PRyMnucl.He3tanp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7taan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7taan_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aanp_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7He3aanp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8dLi7t_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8dLi7t_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7taanp_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7taanp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3aapp_frwrd(x)),lambda x: np.float64(PRyMnucl.Be7He3aapp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6nta_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6nta_bkwrd(x)),lambda x: np.float64(PRyMnucl.He3tLi6g_frwrd(x)),lambda x: np.float64(PRyMnucl.He3tLi6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.anpLi6g_frwrd(x)),lambda x: np.float64(PRyMnucl.anpLi6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6nLi7g_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6nLi7g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6dLi7p_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6dLi7p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li6dBe7n_frwrd(x)),lambda x: np.float64(PRyMnucl.Li6dBe7n_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7nLi8g_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7nLi8g_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7dLi8p_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7dLi8p_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li8paan_frwrd(x)),lambda x: np.float64(PRyMnucl.Li8paan_bkwrd(x)),lambda x: np.float64(PRyMnucl.annHe6g_frwrd(x)),lambda x: np.float64(PRyMnucl.annHe6g_bkwrd(x)),lambda x: np.float64(PRyMnucl.ppndp_frwrd(x)),lambda x: np.float64(PRyMnucl.ppndp_bkwrd(x)),lambda x: np.float64(PRyMnucl.Li7taann_frwrd(x)),lambda x: np.float64(PRyMnucl.Li7taann_bkwrd(x))]
    
    #################################################
    # Local thermal equilibrium for nuclear species #
    #################################################
    def YA(name,Yn,Yp,T):
        x = PRyMini.Nuclides[name]
        A = x[0]+x[1]
        Z = x[1]
        N = A-Z
        Mass = A*PRyMini.ma*PRyMini.MeV+PRyMini.keV*PRyMini.NuclExcessMass[name]-Z*PRyMini.me*PRyMini.MeV
        BindingE = N*PRyMini.NuclExcessMass["n"] + Z*PRyMini.NuclExcessMass["p"]-PRyMini.NuclExcessMass[name]
        NormYA = (Mass/((PRyMini.mn*PRyMini.MeV)**(A-Z)*(PRyMini.mp*PRyMini.MeV)**Z))**(3/2)
        return (2*PRyMini.NuclSpin[name]+1)*zeta(3)**(A-1)*np.pi**((1-A)/2)*2**((3*A-5)/2)*NormYA*(PRyMini.kB*T)**(3/2*(A-1))*eta_of_T(T)**(A-1)*Yp**Z*Yn**(A-Z) *np.exp(BindingE*PRyMini.keV/(PRyMini.kB*T))
    
    #########################################################
    # Nuclear network: Final yields for p,d,t,He3,a,Li7,Be7 #
    #########################################################
    if(PRyMini.smallnet_flag):
        def Y_prime(t,Y):
            rhoBBN = rhoB_BBN(a_of_t(t))
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            dY = PRyMnucl.dYndt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYpdt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYddt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYtdt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe3dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYadt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi7dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYBe7dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
            return dY
            
        def Jacobian(t,Y):
            rhoBBN = rhoB_BBN(a_of_t(t))
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            return PRyMnucl.Jacobian(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
    else:
        def Y_prime_MT(t,Y):
            rhoBBN = rhoB_BBN(a_of_t(t))
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            dY = PRyMnucl.dYndtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYpdtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYddtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYtdtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe3dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYadtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi7dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYBe7dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe6dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi8dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi6dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYB8dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
            return dY
            
        def Jacobian_MT(t,Y):
            rhoBBN = rhoB_BBN(a_of_t(t))
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            return PRyMnucl.JacobianMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
    
        def Y_prime_LT(t,Y):
            rhoBBN = rhoB_BBN(a_of_t(t))
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            dY = PRyMnucl.dYndtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYpdtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYddtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYtdtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe3dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYadtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi7dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYBe7dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYHe6dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi8dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYLi6dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PRyMnucl.dYB8dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
            return dY

        def Jacobian_LT(t,Y):
            rhoBBN = rhoB_BBN(a_of_t(t))
            T_t = T_of_t(t)*PRyMini.MeV_to_Kelvin # temperature in [K]
            return PRyMnucl.JacobianLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
    
    ############################
    # Mid temperature solution #
    ############################
    if(PRyMini.verbose_flag):
        print("Solving nuclear network at mid temperature era")
        
    # MT era definition
    t_init = t_weak
    t_fin = t_nucl
    
    # Weak rates at MT
    def nTOp_frwrd(T):
        return NormWeakRates*nTOp_frwrd_MT(T)
    def nTOp_bkwrd(T):
        return NormWeakRates*nTOp_bkwrd_MT(T)
    
    # Initial conditions at MT
    Yn_i = Yn_HT_f
    Yp_i = Yp_HT_f
    Yd_i = YA("d",Yn_i,Yp_i,PRyMini.T_weak)
    Yt_i = YA("t",Yn_i,Yp_i,PRyMini.T_weak)
    YHe3_i = YA("He3",Yn_i,Yp_i,PRyMini.T_weak)
    Ya_i = YA("a",Yn_i,Yp_i,PRyMini.T_weak)
    YLi7_i = YA("Li7",Yn_i,Yp_i,PRyMini.T_weak)
    YBe7_i = YA("Be7",Yn_i,Yp_i,PRyMini.T_weak)
    if(PRyMini.smallnet_flag == False):
        YHe6_i = YA("He6",Yn_i,Yp_i,PRyMini.T_weak)
        YLi8_i = YA("Li8",Yn_i,Yp_i,PRyMini.T_weak)
        YLi6_i = YA("Li6",Yn_i,Yp_i,PRyMini.T_weak)
        YB8_i = YA("B8",Yn_i,Yp_i,PRyMini.T_weak)
    
    # Solving MT network
    if(PRyMini.smallnet_flag):
        Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i]
        if(PRyMini.julia_flag):
            Y0 = np.float64(Yi_vec)
            tspan = (np.float64(t_init),np.float64(t_fin))
            p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_MT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_MT(x))] + pMLT
            f_Y_prime_MT_jl = de.ODEFunction(PRyMjl.Y_prime_MLT_jl,jac = PRyMjl.Jacobian_MLT_jl)
            prob = de.ODEProblem(f_Y_prime_MT_jl,Y0,tspan,p0,reltol=1.e-6,abstol=1.e-9)
            sol_at_MT = de.solve(prob,de.FBDF())
            sol_at_MT = np.array(sol_at_MT.u)
            Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f = sol_at_MT[-1,:]
        else:
            sol_at_MT = solve_ivp(Y_prime,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian,rtol=1.e-6,atol=1.e-9)
            Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f = sol_at_MT.y[0][-1],sol_at_MT.y[1][-1],sol_at_MT.y[2][-1],sol_at_MT.y[3][-1],sol_at_MT.y[4][-1],sol_at_MT.y[5][-1],sol_at_MT.y[6][-1],sol_at_MT.y[7][-1]
    else:
        Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i,YHe6_i,YLi8_i,YLi6_i,YB8_i]
        if(PRyMini.julia_flag):
            Y0 = np.float64(Yi_vec)
            tspan = (np.float64(t_init),np.float64(t_fin))
            p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_MT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_MT(x))] + pMT
            f_Y_prime_MT_jl = de.ODEFunction(PRyMjl.Y_prime_MT_jl,jac=PRyMjl.Jacobian_MT_jl)
            prob = de.ODEProblem(f_Y_prime_MT_jl,Y0,tspan,p0,reltol=1.e-6,abstol=1.e-9)
            sol_at_MT = de.solve(prob,de.FBDF())
            sol_at_MT = np.array(sol_at_MT.u)
            Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f,YHe6_MT_f,YLi8_MT_f,YLi6_MT_f,YB8_MT_f = sol_at_MT[-1,:]
        else:
            sol_at_MT = solve_ivp(Y_prime_MT,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian_MT,rtol=1.e-6,atol=1.e-9)
            Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f,YHe6_MT_f,YLi8_MT_f,YLi6_MT_f,YB8_MT_f = sol_at_MT.y[0][-1],sol_at_MT.y[1][-1],sol_at_MT.y[2][-1],sol_at_MT.y[3][-1],sol_at_MT.y[4][-1],sol_at_MT.y[5][-1],sol_at_MT.y[6][-1],sol_at_MT.y[7][-1],sol_at_MT.y[8][-1],sol_at_MT.y[9][-1],sol_at_MT.y[10][-1],sol_at_MT.y[11][-1]
    
    if(PRyMini.verbose_flag):
        print("--- running time: %s seconds ---" % (time.time() - start_time))
        print(" ")
    
    ############################
    # Low temperature solution #
    ############################
    if(PRyMini.verbose_flag):
        print("Solving nuclear network at low temperature era")
    
    # LT era definition
    t_init = t_nucl
    t_fin = t_end

    # Weak rates at LT
    def nTOp_frwrd(T):
        return NormWeakRates*nTOp_frwrd_LT(T)
    def nTOp_bkwrd(T):
        return NormWeakRates*nTOp_bkwrd_LT(T)
        
    # Initial conditions at LT
    Yn_i = Yn_MT_f
    Yp_i = Yp_MT_f
    Yd_i = Yd_MT_f
    Yt_i = Yt_MT_f
    YHe3_i = YHe3_MT_f
    Ya_i = Ya_MT_f
    YLi7_i = YLi7_MT_f
    YBe7_i = YBe7_MT_f
    if(PRyMini.smallnet_flag == False):
        YHe6_i = YHe6_MT_f
        YLi8_i = YLi8_MT_f
        YLi6_i = YLi6_MT_f
        YB8_i = YB8_MT_f
        
    if(PRyMini.smallnet_flag):
        Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i]
        if(PRyMini.julia_flag):
            Y0 = np.float64(Yi_vec)
            tspan = (np.float64(t_init),np.float64(t_fin))
            p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_LT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_LT(x))] + pMLT
            f_Y_prime_LT_jl = de.ODEFunction(PRyMjl.Y_prime_MLT_jl,jac = PRyMjl.Jacobian_MLT_jl)
            prob = de.ODEProblem(f_Y_prime_LT_jl,Y0,tspan,p0,abstol=1.e-13)
            sol_at_LT = de.solve(prob,de.CVODE_BDF())
            sol_at_LT = np.array(sol_at_LT.u)
            Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f = sol_at_LT[-1,:]
        else:
            sol_at_LT = solve_ivp(Y_prime,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian,atol=1.e-11)
            Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f = sol_at_LT.y[0][-1],sol_at_LT.y[1][-1],sol_at_LT.y[2][-1],sol_at_LT.y[3][-1],sol_at_LT.y[4][-1],sol_at_LT.y[5][-1],sol_at_LT.y[6][-1],sol_at_LT.y[7][-1]
    else:
        Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i,YHe6_i,YLi8_i,YLi6_i,YB8_i]
        if(PRyMini.julia_flag):
            Y0 = np.float64(Yi_vec)
            tspan = (np.float64(t_init),np.float64(t_fin))
            p0 = [lambda x: np.float64(T_of_t(x)*PRyMini.MeV_to_Kelvin),lambda x: np.float64(rhoB_BBN(a_of_t(x))),lambda x: np.float64(NormWeakRates*nTOp_frwrd_LT(x)),lambda x: np.float64(NormWeakRates*nTOp_bkwrd_LT(x))] + pLT
            f_Y_prime_LT_jl = de.ODEFunction(PRyMjl.Y_prime_LT_jl,jac=PRyMjl.Jacobian_LT_jl)
            prob = de.ODEProblem(f_Y_prime_LT_jl,Y0,tspan,p0,abstol=1.e-16)
            sol_at_LT = de.solve(prob,de.CVODE_BDF())
            sol_at_LT = np.array(sol_at_LT.u)
            Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f,YHe6_f,YLi8_f,YLi6_f,YB8_f = sol_at_LT[-1,:]
        else:
            sol_at_LT = solve_ivp(Y_prime_LT,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian_LT,atol=1.e-13)
            Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f,YHe6_f,YLi8_f,YLi6_f,YB8_f = sol_at_LT.y[0][-1],sol_at_LT.y[1][-1],sol_at_LT.y[2][-1],sol_at_LT.y[3][-1],sol_at_LT.y[4][-1],sol_at_LT.y[5][-1],sol_at_LT.y[6][-1],sol_at_LT.y[7][-1],sol_at_LT.y[8][-1],sol_at_LT.y[9][-1],sol_at_LT.y[10][-1],sol_at_LT.y[11][-1]

    if(PRyMini.verbose_flag):
        print("--- running time: %s seconds ---" % (time.time() - start_time))
        print(" ")

    if(PRyMini.verbose_flag):
        print("-------------------------------------------------")
        print("Predicted primordial abundances at the end of BBN")
        print("-------------------------------------------------")
        print("Yp = ",Yp_f)
        print("Yd = ",Yd_f)
        print("Yt = ",Yt_f)
        print("YHe3 = ",YHe3_f)
        print("Ya = ",Ya_f)
        print("YLi7 = ",YLi7_f)
        print("YBe7 = ",YBe7_f)
        print(" ")
        print("--- PRyMordial runned in: %s seconds ---" % (time.time() - start_time))
        
    #####################
    # Final predictions #
    #####################
    # N effective at the end of BBN era
    if(PRyMini.NP_thermo_flag):
        Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1],TNP_vec[-1])
    else:
        Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1])
    # Abundance of a single species of relativistic neutrino x 10^6
    Omeganurel_f = Omeganuh2_relnu()*1.e+6
    # Inverse of abundance of non-relativistic nu in units of sum of nu masses in [eV]
    OneOverOmeganunr_f = 1./(Omeganuh2_nrnu()*1.e-6)
    # Primordial helium-4 abundance as (nucleon) mass fraction (BBN definition)
    YP_f = 4.*Ya_f
    # Primordial helium-4 abundance as (baryon) mass fraction (CMB definition)
    YPCMB_f = (PRyMini.He4Overma/4.)*YP_f/((PRyMini.He4Overma/4.)*YP_f+PRyMini.HOverma*(1.-YP_f))
    # Primordial deuterium abundance as relative number density to hydrogen x 10^5
    DoHx1e5 = Yd_f/Yp_f*1e+5
    # Primordial helium-3 abundance as relative number density to hydrogen x 10^5
    He3oHx1e5 = (Yt_f+YHe3_f)/Yp_f*1e+5 # includes decay of tritium
    # Primordial lythium-7 abundance as relative number density to hydrogen x 10^10
    Li7oHx1e10 = (YLi7_f+YBe7_f)/Yp_f*1e+10 # includes decay of beryllium-7
    return np.array([Neff_f,Omeganurel_f,OneOverOmeganunr_f,YP_f,YPCMB_f,DoHx1e5,He3oHx1e5,Li7oHx1e10])
