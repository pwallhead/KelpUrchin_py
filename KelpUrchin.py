# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:00:09 2020

Functions for calculating and plotting Kelp-Urchin dynamics.

@author: Phil Wallhead, NIVA (pwa@niva.no)
         Magnus Norling, NIVA.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def int(S0=0, U0=[], ndt=1, ns=1, nagesU=15, area=200, dtS=np.arange(0,1.01,0.01), \
        fix_age_structureU=False, model_age_structureU=False, Unorm=[], UBnorm=[], UBtot0=[], agesel_bulk=[], \
        theta=[], kappa=[], parlogstd=0, parseed=[], \
        hparsUp=[], UBtotp=[], sigUBtotp=[], lognormal_UBtot_stoch=False, UBnormp=[], \
        htypeS='none', hparsS=[], hadaptationS='none', hSSgrid1=[], hSUgrid1=[], hSgrid=[], \
        htypeU='none', hparsU=[], hadaptationU='none', hUSgrid1=[], hUUgrid1=[], hUgrid=[], \
        hurryup_par=0, cut_losses_Sc=11, cut_losses_rate=0, \
        UTDM=[8.9,16.3,24.3,32.4,39.9,46.4,51.4,55.2,57.7,59.4,60.5,61.2,61.6,61.8,62.0], \
        bUIBM=[-7.086,2.818], sigUIBM=0.167, \
        bUIGM=[-5.037,1.472,0.266], sigUIGM=0.980, \
        ftype='loglinear', dT=0, dT_summer=0, dT_winter=0, dpCO2=0, \
        fix_eps_rU=False, seed=17, eps_rU=[], 
        fix_eps_So=False, eps_So=[], fix_eps_UBtoto=False, eps_UBtoto=[]):

    """
    Integrate stochastic difference/differential equations defining Kelp-Urchin dynamics.
    Kelp is modelled as a bulk variable while the urchin population can be bulk or age-structured.
    The approach loosely follows Marzloff et al., 2013.
    
    Basic Inputs:  
                   S0 = initial kelp biomass density [kgww/m2] (scalar or 1*ns numpy array)
                   U0 = initial urchin number density for each age class [/m2] (nagesU vector or nagesU*ns numpy array, empty for urchin barren default)
                  ndt = number of annual time steps to simulate
                   ns = number of simulations (ensemble members)
                   
    Model parameter inputs:
                theta = dictionary of dynamical model parameter names and input values, e.g.:
                        theta = {'rS':20, 'alphaS':0.8} to set parameters 'rS' and 'alphaS'
                        All unset parameters will use internal default values
                kappa = dictionary of auxiliary parameter names and input values
    
    Outputs:        S = kelp biomass density (1 x ns) [kgww/m2] where ns = no. simulations
                    U = urchin number density for each age class [/m2] (nagesU*ns numpy array)
                  out = dictionary containing various other outputs
    
    Uses: intKelp
            anintLogistic
    
    
    Example 1: Run for 1 year with S0=4 kgww/m2 and default urchin barren initial condition:
    import KelpUrchin
    import numpy as np
    S,U,out = KelpUrchin.int(4,[],1)
    S #Should give array([[1.15593335]]) - note this is not affected by urchin recruitment stochasticity
    U #Note that U[0,0] varies between repeat runs due to recruitment stochasticity
      #This can be switched off, resulting in a deterministic system, by setting the stochasticity standard deviation to zero 
      #(S,U,out = KelpUrchin.int(4,[],1,theta={'rUsig':0}))
    
    
    Example 2: Run a deterministic simulation for 20 years from an initial condition of moderate kelp and urchin biomass.
               Compare the results of using: 1) variable urchin age structure (default, 16D system)
                                             2) fixed urchin age structure (fix_age_structureU = True, 2D system) 
    S0 = 5; U0 = 5*0.71**np.arange(15)
    S1,U1,out1 = KelpUrchin.int(S0,U0,20,theta={'rUsig':0})
    S2,U2,out2 = KelpUrchin.int(S0,U0,20,theta={'rUsig':0},fix_age_structureU=True)
    #Compare the kelp biomass densities and total urchin biomass densities:
    np.column_stack((out1['St'],out2['St'],out1['UBtott'],out2['UBtott']))
    #The final urchin biomass density is higher with the fixed age structure, because the assumed
    #age structure (by default taken from U0) does not account for predation mortality associated
    #with the kelp coverage.


    Example 3: Run ns=1000 stochastic simulations for 20 years from an urchin barren initial condition.
               Compare the results of using: 1) variable urchin age structure (default, 16D system)
                                             2) fixed urchin age structure (fix_age_structureU = True, 2D system) 
               Time the function calls to assess speed.
    import time
    S0 = 0; U0 = 12.5*0.71**np.arange(15)
    ns = 1000
    t0=time.time(); S1,U1,out1 = KelpUrchin.int(S0,U0,20,ns=ns); print(time.time()-t0)
    t0=time.time(); S2,U2,out2 = KelpUrchin.int(S0,U0,20,ns=ns,fix_age_structureU=True); print(time.time()-t0)

    #Check annual statistics and final probability distributions of total urchin biomass:
    import matplotlib.pyplot as plt
    print(np.column_stack((np.mean(out1['UBtott'],1),np.mean(out2['UBtott'],1),np.std(out1['UBtott'],1),np.std(out2['UBtott'],1))))
    x=0.1*np.arange(21);plt.hist(out1['UBtott'][-1,:],bins=x,alpha=0.5);plt.hist(out2['UBtott'][-1,:],bins=x,alpha=0.5);plt.show()
    #Agreement is good here, showing that the fixed age structure approximation is a good one in this case.
    #Note: Run speed here appears to be ~10 times slower than Matlab
    
    
    Example 4: Run ns=1000 stochastic simulations for 20 years from an urchin barren initial condition with fixed fraction harvest/cull.
               Repeat for harvested fractions (0,0.1,...,1) to evaluate kelp recovery probability for a given harvesting effort.
               Compare the results of using: 1) variable urchin age structure (default, 16D system)
                                             2) fixed urchin age structure (fix_age_structureU = True, 2D system) 

    S0 = 0; U0 = 12.5*0.71**np.arange(15)
    hfracv = 0.1*np.arange(11); nexps = len(hfracv)
    pr1 = np.nan*np.ones(nexps)
    pr2 = np.nan*np.ones(nexps)
    meanU1 = np.nan*np.ones([15,nexps])
    meanUBtot1 = np.nan*np.ones(nexps)
    stdUBtot1 = np.nan*np.ones(nexps)
    stdtUBtot1 = np.nan*np.ones(nexps)
    stdlogUBtot1 = np.nan*np.ones(nexps)
    meanUBnorm1 = np.nan*np.ones([15,nexps])
    meanUBtot2 = np.nan*np.ones(nexps)
    stdUBtot2 = np.nan*np.ones(nexps)
    stdtUBtot2 = np.nan*np.ones(nexps)
    ns = 1000
    for i in range(nexps):
        S1,U1,out1 = KelpUrchin.int(S0,U0,20,ns=ns,htypeU='fixed fraction',hparsU=hfracv[i])
        pr1[i] = np.sum(out1['St'][-1,:]>=5.5)/ns #Recovery defined as final kelp biomass >= 0.5*carrying capacity
        meanU1[:,i] = out1['meanUt'][:,-1]
        meanUBtot1[i] = np.mean(np.mean(out1['UBtott'],1))
        stdUBtot1[i] = np.mean(np.std(out1['UBtott'],1))
        stdtUBtot1[i] = np.mean(np.std(out1['UBtott'],0))
        stdlogUBtot1[i] = np.mean(np.std(np.log(out1['UBtott']),1))
        meanUBnorm1[:,i] = np.mean(out1['meanUBt']/np.mean(out1['UBtott'],1).transpose(),1)
        
        S2,U2,out2 = KelpUrchin.int(S0,U0,20,ns=ns,htypeU='fixed fraction',hparsU=hfracv[i],fix_age_structureU=True)
        pr2[i] = np.sum(out2['St'][-1,:]>=5.5)/ns
        meanUBtot2[i] = np.mean(np.mean(out2['UBtott'],1),0)
        stdUBtot2[i] = np.mean(np.std(out2['UBtott'],1))
        stdtUBtot2[i] = np.mean(np.std(out2['UBtott'],0))
    
    #The transition is somewhat more abrupt with variable age structure, but agreement is reasonable, see: 
    np.column_stack((hfracv,pr1,pr2))
    
    #Sort and plot the simulated relationsips U~UBtot etc.
    Isort = np.argsort(meanUBtot1)
    meanUBtot1s = meanUBtot1[Isort]
    meanU1s = meanU1[:,Isort]
    stdUBtot1s = stdUBtot1[Isort]
    stdlogUBtot1s = stdlogUBtot1[Isort]
    meanUBnorm1s = meanUBnorm1[:,Isort]
#    plt.figure(1); plt.clf()
#    for i in range(15):
#        plt.plot(meanUBtot1s,meanU1s[i,:],'o-')
#    plt.rc('axes', labelsize=40)
#    plt.figure(2); plt.clf()
#    plt.plot(meanUBtot1s,stdUBtot1s,'o-')
#    plt.rc('axes', labelsize=40)
    
    #Alternative dimensional-reduction approach using simulated relationships (U~UBtot, stdUBtot~UBtot)
    pr3 = np.nan*np.ones(nexps)
    meanUBtot3 = np.nan*np.ones(nexps)
    stdUBtot3 = np.nan*np.ones(nexps)
    stdtUBtot3 = np.nan*np.ones(nexps)    
    ns = 1000
    fudge_factor = 0.7 #0.7 seems to work quite well
    for i in range(nexps):
        S3,U3,out3 = KelpUrchin.int(S0,U0,20,ns=ns,htypeU='fixed fraction',hparsU=hfracv[i],
                       fix_age_structureU=True,model_age_structureU=True,UBtotp=meanUBtot1s,sigUBtotp=fudge_factor*stdUBtot1s,UBnormp=meanUBnorm1s)
        pr3[i] = np.sum(out3['St'][-1,:]>=5.5)/ns
        meanUBtot3[i] = np.mean(np.mean(out3['UBtott'],1),0)
        stdUBtot3[i] = np.mean(np.std(out3['UBtott'],1),0)
        stdtUBtot3[i] = np.mean(np.std(out3['UBtott'],0))
        
    np.column_stack((hfracv,pr1,pr2,pr3)) #Results are quite similar for both approaches
    lfontsize = 25
    fontsize = 25
    legfontsize = 25
    plt.figure(1); plt.clf()
    plt.subplot(2,2,1)
    plt.plot(hfracv,pr1,'ko-')
    plt.plot(hfracv,pr2,'bo-')
    plt.plot(hfracv,pr3,'ro-')
    plt.ylabel('Probability of kelp recovery', fontsize=lfontsize)
    plt.xlabel('Harvested fraction', fontsize=lfontsize)
    plt.legend(['16D system','2D system (1)','2D system (2)'],fontsize=legfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.subplot(2,2,2)
    plt.plot(hfracv,meanUBtot1,'ko-')
    plt.plot(hfracv,meanUBtot2,'bo-')
    plt.plot(hfracv,meanUBtot3,'ro-')
    plt.ylabel('Time-av. ensemble-mean urchin biomass [kg/m2]', fontsize=lfontsize)
    plt.xlabel('Harvested fraction', fontsize=lfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.subplot(2,2,3)
    plt.plot(hfracv,stdUBtot1,'ko-')
    plt.plot(hfracv,stdUBtot2,'bo-')
    plt.plot(hfracv,stdUBtot3,'ro-')
    plt.ylabel('Time-av. ensemble-st.dev. urchin biomass [kg/m2]', fontsize=lfontsize)
    plt.xlabel('Harvested fraction', fontsize=lfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.subplot(2,2,4)
    plt.plot(hfracv,stdtUBtot1,'ko-')
    plt.plot(hfracv,stdtUBtot2,'bo-')
    plt.plot(hfracv,stdtUBtot3,'ro-')
    plt.ylabel('Ensemble-mean time-st.dev. urchin biomass [kg/m2]', fontsize=lfontsize)
    plt.xlabel('Harvested fraction', fontsize=lfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels

    
    
    Example 4a: Run ns=1000 stochastic simulations for 20 years from an urchin barren initial condition with culling by lime treatment.
                Repeat for 0:12 treatments/yr to evaluate kelp recovery probability for a given culling effort.
                Compare the results of using: 1) variable urchin age structure (default, 16D system)
                                              2) fixed urchin age structure (fix_age_structureU = True, 2D system) 

    S0 = 0; U0 = 12.5*0.71**np.arange(15)
    ntreatmentsv = np.arange(13); nexps = len(ntreatmentsv)
    pr1 = np.nan*np.ones(nexps)
    pr2 = np.nan*np.ones(nexps)
    meanRU1 = np.nan*np.ones([15,nexps])
    meanUBnorm1 = np.nan*np.ones([15,nexps])
    meanU1 = np.nan*np.ones([15,nexps])
    meanUBtot1 = np.nan*np.ones(nexps)
    stdUBtot1 = np.nan*np.ones(nexps)
    stdtUBtot1 = np.nan*np.ones(nexps)
    stdlogUBtot1 = np.nan*np.ones(nexps)
    rUsig = 0.82
    ns = 1000
    for i in range(nexps):
        S1,U1,out1 = KelpUrchin.int(S0,U0,20,theta={'rUsig':rUsig},ns=ns,htypeU='lime treatment',hparsU=ntreatmentsv[i])
        pr1[i] = np.sum(out1['St'][-1,:]>=5.5)/ns #Recovery defined as final kelp biomass >= 0.5*carrying capacity
#        meanRU1[:,i] = np.mean(U1/out1['UBtott'][-1,:],1)
#        meanUBnorm1[:,i] = np.mean(out1['UB']/out1['UBtott'][-1,:],1)
        meanU1[:,i] = np.mean(out1['meanUt'],1)
        meanUBtot1[i] = np.mean(np.mean(out1['UBtott'],1),0)
        stdUBtot1[i] = np.mean(np.std(out1['UBtott'],1),0)
        stdtUBtot1[i] = np.mean(np.std(out1['UBtott'],0))
        stdlogUBtot1[i] = np.mean(np.std(np.log(out1['UBtott']),1),0)
        meanUBnorm1[:,i] = np.mean(out1['meanUBt']/np.mean(out1['UBtott'],1).transpose(),1)
        
    #Sort and plot the simulated relationsips U~UBtot etc.
    Isort = np.argsort(meanUBtot1)
    meanUBtot1s = meanUBtot1[Isort]
    meanU1s = meanU1[:,Isort]
    stdUBtot1s = stdUBtot1[Isort]
    stdlogUBtot1s = stdlogUBtot1[Isort]
    meanUBnorm1s = meanUBnorm1[:,Isort]
    plt.figure(1); plt.clf()
    for i in range(15):
        plt.plot(meanUBtot1s,meanU1s[i,:],'o-')
    plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    plt.figure(2); plt.clf()
    plt.subplot(2,2,1)
    plt.plot(meanUBtot1s,stdUBtot1s,'o-')
    plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    plt.subplot(2,2,2)
    plt.plot(meanUBtot1s,stdlogUBtot1s,'o-')
    plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    
    pr2 = np.nan*np.ones(nexps)
    meanUBtot2 = np.nan*np.ones(nexps)
    stdUBtot2 = np.nan*np.ones(nexps)
    stdtUBtot2 = np.nan*np.ones(nexps)
    for i in range(nexps):    
        S2,U2,out2 = KelpUrchin.int(S0,U0,20,theta={'rUsig':rUsig},ns=ns,htypeU='lime treatment',hparsU=ntreatmentsv[i],
                       fix_age_structureU=True)
        pr2[i] = np.sum(out2['St'][-1,:]>=5.5)/ns
        meanUBtot2[i] = np.mean(np.mean(out2['UBtott'],1),0)
        stdUBtot2[i] = np.mean(np.std(out2['UBtott'],1))
        stdtUBtot2[i] = np.mean(np.std(out2['UBtott'],0))
    
    #Kelp regrowth is a bit harder to achieve with variable age structure, but agreement is reasonable, see: 
    np.column_stack((ntreatmentsv,pr1,pr2))
    np.column_stack((np.std(out1['UBtott'],1),np.std(out2['UBtott'],1)))
    #Probably the reduced system is lacking some stochasticity since the age structure is deterministically dependent on UBtot
    
    #Alternative approach using the harvesting effort parameter
    pr3 = np.nan*np.ones(nexps)
    meanUBtot3 = np.nan*np.ones(nexps)
    stdUBtot3 = np.nan*np.ones(nexps)
    stdtUBtot3 = np.nan*np.ones(nexps)
    fudge_factor = 0.5 #0.5 seems to work quite well
    for i in range(nexps):    
        S3,U3,out3 = KelpUrchin.int(S0,U0,20,theta={'rUsig':rUsig},ns=ns,htypeU='lime treatment',hparsU=ntreatmentsv[i],
                       fix_age_structureU=True,model_age_structureU=True,hparsUp=ntreatmentsv,sigUBtotp=fudge_factor*stdUBtot1,UBnormp=meanUBnorm1)
        pr3[i] = np.sum(out3['St'][-1,:]>=5.5)/ns
        meanUBtot3[i] = np.mean(np.mean(out3['UBtott'],1),0)
        stdUBtot3[i] = np.mean(np.std(out3['UBtott'],1),0)
        stdtUBtot3[i] = np.mean(np.std(out3['UBtott'],0))
    
    #Alternative approach using the total extant biomass UBtot
    pr4 = np.nan*np.ones(nexps)
    meanUBtot4 = np.nan*np.ones(nexps)
    stdUBtot4 = np.nan*np.ones(nexps)
    stdtUBtot4 = np.nan*np.ones(nexps)
    fudge_factor = 0.7 #0.7 seems to work quite well
    for i in range(nexps):    
        S4,U4,out4 = KelpUrchin.int(S0,U0,20,theta={'rUsig':rUsig},ns=ns,htypeU='lime treatment',hparsU=ntreatmentsv[i],
                       fix_age_structureU=True,model_age_structureU=True,UBtotp=meanUBtot1s,sigUBtotp=fudge_factor*stdUBtot1s,UBnormp=meanUBnorm1s)
        pr4[i] = np.sum(out4['St'][-1,:]>=5.5)/ns
        meanUBtot4[i] = np.mean(np.mean(out4['UBtott'],1),0)
        stdUBtot4[i] = np.mean(np.std(out4['UBtott'],1))
        stdtUBtot4[i] = np.mean(np.std(out4['UBtott'],0))

    #Alternative approach using the total extant biomass UBtot and lognormal stochasticity
    pr5 = np.nan*np.ones(nexps)
    meanUBtot5 = np.nan*np.ones(nexps)
    stdUBtot5 = np.nan*np.ones(nexps)
    stdtUBtot5 = np.nan*np.ones(nexps)
    fudge_factor = 0.35 #0.35 seems to work quite well
    for i in range(nexps):    
        S5,U5,out5 = KelpUrchin.int(S0,U0,20,theta={'rUsig':rUsig},ns=ns,htypeU='lime treatment',hparsU=ntreatmentsv[i],
                       fix_age_structureU=True,model_age_structureU=True,UBtotp=meanUBtot1s,sigUBtotp=fudge_factor*stdlogUBtot1s,lognormal_UBtot_stoch=True,UBnormp=meanUBnorm1s)
        pr5[i] = np.sum(out5['St'][-1,:]>=5.5)/ns
        meanUBtot5[i] = np.mean(np.mean(out5['UBtott'],1),0)
        stdUBtot5[i] = np.mean(np.std(out5['UBtott'],1))
        stdtUBtot5[i] = np.mean(np.std(out5['UBtott'],0))
    
    #Compare the outputs
    lfontsize = 20
    fontsize = 20
    legfontsize = 20
    plt.figure(3); plt.clf()
    plt.subplot(2,2,1)
    plt.plot(ntreatmentsv,pr1,'ko-')
    plt.plot(ntreatmentsv,pr2,'bo-')
    plt.plot(ntreatmentsv,pr3,'go-')
    plt.plot(ntreatmentsv,pr4,'ro-')
    plt.plot(ntreatmentsv,pr5,'co-')
    plt.ylabel('Probability of kelp recovery', fontsize=lfontsize)
    plt.xlabel('No. lime treatments per year', fontsize=lfontsize)
    plt.legend(['16D system','2D system (1)','2D system (2)','2D system (3)','2D system (4)'],fontsize=legfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.subplot(2,2,2)
    plt.plot(ntreatmentsv,meanUBtot1,'ko-')
    plt.plot(ntreatmentsv,meanUBtot2,'bo-')
    plt.plot(ntreatmentsv,meanUBtot3,'go-')
    plt.plot(ntreatmentsv,meanUBtot4,'ro-')
    plt.plot(ntreatmentsv,meanUBtot5,'co-')
    plt.ylabel('Time-av. ensemble-mean urchin biomass [kg/m2]', fontsize=lfontsize)
    plt.xlabel('No. lime treatments per year', fontsize=lfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.subplot(2,2,3)
    plt.plot(ntreatmentsv,stdUBtot1,'ko-')
    plt.plot(ntreatmentsv,stdUBtot2,'bo-')
    plt.plot(ntreatmentsv,stdUBtot3,'go-')
    plt.plot(ntreatmentsv,stdUBtot4,'ro-')
    plt.plot(ntreatmentsv,stdUBtot5,'co-')
    plt.ylabel('Time-av. ensemble-st.dev. urchin biomass [kg/m2]', fontsize=lfontsize)
    plt.xlabel('No. lime treatments per year', fontsize=lfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.subplot(2,2,4)
    plt.plot(ntreatmentsv,stdtUBtot1,'ko-')
    plt.plot(ntreatmentsv,stdtUBtot2,'bo-')
    plt.plot(ntreatmentsv,stdtUBtot3,'go-')
    plt.plot(ntreatmentsv,stdtUBtot4,'ro-')
    plt.plot(ntreatmentsv,stdtUBtot5,'co-')
    plt.ylabel('Ensemble-mean time-st.dev. urchin biomass [kg/m2]', fontsize=lfontsize)
    plt.xlabel('No. lime treatments per year', fontsize=lfontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    
    Conclusion: The reduction method interpolating UBtot with normal stochasticity in UBtot (4)
                seems to work best overall (with fudge_factor=0.7).
    
    
    Example 5: Run for 1 year using 99 different fixed values of the stochastic variable eps_rU, using quantiles of the normal distribution.
               Use these simulations to approximate the expected final values of (S,U). 
               Compare the results of using: 1) using the ensemble to perform array calculations (ns=99)
                                             2) calling the function in a loop 
               Time both approaches to assess speed.
    import time
    from scipy.stats import norm
    S0 = 4; U0 = 12.5*0.71**np.arange(15)
    q1 = norm.ppf(0.01*np.arange(1,100)) #Evenly spaced quantiles of normal distribution (1%,2%,...,99%)
    ns = len(q1)
    
    #Here is the correct approach using fast array computations
    t0=time.time()
    S1,U1,out1 = KelpUrchin.int(S0,U0,1,ns=ns,fix_eps_rU=True,eps_rU=q1)
    ES1 = np.mean(S1,1); EU1 = np.mean(U1,1)
    print(time.time()-t0) #0.001 s on my PC (Intel core i7, 2x2.6 GHz, 32GB RAM)
    
    #Here is the incorrect approach using a loop
    S2 = np.nan*np.ones([1,ns])
    U2 = np.nan*np.ones([15,ns])
    t0=time.time()
    for i in range(ns):
        S21,U21,out = KelpUrchin.int(S0,U0,1,fix_eps_rU=True,eps_rU=q1[i])
        S2[0,i] = S21[0,0]; U2[:,i] = U21[:,0]
    ES2 = np.mean(S2,1); EU2 = np.mean(U2,1)
    print(time.time()-t0) #0.096 s on my PC (Intel core i7, 2x2.6 GHz, 32GB RAM)
    
    #Check the results are the same:
    print(np.column_stack((ES1,ES2)))
    print(np.column_stack((EU1,EU2)))
    #Conclusion: The results are the same, but the loop approach (2) is about 100 times slower
    
    
    Example 6: Run the reduced 2D system (fix_age_structureU=True) for 1 year using 99 different fixed values 
               of the stochastic variable eps_rU, using quantiles of the normal distribution.
               Use these simulations to approximate the expected final values of (S,UBtot). 
    
    from scipy.stats import norm
    q1 = norm.ppf(0.01*np.arange(1,100)) #Evenly spaced quantiles of normal distribution (1%,2%,...,99%)
    ns = len(q1)
    S0 = 4; UBtot0 = 0.5
    S,U,out = KelpUrchin.int(S0=S0,UBtot0=UBtot0,ndt=1,ns=ns,fix_eps_rU=True,eps_rU=q1,fix_age_structureU=True)
    UBtot = out['UBtott'][-1,:]
    ES = np.mean(S) #Should be 3.3638
    EUBtot = np.mean(UBtot) #Should be 0.3842
    
    #Phil Wallhead, Magnus Norling 05/08/2020
    """

    #I1 = np.mat(np.ones(ns)) #Useful vector
    I1 = np.ones(ns) #Useful vector
    out = dict() #Output dictionary


    #Prepare environmental change arrays e.g. dT (ndt*ns)
    dT = prep_array(dT,ns,ndt)
    dT_summer = prep_array(dT_summer,ns,ndt)  
    dT_winter = prep_array(dT_winter,ns,ndt)
    dpCO2 = prep_array(dpCO2,ns,ndt)


    #Set values of full dynamical model parameter dictionary (thetaf)
    thetaf = {'rS':25*I1, 'alphaS':1.0*I1, 'KS':11*I1, 'gUS':7.3*I1, \
          'TD_halfgUS':20*I1, 'dTD_halfgUS':5*I1, \
          'rUmean':12.5*I1, 'rUsig':0.82*I1, 'frU_S':0.025*I1, \
          'pU':0.71*I1, 'fpU_S':0.72*I1, 'TD_halfpUmortp':50*I1, \
          'dTD_halfpUmortp':5*I1, 'dfGI_summer':0.17*I1, 'dfGI_winter':0.047*I1}
    #Default parameter values from Wallhead et al. (2018), see below for parameter meanings.

    if (ftype=='loglinear'): #Default set using log-linear sensitivity functions
        thetaf.update({'falphaS_T':1.068*I1, 'falphaS_pCO2':1.014*I1,
                       'frU_T':0.20*I1, 'frU_pCO2':0.72*I1, 'fpU_T':0.95*I1})
    if (ftype=='linear'): #Default set using linear sensitivity functions
        thetaf.update({'falphaS_T':0.081*I1, 'falphaS_pCO2':0.034*I1, \
                       'frU_T':-0.54*I1, 'fEU_pCO2':-0.0042*I1, 'fpFU_pCO2':-0.013*I1, \
                       'fpLU_pCO2':-0.029*I1, 'fpJU_pCO2':-0.112*I1, 'fpU_T':-0.045*I1})    
    #Default parameter values from Wallhead et al. (2018), see below for parameter meanings.
    #Note that the urchin (recruitment, mortality) sensitivities to kelp coverage (frU_S, fpU_S)
    #are always loglinear irrespective of input ftype

    if (htypeU=='lime treatment'):
        thetaf.update({'min_pUlime1':0.6*I1})   #Minimum urchin survival rate from 1 lime treatment, applicable to older urchins [] 
        #Maximum of 80% mortality in adults observed from 3 treatments (Fagerli pers. comm.) => min_pUlime1 = 0.2**(1/3) ~ 0.6
        thetaf.update({'TD_halfpUlime1':20*I1}) #Test diameter for which urchin survival due to liming decreases by half the maximum decrease [mm]
        thetaf.update({'dTD_halfpUlime1':10*I1})#Change in urchin test diameter for urchin survival due to liming to decrease by half the maximum decrease [mm]
        #0% mortality observed in age classes 0-2 years, maximum mortality in adults age 4+ (Fagerli pers. comm.) => TD_half = 0.5*(15+45) ~ 30 mm, dTD = 15 mm
        #Note we assume that the liming mortality is size-dependent (smaller => more chance of shelter/refuge)

    if (hadaptationS!='none' or hadaptationU!='none'):
        thetaf.update({'Sosig':0.3*I1})         #Logarithmin std in kelp biomass observations (30% error seems reasonable default)
        thetaf.update({'UBtotosig':0.3*I1})     #Logarithmin std in urchin biomass observations (30% error seems reasonable default)

    #Update thetaf with input dictionary theta
    theta = prep_array(theta,ns)
    thetaf.update(theta)
    #Add simple parameter uncertainty if req'd
    if parlogstd>0:
        if parseed:
            np.random.seed(parseed)
        for str1 in thetaf:
            eps1 = np.random.randn(1,ns) 
            thetaf[str1] = thetaf[str1]*np.exp(parlogstd*eps1 - 0.5*parlogstd**2)
    out.update({'thetaf':thetaf})


    #Set values of full auxiliary model parameter dictionary (kappaf)
    kappaf = {'SA_HK':0.5*I1, 'Utot_HK':24.7*I1, 'Utot_HB':43*I1, \
              'Utot_VB':11.1*I1, 'dT_HV':1.76*I1, 'SA_I':0.5*I1}
    kappa = prep_array(kappa,ns)
    kappaf.update(kappa) #Update with input dictionary kappa
    #Add simple parameter uncertainty if req'd
    if parlogstd>0:
        for str1 in kappaf:
            eps1 = np.random.randn(1,ns) 
            kappaf[str1] = kappaf[str1]*np.exp(parlogstd*eps1 - 0.5*parlogstd**2)
    out.update({'kappaf':kappaf}) 


    #Standardize array shapes to facilitate fast calculations
    UTDM = prep_array(UTDM,ns)
    bUIBM = prep_array(bUIBM,ns)
    sigUIBM = prep_array(sigUIBM,ns)
    bUIGM = prep_array(bUIGM,ns)
    sigUIGM = prep_array(sigUIGM,ns)    
    if (fix_eps_rU==True):
        eps_rU = prep_array(eps_rU,ns,ndt)
    if (htypeU!='none'):
        htypeU_coded = ['fixed fraction','lime treatment'] #These are all the allowable types -- append this list if a new harvesting type is added
        if (htypeU not in htypeU_coded):
            exit("Unknown/uncoded urchin harvest type htypeU")
        if (hadaptationU!='none'):
            hparsU = np.nan*np.ones([ndt,ns])
        else:
            hparsU = prep_array(hparsU,ns,ndt)        
    out.update({'UTDM':UTDM, 'bUIBM':bUIBM, 'sigUIBM':sigUIBM, 'bUIGM':bUIGM, 'sigUIGM':sigUIGM})


    #Assign named parameters from thetaf, converting to internal model units and imposing hard limits.
    if (ftype=='loglinear'):
        fmin = 0
    elif (ftype=='linear'):
        fmin = -np.inf

    #Kelp parameters
    rS = np.maximum(0, thetaf['rS'])/1e3                   #Kelp recruitment flux [kgww/m2/year]
    alphaS = np.maximum(0, thetaf['alphaS'])               #Kelp intrinsic population growth rate [year^-1]   
    falphaS_T = np.maximum(fmin, thetaf['falphaS_T'])      #Adjustment factor for alphaS per 1 degC warming []
    falphaS_pCO2 = np.maximum(fmin, thetaf['falphaS_pCO2'])#Adjustment factor for alphaS per 100 uatm pCO2 increase []
    KS = np.maximum(0, thetaf['KS'])                       #Kelp carrying capacity [kgww/m2]
    gUS = np.maximum(0, thetaf['gUS'])                     #Urchin grazing rate as fraction of individual body mass [dt^-1]
    TD_halfgUS = np.maximum(0, thetaf['TD_halfgUS'])       #Test diameter for which urchin grazing rate increases linearly to half its maximum value 
    dTD_halfgUS = np.maximum(0, thetaf['dTD_halfgUS'])     #Change in urchin test diameter for gUS to increase linearly to half maximum value

    #Urchin parameters
    rUmean = np.maximum(0, thetaf['rUmean'])               #Urchin mean recruitment flux [individuals/m2/year]
    rUsig = np.maximum(0, thetaf['rUsig'])                 #Logarithmic standard deviation in urchin recruitment flux between time steps [-]
    frU_S = np.maximum(0, thetaf['frU_S'])                 #Adjustment factor for rU at maximum kelp cover []
    frU_T = np.maximum(fmin, thetaf['frU_T'])              #Adjustment factor for rU per 1 degC warming []
    if (ftype=='loglinear'):
        frU_pCO2 = np.maximum(fmin, thetaf['frU_pCO2'])    #Adjustment factor for rU per 100 uatm pCO2 increase [] 
    elif (ftype=='linear'):
        #With a linear sensitivity model, the various stage sensitivities cannot be combined 
        #into a single factor frU_pCO2. Rather they are separately resolved as follows:
        fEU_pCO2 = thetaf['fEU_pCO2']                      #Linear sensitivity parameter for urchin egg production
        fpFU_pCO2 = thetaf['fpFU_pCO2']                    #Linear sensitivity parameter for urchin egg fertilization probability
        fpLU_pCO2 = thetaf['fpLU_pCO2']                    #Linear sensitivity parameter for urchin larvae survival probability
        fpJU_pCO2 = thetaf['fpJU_pCO2']                    #Linear sensitivity parameter for urchin juvenile survival probability
    pU = np.minimum(1, np.maximum(0, thetaf['pU']))        #Urchin survival probability = (1 - probability of dying between time steps) [-]
    fpU_S = np.maximum(0, thetaf['fpU_S'])                 #Adjustment factor for pU due to kelp-associated predation at maximum kelp cover []
    fpU_T = np.maximum(fmin, thetaf['fpU_T'])              #Adjustment factor for pU per 1 degC warming []
    TD_halfpUmortp = np.maximum(0, thetaf['TD_halfpUmortp'])   #Test diameter for which urchin predation probability drops linearly to half its maximum value [mm] 
    dTD_halfpUmortp = np.maximum(0, thetaf['dTD_halfpUmortp']) #Change in urchin test diameter for predation probability to drop linearly to half maximum value [mm]
    dfGI_summer = thetaf['dfGI_summer']                    #Fractional increase in urchin gonad index due to July-August warming [] 
    dfGI_winter = thetaf['dfGI_winter']                    #Fractional increase in urchin gonad index due to December-January warming []
    if (htypeU=='lime treatment'):
        min_pUlime1 = thetaf['min_pUlime1']                #Minimum survival rate from 1 lime treatment, applicable to older urchins []
        TD_halfpUlime1 = thetaf['TD_halfpUlime1']          #Test diameter for which urchin survival due to liming decreases by half the maximum decrease [mm]
        dTD_halfpUlime1 = thetaf['dTD_halfpUlime1']        #Change in test diameter for urchin survival due to liming to decrease to half the maximum decrease [mm]
    if (hadaptationS!='none' or hadaptationU!='none'):
        Sosig = thetaf['Sosig']                            #Logarithmic standard deviation in kelp biomass observational errors []
        UBtotosig = thetaf['UBtotosig']                    #Logarithmic standard deviation in urchin biomass observational errors []        


    #Auxiliary parameters
    #SA_HK = np.minimum(1, np.maximum(0, kappaf['SA_HK']))  #Assumed level of kelp cover in the Hammerfest kelp sites (0-1) []
    #dT_HV = kappaf['dT_HV']                                #Assumed present-day temperature difference between Hammerfest and Vega [degC]
    SA_I = np.minimum(1, np.maximum(0, kappaf['SA_I']))    #Assumed level of kelp cover at the Indresjkær (station 3) kelp sites (0-1) []


    #Age-dependent grazing and predation/mortality factors
    adgrazfac = np.minimum(1, np.maximum(0, 0.5 + 0.5*(UTDM-TD_halfgUS)/dTD_halfgUS))    
    adpredfac = np.minimum(1, np.maximum(0, 0.5 - 0.5*(UTDM-TD_halfpUmortp)/dTD_halfpUmortp)) 
    out.update({'adgrazfac':adgrazfac, 'adpredfac':adpredfac})
    if (htypeU=='lime treatment'):
        pUlime1 = np.minimum(1, np.maximum(min_pUlime1, 
           0.5*(1+min_pUlime1) - ((UTDM-TD_halfpUlime1)/dTD_halfpUlime1)*0.5*(1-min_pUlime1))) 
        out.update({'pUlime1':pUlime1})    


    #Fix the random seeds for repeatability if req'd
    if (fix_eps_rU==True and len(eps_rU)==0):
        np.random.seed(seed)
        eps_rU = np.random.randn(ndt,ns)
    if (fix_eps_So==True and len(eps_So)==0):
        np.random.seed(seed+100)
        eps_So = np.random.randn(ndt,ns)    
    if (fix_eps_UBtoto==True and len(eps_UBtoto)==0):
        np.random.seed(seed+200)
        eps_UBtoto = np.random.randn(ndt,ns)          


    #Adjust parameters for initial sensitivity factors (necessary here to set default ICs)
    if (ftype=='loglinear'):
        alphaSc = alphaS * (falphaS_T**dT[0,:]) * (falphaS_pCO2**(dpCO2[0,:]/100))
        rUmeanc = rUmean * (frU_T**dT[0,:]) * (frU_pCO2**(dpCO2[0,:]/100))
        pUc = pU * (fpU_T**dT[0,:])

    if (ftype=='linear'):
        alphaSc = alphaS * np.maximum(0,(1+falphaS_T*dT[0,:])) * np.maximum(0,1+falphaS_pCO2*(dpCO2[0,:]/100))
        rUmeanc = rUmean * np.maximum(0,(1+frU_T*dT[0,:])) * \
                           np.maximum(0,1+fEU_pCO2*(dpCO2[0,:]/100)) * \
                           np.maximum(0,1+fpFU_pCO2*(dpCO2[0,:]/100)) * \
                           np.maximum(0,1+fpLU_pCO2*(dpCO2[0,:]/100)) * \
                           np.maximum(0,1+fpJU_pCO2*(dpCO2[0,:]/100))
        pUc = pU * (np.maximum(0,1+fpU_T*dT[0,:]))
        
    fGI_T = 1 + 0.5*(dfGI_summer*dT_summer[0,:] + dfGI_winter*dT_winter[0,:]) #Factor to correct gonad indices for warming


    #Prepare initial condition matrices (default to urchin barren initial conditions if U0 is empty)
    #Do this here because we may make use of parameters (rUmeanc, pUc)
    S0 = prep_array(S0,ns)
    if (len(U0)==0): 
        U0 = rUmeanc * (pUc ** np.arange(nagesU).reshape(nagesU,1))
    else:
        U0 = prep_array(U0,ns)
    S = S0.copy() #NOTE: If you do not use ".copy()" here then Python will CHANGE out.S0 as S changes
    U = U0.copy()


    #Set urchin individual biomass/gonad matrices (UIBM,UIGM) and age-structured biomass/gonad initial conditions (UB,UG)
    FDiet = np.minimum(1, (S/KS)/SA_I)
    UIBM = np.exp(bUIBM[0,:] + bUIBM[1,:]*np.log(UTDM) + 0.5*sigUIBM**2)/1e3; #[kgww]
    UIGM = np.exp(bUIGM[0,:] + bUIGM[1,:]*np.log(1e3*UIBM) + bUIGM[2,:]*FDiet + 0.5*sigUIGM**2)/1e3; #gonad mass [kgww]
    out.update({'UIBM0':UIBM, 'UIGM0':UIGM})
    UB = UIBM * U
    UG = UIGM * U


    #Set the fixed age structure UBnorm if req'd
    if (fix_age_structureU==True):  
        #Allow input of initial total urchin biomass (UBtot0)
        UBtot0 = prep_array(UBtot0,ns,1)
        if (len(UBtot0)==0):
            UBtot0 = np.nan*np.ones([1,ns])
            UBtot0[0,:] = np.sum(UB,0) #If not supplied, use the UB calculated from (default) U0 and UIBM
            
        if (len(Unorm)==0):
            Unorm = U0.copy()
        Unorm = Unorm/np.sum(Unorm,0) #Unorm is normalized such that total abundance = 1
        
        if (model_age_structureU==True):    
            if len(hparsUp)>0:
                for j in range(nagesU):
                    for k in range(ns): #Note numpy is row-major by default
                        UB[j,k] = UBtot0[0,k]*np.interp(0,hparsUp,UBnormp[j,:]) #Assume hparsU=0 prior to IC    
            elif len(UBtotp)>0:
                for j in range(nagesU):
                    for k in range(ns): #Note numpy is row-major by default
                        UB[j,k] = UBtot0[0,k]*np.interp(UBtot0[0,k],UBtotp,UBnormp[j,:])
            U = UB/UIBM
                    
        else:
            Unorm1 = Unorm/Unorm[0,:] #Unorm1 is normalized such that first age class abundance = 1 
            if (len(UBnorm)==0): 
                UBnorm = UB.copy()
            UBnorm = UBnorm/np.sum(UBnorm,0) #UBnorm is normalized such that total biomass = 1
            out.update({'Unorm1':Unorm1, 'Unorm':Unorm, 'UBnorm':UBnorm}) 
            
            #Make sure we start in the reduced space
            UB = UBnorm*UBtot0
            U = UB/UIBM
            UG = UIGM*U            


    #Preallocation
    meanSt = np.nan*np.ones([1,ndt+1])
    stdSt = np.nan*np.ones([1,ndt+1])
    St = np.nan*np.ones([ndt+1,ns])
    meanUt = np.nan*np.ones([nagesU,ndt+1])
    stdUt = np.nan*np.ones([nagesU,ndt+1]) 
    Utott = np.nan*np.ones([ndt+1,ns])
    Ubulkt = np.nan*np.ones([ndt+1,ns])
    meanUBt = np.nan*np.ones([nagesU,ndt+1]) 
    stdUBt = np.nan*np.ones([nagesU,ndt+1])
    UBtott = np.nan*np.ones([ndt+1,ns]) 
    UBbulkt = np.nan*np.ones([ndt+1,ns]) 
    meanUBnormt = np.nan*np.ones([nagesU,ndt+1]) 
    meanUGt = np.nan*np.ones([nagesU,ndt+1])
    stdUGt = np.nan*np.ones([nagesU,ndt+1])
    UGtott = np.nan*np.ones([ndt+1,ns])
    NUAht = np.nan*np.ones([ndt,ns])
    UBAht = np.nan*np.ones([ndt,ns])
    UGAht = np.nan*np.ones([ndt,ns])


    #Record initial conditions
    if (len(agesel_bulk)==0):
        agesel_bulk = np.arange(2,nagesU)
    meanSt[0,0] = np.mean(S,1); stdSt[0,0] = np.std(S,1); St[0,:] = S
    meanUt[:,0] = np.mean(U,1); stdUt[:,0] = np.std(U,1) 
    Utott[0,:] = np.sum(U,0); Ubulkt[0,:] = np.sum(U[agesel_bulk,],0)
    meanUBt[:,0] = np.mean(UB,1); stdUBt[:,0] = np.std(UB,1)
    UBtott[0,:] = np.sum(UB,0); UBbulkt[0,:] = np.sum(UB[agesel_bulk,],0)
    meanUBnormt[:,0] = np.mean(UB/UBtott[0,:],1)
    meanUGt[:,0] = np.mean(UG,1); stdUGt[:,0] = np.std(UG,1)
    UGtott[0,:] = np.sum(UG,0)
    out.update({'S0':S.copy(),'U0':U.copy()})


    #Time stepping
    for i in range(ndt):

        #Adjust parameters for sensitivity factors at time step i if necessary
        if (i>0):
            if (ftype=='loglinear'):
                alphaSc = alphaS * (falphaS_T**dT[i,:]) * (falphaS_pCO2**(dpCO2[i,:]/100))
                rUmeanc = rUmean * (frU_T**dT[i,:]) * (frU_pCO2**(dpCO2[i,:]/100))
                pUc = pU * (fpU_T**dT[i,:])

            if (ftype=='linear'):
                alphaSc = alphaS * np.maximum(0,(1+falphaS_T*dT[i,:])) * np.maximum(0,1+falphaS_pCO2*(dpCO2[i,:]/100))
                rUmeanc = rUmean * np.maximum(0,(1+frU_T*dT[i,:])) * \
                                   np.maximum(0,1+fEU_pCO2*(dpCO2[i,:]/100)) * \
                                   np.maximum(0,1+fpFU_pCO2*(dpCO2[i,:]/100)) * \
                                   np.maximum(0,1+fpLU_pCO2*(dpCO2[i,:]/100)) * \
                                   np.maximum(0,1+fpJU_pCO2*(dpCO2[i,:]/100))
                pUc = pU * (np.maximum(0,1+fpU_T*dT[i,:]))
                
            fGI_T = 1 + 0.5*(dfGI_summer*dT_summer[i,:] + dfGI_winter*dT_winter[i,:]) #Factor to correct gonad indices for warming


        #Adapt the harvest parameters to the present state if req'd
        if (htypeU!='none' and hadaptationU!='none'):
            #Set the observed kelp and urchin biomasses
            So = St[i,:]
            if (hadaptationU=='Sinterp' or hadaptationU=='SUinterp'):
                if fix_eps_So==True: 
                    eps_So1 = eps_So[i,:] 
                else:
                    eps_So1 = np.random.randn(1,ns)
                So = So*np.exp(Sosig*eps_So1 - 0.5*Sosig**2)
            UBtoto = UBtott[i,:]
            if (hadaptationU=='Uinterp' or hadaptationU=='SUinterp'):
                if fix_eps_UBtoto==True: 
                    eps_UBtoto1 = eps_UBtoto[i,:] 
                else:
                    eps_UBtoto1 = np.random.randn(1,ns)
                UBtoto = UBtoto*np.exp(UBtotosig*eps_UBtoto1 - 0.5*UBtotosig**2)  

            if i==0:
                hUfun = []
            hparsU[i,:],hUfun = hadaptationUfn(So,UBtoto,i,ndt,hadaptationU,
                  hUSgrid1,hUUgrid1,hUgrid,hUfun_inp=hUfun,
                  hurryup_par=hurryup_par,cut_losses_Sc=cut_losses_Sc,
                  cut_losses_rate=cut_losses_rate)
#            #Interpolate the harvest effort to the observed (S, UBtot)
#            if (hadaptationU=='SUinterp'):
#                if (len(hUgrid.shape)==2 and i==0):
#                    hUfun = RectBivariateSpline(hUUgrid1, hUSgrid1, hUgrid)
#                elif len(hUgrid.shape)==3:
#                    hUfun = RectBivariateSpline(hUUgrid1, hUSgrid1, hUgrid[:,:,i])
#                hparsU[i,:] = np.maximum(0,hUfun(UBtoto, So, grid=False))
#                
#            elif (hadaptationU=='Sinterp'):
#                if len(hUgrid.shape)==2:
#                    hparsU[i,:] = np.interp(So,hUSgrid1,hUgrid[:,i]) 
#                else:
#                    hparsU[i,:] = np.interp(So,hUSgrid1,hUgrid) 
#                    
#            #Hurry-up effect if required
#            if (hurryup_par>0):
#                hparsU[i,:] = hparsU[i,:]*np.exp(hurryup_par/(ndt-i))
#                
#            #Cut losses if required growback is too large
#            if (cut_losses_rate<np.inf):
#                #print(np.max(np.log(cut_losses_Sc/S)/(ndt-i)))
#                #print(cut_losses_rate)
#                hparsU[i,np.where(np.log(cut_losses_Sc/S)/(ndt-i)>cut_losses_rate)] = 0
##                sel = np.where(np.log(cut_losses_Sc/S)/(ndt-i)>cut_losses_rate)
##                if (sel[0].size>0):
##                    print(sel)
##                    hparsU[i,np.where(np.log(cut_losses_Sc/S)/(ndt-i)>cut_losses_rate)] = 0
            
#            if (htypeU=='lime treatment'):
#                hparsU[i,:] = np.round(hparsU[i,:])


        #Correct for urchin harvest, assumed to occur at year-start
        if (htypeU=='fixed fraction'): #Harvest/cull a fixed, predetermined fraction each year (e.g. exhaustive removal within a given area)
            areahU = hparsU[i,:] * area
            NUAh = np.sum(U,0) * areahU
            UBAh = np.sum(UB,0) * areahU
            UGAh = np.sum(UG,0) * areahU
            pUh = hparsU[i,:]
            UBc = (1-pUh) * UB
        elif (htypeU=='lime treatment'): #Apply hparsU lime treatments over the year
            NUAh = 0; UBAh = 0; UGAh = 0 #We don't extract any urchins, only kill them
            pUh = (1-pUlime1**hparsU[i,:]) #pUlime1 is the age-structured survival rate from 1 lime treatment
            UBc = (1-pUh) * UB
        else:
            pUh = 0
            UBc = UB


        #Calculate grazing flux based on post-harvest biomass at year start (UBc)
        G0 = gUS * np.sum(adgrazfac*UBc,0) #Total grazing flux is summed over urchin age classes [kgww/m2/year]


        #Calculate kelp biomass at year end (S1) and average kelp coverage during the year (SA) (needed to calculate urchin mortality)
        #S1 = intKelp(1,S,rS,alphaSc,KS,G0,0)[0] #Update the Kelp biomass (provisionally using only G0)
        #SA = 0.5*(S+S1)/KS #Use average of year-start and (provisional) year-end S values = provisional year-mean estimate
        S1t = intKelp(dtS,S,rS,alphaSc,KS,G0,0)[0] #Update the Kelp biomass (provisionally using only G0)
        SA = np.mean(S1t,0)/KS


        #Calculate urchin mortality and net survival probability (pUcc)
        pUmortp_max = 1 - fpU_S**SA #Maximum mortality probability due to predation (for given kelp density)
        pUcc = (1-pUh) * pUc * (1-pUmortp_max*adpredfac) #Net survival probability accounting for age-dependent predation mortality
        #Prob(survival) = Prob(not harvested) * 
        #                 Prob(not dying from natural causes|not harvested) *
        #                 Prob(not dying from predation|not dying from natural causes and not harvested)


        #Update the age-structured urchin abundance density for ages >1 (accounts for harvesting, natural mortality, and predation)
        U[1:,] = pUcc[:-1,] * U[:-1,]


        #Individual recruitment to 1st age class
        UA = np.pi*np.sum(((1e-3*UTDM/2)**2)*U,0) #Total area per m2 occupied by urchins (should never exceed 1)
        fA = np.maximum(0, 1-UA) #Settlement area limitation factor.
        fS = frU_S**SA           #Kelp coverage factor.
        if fix_eps_rU==True: 
            eps_rU1 = eps_rU[i,:] 
        else:
            eps_rU1 = np.random.randn(1,ns)


        U[0,:] = rUmeanc * np.exp(rUsig*eps_rU1 - 0.5*rUsig**2) * fA * fS #New recruits to 1st age class
        #NOTE: This model assumes zero correlation between adult abundance and recruitment (Fagerli et al., 2013; Marzloff et al., 2013)
        #      This assumption may not be appropriate for strongly retentive systems


        #If imposing a fixed age_structure, correct the within-time-step age structure U to account for the effect
        #of stochastic variation between age classes on the update of UBtot
        if (fix_age_structureU==True):

            if (model_age_structureU==True): #Use modelled age structure: U(k) = f_k(UBtot)
                U[0,:] = rUmeanc * fA * fS
                UB = UIBM * U
                UBtot = np.sum(UB,0)
                
                sigUBtot1 = np.nan*np.ones([1,ns])
                if len(hparsUp)>0:
                    for k in range(ns):
                        sigUBtot1[0,k] = np.interp(hparsU[i,k],hparsUp,sigUBtotp) 
                elif len(UBtotp)>0:
                    for k in range(ns):
                        sigUBtot1[0,k] = np.interp(UBtot[k],UBtotp,sigUBtotp)
                
                if (lognormal_UBtot_stoch==True):
                    UBtot = UBtot*np.exp(sigUBtot1*eps_rU1 - 0.5*sigUBtot1**2)
                else:
                    UBtot = np.maximum(0, UBtot + sigUBtot1*eps_rU1)
                #This should account for the stochasticity in UBtot that is induced by the maturation of variable cohorts 
                #(and which is not explicitly simulated by the reduced system).
                
                if len(hparsUp)>0:
                    for j in range(nagesU):
                        for k in range(ns): #Note numpy is row-major by default
                            UB[j,k] = UBtot[0,k]*np.interp(hparsU[i,k],hparsUp,UBnormp[j,:])
                elif len(UBtotp)>0:
                    for j in range(nagesU):
                        for k in range(ns): #Note numpy is row-major by default
                            UB[j,k] = UBtot[0,k]*np.interp(UBtot[0,k],UBtotp,UBnormp[j,:])
                U = UB / UIBM
                #This ensures that we remain in the reduced space UBtot <=> U
                        
            else:
                UB = UIBM * U
                UBtot = sum(UB,0)
                Ustar = Unorm1 * (rUmeanc*fA*fS)
                UBtotstar = sum(UIBM*Ustar,0) #Quasi-equilibrium total biomass UBtotstar
            
                Utot = sum(U,0)
                Untr = Unorm * Utot 
                #In the limit UBtot = UBtotstar, Untr provides a good approx. of stochastic variability in UBtot
            
                gammaU = np.minimum(1, np.maximum(1e-8, UBtot) / UBtotstar)
                U = (1-gammaU)*U + gammaU*Untr;
                    
            #U is now the age-abundance distrbution *within the time step*.
            #It is a function of the harvesting, mortality, and recruitment within the time step,
            #plus the value of UBtot at the beginning of the time step.


        FDiet = np.minimum(1, SA/SA_I)
        UB = UIBM * U
        UIGM = np.exp(bUIGM[0,:] + bUIGM[1,:]*np.log(1e3*UIBM) + bUIGM[2,:]*FDiet + 0.5*sigUIGM**2)/1e3; #gonad mass [kgww]
        UIGM = UIGM * fGI_T #Correct for warming
        UG = UIGM * U


        #Calculate grazing flux based on pre-harvest biomass at year end (UB)
        G1 = gUS * np.sum(adgrazfac*UB,0); #Total grazing flux is summed over urchin age classes [kgww/m2/year]
        G = 0.5*(G0 + G1) #Average grazing flux between start and end of time step (post and pre-harvest)
        S = intKelp(1,S,rS,alphaSc,KS,G,0)[0] #Update the Kelp biomass (finally using G0 and G1)


        #Impose a fixed age structure if req'd (dimensional reduction to 2D system (S,UBtot))
        if (fix_age_structureU==True and model_age_structureU==False):
            UB = UBnorm * np.sum(UB,0) #UBnorm is normalized such that sum(UBnorm) = 1
            #Note: Only the total biomass UBtot=sum(UB,1) is remembered between time steps
            U = UB/UIBM
            UG = UIGM*U


        #Store statistics at the end of the time step
        meanSt[0,i+1] = np.mean(S,1) #Ensemble-mean kelp biomass density [kgww/m2] (1*(ndt+1))
        stdSt[0,i+1] = np.std(S,1) #Ensemble standard deviation in kelp biomass density [kgww/m2] (1*(ndt+1))
        St[i+1,:] = S #Kelp biomass density at all times [kgww/m2] ((ndt+1)*ns)
        meanUt[:,i+1] = np.mean(U,1) #Ensemble-mean urchin abundance [#/m2] (nagesU*(ndt+1))
        stdUt[:,i+1] = np.std(U,1) #Ensemble standard deviation in urchin abundance [#/m2] (nagesU*(ndt+1))
        Utott[i+1,:] = np.sum(U,0) #Total urchin abundance [#/m2] ((ndt+1)*ns)
        Ubulkt[i+1,:] = np.sum(U[agesel_bulk,],0) #Total bulk urchin abundance [#/m2] ((ndt+1)*ns)
        meanUBt[:,i+1] = np.mean(UB,1) #Ensemble-mean urchin biomass density [kgww/m2] (nagesU*(ndt+1))
        stdUBt[:,i+1] = np.std(UB,1) #Ensemble standard deviation in urchin biomass density [kgww/m2] (nagesU*(ndt+1))
        UBtott[i+1,:] = np.sum(UB,0) #Total urchin biomass density [kgww/m2] ((ndt+1)*ns)
        UBbulkt[i+1,:] = np.sum(UB[agesel_bulk,],0) #Total bulk urchin biomass density [kgww/m2] ((ndt+1)*ns)
        meanUBnormt[:,i+1] = np.mean(UB/UBtott[i+1,:],1)
        meanUGt[:,i+1] = np.mean(UG,1) #Ensemble-mean urchin gonad biomass density [kgww/m2] (nagesU*(ndt+1))
        stdUGt[:,i+1] = np.std(UG,1) #Ensemble standard deviation in urchin gonad biomass density [kgww/m2] (nagesU*(ndt+1))
        UGtott[i+1,] = np.sum(UG,0) #Total urchin gonad biomass density [kgww/m2] ((ndt+1)*ns)
        if (htypeU!='none'): 
            NUAht[i,:] = NUAh #Harvested urchin number over entire modelled area [] (ndt*ns)
            UBAht[i,:] = UBAh #Harvested urchin biomass over entire modelled area [kgww] (ndt*ns)
            UGAht[i,:] = UGAh #Harvested urchin gonad biomass over entire modelled area [kgww] (ndt*ns)


    out.update({'S':S,'U':U,'UB':UB,'UG':UG,
                'meanSt':meanSt,'stdSt':stdSt,'St':St,
                'meanUt':meanUt,'stdUt':stdUt,'Utott':Utott,'Ubulkt':Ubulkt,
                'meanUBt':meanUBt,'stdUBt':stdUBt,'UBtott':UBtott,'UBbulkt':UBbulkt,
                'meanUBnormt':meanUBnormt,
                'meanUGt':meanUGt,'stdUGt':stdUGt,'UGtott':UGtott})
    if (htypeU!='none'):
        out.update({'NUAht':NUAht,'UBAht':UBAht,'UGAht':UGAht,'pUh':pUh,'hparsU':hparsU})  

    return [S,U,out]


def intKelp(t,S0,rS=0.025,alphaS=1,KS=11,G=1,hS=0):
    """
    Integrate seaweed/kelp biomass from t=0 to times t assuming logistic growth with specified grazing/harvesting.
    
      dS/dt = rS + alphaS*(1-S/KS)*S - G - hS
    
    Input parameters (S0,rS,alphaS,KS,G,hS) can be scalars or vectors of size (1*ns)
    
    Output St (nt*ns) is the kelp biomass at times t for each of the ns time series.
    Output HS (nt*ns) is the total harvested biomass over the period t=0 to t(end).
    
    Uses: anintLogistic.m
    
    Example 1: Integrate for 1 year from initial condition S0=2 kgww/m2 using default parameters, 
               no harvesting, and outputting only the final biomass.
    import KelpUrchin
    import numpy as np
    St = KelpUrchin.intKelp(1,2)[0]
    St #Should give array([[2.89977017]])
    
    Example 2: Integrate for 10 years from initial condition S0=2 kgww/m2, using default parameters,
               no harvesting, and outputting values for t=0,1,...,10 years.
    St = KelpUrchin.intKelp(np.arange(11),2)[0]
    St[-1,:] #Should give array([9.8940805])
    
    Example 3: Integrate for 10 years from initial condition S0=2 kgww/m2, using a range of 
               values for the intrinsic growth rate (0,0.1,...,2 yr-1), no harvesting, 
               and default values for other parameters.
    St = KelpUrchin.intKelp(np.arange(11),2,alphaS=0.1*np.arange(21))[0]
    St[-1,-1] #Should give 10.488736562584908
    
    Example 4: Integrate for 10 years from initial condition S0=5 kgww/m2 using default parameters, 
               with harvesting flux of (0,0.5,1,...5) kgww/m2/year (constant flux).
    St,HS = KelpUrchin.intKelp(np.arange(11),5,hS=0.5*np.arange(11))
    St[-1,2] #Should give 8.379204349925187
    HS[:,2] #Here the cumulative harvesting flux is never limited by resource depletion
    St[-1,-1] #Should give 0.0
    HS[:,-1] #Here the cumulative harvesting flux is limited to 6.11174078
      
    Example 5: Speed test: repeat example 4 for 1000 years with hS=(0,0.005,...5) kgww/m2/year.
    import time
    t0 = time.time(); St,HS = KelpUrchin.intKelp(np.arange(1001),5,hS=0.005*np.arange(1001)); print(time.time()-t0)
    #Executes in 0.06-0.09 s on my PC (Intel core i7, 2x2.6 GHz, 32GB RAM)
    St[-1,10] #Should give 9.8560303029249
    """
    
    St,HS = anintLogistic(t,S0,rS,alphaS,KS,G,hS)[:2]
    #Note: Calling another function here may seem pointless. It is done for consistency with the
    #      Matlab code, which includes extra functionality that may be translated in future (and which
    #      favours the use of anintLogistic).
    return St,HS


def anintLogistic(t,X0,r,alpha,K,G=0,h=0,Ft=[]):
    """
    Analytically integrate the harvested logistic equation with grazing and recruitment:
    
       dX/dt = r + alpha*(1-X/K)*X - G - h
    
    with initial condition X0. The result is a generalized Beverton-Holt update.
    
    Output Xt is an array with dimensions (nt*ns), where nt is the length of the 
    input vector t and ns is number of time series/simulations, inferred from the maximum 
    length of the input parameters (X0,r,alpha,K,G,h) (repeated values assumed for scalars).
    
    Output Ht is the harvested biomass integrated to times t, accounting for non-negativity constraints.
    Note: When the resource is depleted (X=0) it is assumed that harvesting is limited
    to the residual of recruitment minus grazing (if any). 
    
    If net losses are zero at all times, then  analytical solutions are obtained for the
    the non-autonomous, loss-free system:
    
       dX/dt = alpha(t)*(1-X/K)*X
    
    Here the time dependence of alpha can be specified either by passing an integral vector Ft (nt*1 or nt*ns)
    specifying the integral from 0 to times t of f(t), where alpha(t) = alpha*f(t).
    """
    out = dict() #Output dictionary
    t = np.array(t).reshape(-1,1) #t is a column vector of times to integrate to

    X0 = np.asarray(X0)
    r = np.asarray(r)
    alpha = np.asarray(alpha)
    K = np.asarray(K)
    G = np.asarray(G)
    h = np.asarray(h)

    nt = len(t)
    ns = max([X0.size,r.size,alpha.size,K.size,G.size,h.size])

    X0 = prep_array(X0,ns,ndt=[],asvector=True)
    r = prep_array(r,ns,ndt=[],asvector=True)
    alpha = prep_array(alpha,ns,ndt=[],asvector=True)
    K = prep_array(K,ns,ndt=[],asvector=True)
    G = prep_array(G,ns,ndt=[],asvector=True)
    h = prep_array(h,ns,ndt=[],asvector=True)
    #X0, r, etc. should now all be 2D arrays of size (1,ns)

    Xt = np.nan*np.ones([nt,ns])
    Ht = np.nan*np.ones([nt,ns])
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')

    L = G + h - r

    if (np.sum(L!=0)>0): #Solve lossy system with all parameters time-independent
        if (len(Ft)>0):
            print('No analytical solution available for non-autonomous logistic growth with non-zero forcing flux')
        #Equilibria:
        #dX/dt = alpha*(1-X/K)*X - L = 0
        # => X* = (-b +/- sqrt(b^2-4ac))/2a
        #       = (-alpha +/- sqrt(alpha^2 - 4(-alpha/K)(-L)))/(-2.alpha/K)
        #       = K/2 * (1 +/- sqrt(1 - 4L/(alpha.K)))
        #       = K/2 * (1 +/- sqrt(disc))
        #Hence we have: two equilibria (one stable, one unstable) if disc>0
        #               one equilibrium (stable) is disc=0
        #               no equilibria if disc<0
        disc = 1-4*L/(alpha*K)
        rec1 = np.asarray((alpha!=0) & (disc>0)).nonzero()[1]; nrec1 = len(rec1)
        rec2 = np.asarray((alpha!=0) & (disc==0)).nonzero()[1]; nrec2 = len(rec2)
        rec3 = np.asarray((alpha!=0) & (disc<0)).nonzero()[1]; nrec3 = len(rec3)
        rec4 = np.asarray(alpha==0).nonzero()[1]; nrec4 = len(rec4)
        out.update({'disc':disc,'rec1':rec1,'rec2':rec2,'rec3':rec3,'rec4':rec4}) 

        if (nrec1>0):
            #Case 1: disc>0 (=> L < alpha*K/4)
            #
            #Here there are two equilibria (Xp,Xm) and the solution is obtained by partial fractions:
            #dX/dt = -alpha/K * (X^2 - K*X + L*K/alpha)
            #=> 1/(X^2 - K*X + L*K/alpha) * dX = -alpha/K * dt
            #LHS = [1/(X-Xp) - 1/(X-Xm)] * dX/(Xp-Xm)
            #=> [1/(X-Xp) - 1/(X-Xm)] * dX = -alpha*(Xp-Xm)/K * dt, where Xp/Xm are the +/- equilibria
            #Integrating over t:
            #-> log((Xt-Xp)/(Xt-Xm)) - log((X0-Xp)/(X0-Xm)) = -alpha*(Xp-Xm)*t/K
            #=> (Xt-Xp)/(Xt-Xm) = ((X0-Xp)/(X0-Xm))*exp(-alpha*(Xp-Xm)*t/K) = Ct
            #-> Xt = (Xp-Ct*Xm)/(1-Ct), where Ct = ((X0-Xp)/(X0-Xm))*exp(-alpha*(Xp-Xm)*t/K)
            sqdisc1 = np.sqrt(disc[:,rec1])
            X01 = X0[:,rec1]; r1 = r[:,rec1]; alpha1 = alpha[:,rec1]; K1 = K[:,rec1]; G1 = G[:,rec1]; h1 = h[:,rec1]; L1 = L[:,rec1]
            Xp1 = K1/2 * (1+sqdisc1) #This is the stable equilibrium
            Xm1 = K1/2 * (1-sqdisc1) #This is the unstable equilibrium
            gamma1 = alpha1*(Xp1-Xm1)/K1
            
            Ct1 = ((X01-Xp1)/(X01-Xm1)) * np.exp(-1*t*gamma1) #Note that Ct1 is an (nt*ns) array
            
            Xt[:,rec1] = (Xp1-Ct1*Xm1) / (1-Ct1)
            Ht[:,rec1] = t*h1

            #However, this solution does not in general respect non-negativity.
            #Therefore we set to zero for all times t>=tc, where tc is the crossing time (X=0)            
            tc1 = (1/gamma1) * np.log((Xm1/Xp1)*((X01-Xp1)/(X01-Xm1)))
            tc1[X01==0] = 0/gamma1[X01==0] #Numerical safety: When X01=0 we should always have numerator=0
            sel10 = np.asarray(np.isnan(tc1)==False).nonzero()[1] #First subset for not nan, otherwise the subset sel11 will not work
            if (len(sel10)>0):
                sel11 = np.asarray((tc1[0,sel10]>=0) & (tc1[0,sel10]<max(t)) & (L1[0,sel10]>0)).nonzero()
                sel1 = sel10[sel11]
                nsel1 = len(sel1)
                for i in range(nsel1):
                    sel11 = sel1[i]
                    selt = np.asarray(t>=tc1[0,sel11]).nonzero()[0]
                    Xt[selt,rec1[sel11]] = 0
                    Ht[selt,rec1[sel11]] = tc1[0,sel11]*h1[0,sel11] + (t[selt,0]-tc1[0,sel11])*np.minimum(h1[0,sel11], np.maximum(np.array([[0]]), r1[0,sel11]-G1[0,sel11]))

            out.update({'Xp1':Xp1,'Xm1':Xm1,'gamma1':gamma1,'Ct1':Ct1,'tc1':tc1}) 

        if (nrec2>0):
            #Case 2: disc==0 (=> L = alpha*K/4)
            #
            #Here there is one stable equilibrium:
            #dX/dt = -alpha/K * (X^2 - K*X + L*K/alpha)
            #=> 1/(X^2 - K*X + L*K/alpha) * dX = -alpha/K * dt
            #If L=alpha*K/4, LHS = dX / (X-K/2)^2
            #Integrating over t:
            #-> -1/(Xt-K/2) + 1/(X0-K/2) = -alpha*t/K
            #=> Xt = K/2 + 1/(alpha*t/K + C), where C = 1/(X0-K/2)
            X02 = X0[:,rec2]; r2 = r[:,rec2]; alpha2 = alpha[:,rec2]; K2 = K[:,rec2]; G2 = G[:,rec2]; h2 = h[:,rec2]; L2 = L[:,rec2]
            C2 = 1/(X02-K2/2)
            Xt[:,rec2] = K2/2 + 1/(t*(alpha2/K2)+C2)
            Ht[:,rec2] = t*h2

            #However, this solution does not in general respect non-negativity.
            #Therefore we set to zero for all times t>=tc, where tc is the crossing time (X=0)
            #2/K + 1/(X0-K/2) = -alpha*tc/K
            #=> 2*X0/(K*(X0-K/2)) = -alpha*tc/K
            #=> tc = 2*X0/(alpha*(K/2-X0))
            tc2 = 2*X02 / (alpha2*(K2/2-X02))
            sel2 = np.asarray(np.isnan(tc2)==False).nonzero()[1] #First subset for not nan, otherwise the subset sel21 will not work
            if (len(sel2)>0):   
                sel21 = np.asarray((tc2[0,sel2]>=0) & (tc2[0,sel2]<max(t)) & (L2[0,sel2]>0)).nonzero() 
                sel2 = sel2[sel21]
                nsel2 = len(sel2)
                for i in range(nsel2):
                    sel21 = sel2[i]
                    selt = np.asarray(t>=tc2[0,sel21]).nonzero()[0]
                    Xt[selt,rec2[sel21]] = 0
                    Ht[selt,rec2[sel21]] = tc2[0,sel21]*h2[0,sel21] + (t[selt,0]-tc2[0,sel21])*np.minimum(h2[0,sel21], np.maximum(0, r2[0,sel21]-G2[0,sel21]))

            out.update({'K2':K2,'C2':C2,'tc2':tc2}) 

        if (nrec3>0):
            #Case 3: disc<0 (=> L > alpha*K/4)
            #
            #Here there are no equilibria and we use a different integration trick:
            #dX/dt = -alpha/K * (X^2 - K*X + L*K/alpha)
            #=> 1/(X^2 - K*X + L*K/alpha) * dX = -alpha/K * dt
            #=> 1/((X-K/2)^2 + a^2) * dX = -alpha/K * dt, where a^2 = L*K/alpha - K^2/4
            #Integrating over t:
            #1/a * [tan-1((Xt-K/2)/a) - tan-1((X0-K/2)/a)] = -alpha*t/K
            #=> Xt = K/2 + a*tan(-a*alpha*t/K + C), where C = tan-1((X0-K/2)/a)
            X03 = X0[:,rec3]; r3 = r[:,rec3]; alpha3 = alpha[:,rec3]; K3 = K[:,rec3]; G3 = G[:,rec3]; h3 = h[:,rec3]; L3 = L[:,rec3]
            a3 = np.sqrt(L3*K3/alpha3 - 0.25*K3**2)
            C3 = np.arctan((X03-K3/2)/a3)
            gamma3 = a3*alpha3/K3
            Xt[:,rec3] = K3/2 + a3*np.tan(-1*t*gamma3 + C3)
            Ht[:,rec3] = t*h3

            #However, this solution does not in general respect non-negativity.
            #Therefore we set to zero for all times t>=tc, where tc is the crossing time (X=0)
            tc3 = (np.arctan((X03-K3/2)/a3)-np.arctan(-K3/(2*a3)))/gamma3
            tc3[X03==0] = 0/gamma3[X03==0] #Numerical safety: When X03=0 we should always have numerator=0 
            sel30 = np.asarray(np.isnan(tc3)==False).nonzero()[1] #First subset for not nan, otherwise the subset sel31 will not work
            if (len(sel30)>0):   
                sel31 = np.asarray((tc3[0,sel30]>=0) & (tc3[0,sel30]<max(t)) & (L3[0,sel30]>0)).nonzero()
                sel3 = sel30[sel31]
                nsel3 = len(sel3)
                for i in range(nsel3):
                    sel31 = sel3[i]
                    selt = np.asarray(t>=tc3[0,sel31]).nonzero()[0]
                    Xt[selt,rec3[sel31]] = 0
                    Ht[selt,rec3[sel31]] = tc3[0,sel31]*h3[0,sel31] + (t[selt,0]-tc3[0,sel31])*np.minimum(h3[0,sel31], np.maximum(0, r3[0,sel31]-G3[0,sel31]))

            out.update({'a3':a3,'C3':C3,'gamma3':gamma3,'C3':C3,'tc3':tc3}) 

        if (nrec4>0):
            #Case 4: alpha==0
            #
            #Here there is no equilibrium:
            #dX/dt = -L
            #=> Xt = max(0, X0 - L*t)
            X04 = X0[:,rec4]; r4 = r[:,rec4]; G4 = G[:,rec4]; h4 = h[:,rec4]; L4 = L[:,rec4]
            Xt[:,rec4] = X04 - t*L4
            Ht[:,rec4] = t*h4

            #However, this solution does not in general respect non-negativity.
            #Therefore we set to zero for all times t>=tc, where tc is the crossing time (X=0)
            tc4 = X04/L4
            sel40 = np.asarray(np.isnan(tc4)==False).nonzero()[1] #First subset for not nan, otherwise the subset sel31 will not work
            if (len(sel40)>0):
                sel41 = np.asarray((tc4[0,sel40]>=0) & (tc4[0,sel40]<max(t)) & (L4[0,sel40]>0)).nonzero()                
                sel4 = sel40[sel41]
                nsel4 = len(sel4)
                
                for i in range(nsel4):
                    sel41 = sel4[i]
                    selt = np.asarray(t>=tc4[0,sel41]).nonzero()[0]
                    Xt[selt,rec4[sel41]] = 0
                    Ht[selt,rec4[sel41]] = tc4[0,sel41]*h4[0,sel41] + (t[selt,0]-tc4[0,sel41])*np.minimum(h4[0,sel41], np.maximum(0, r4[0,sel41]-G4[0,sel41]))

            out.update({'tc4':tc4}) 

    else: #Solve net-loss-free system with possibly time-dependent alpha
        
        #Equilibria:
        #dX/dt = alpha(t)*(1-X/K)*X = 0
        # => X* = 0 (unstable), K (stable)
        
        #The analytical solution is obtained by partial fractions:
        #dX/dt = -alpha(t)/K * (X^2 - K*X)
        #=> 1/(X^2 - K*X) * dX = -alpha(t)/K * dt
        #LHS = [1/(X-K) - 1/X] * dX/K
        #=> [1/(X-K) - 1/X] * dX = -alpha(t) * dt
        #Integrating over t:
        #-> log((Xt-K)/Xt) - log((X0-K)/X0) = - \int_0^t alpha(s) ds = A(t)
        #=> (Xt-K)/Xt = ((X0-K)/X0)*exp(-A(t))
        #=> Xt = K / (1 - ((X0-K)/X0)*exp(-A(t)))

        if (len(Ft)>0):
            A = Ft*alpha
        else:
            A = t*alpha

        Xt = K / (1 - ((X0-K)/X0)*np.exp(-A))  
        Ht = 0

        out.update({'A':A})

    return Xt,Ht,out


def prep_array(X,ns=1,ndt=[],asvector=False):
    """
    Convert Input scalar, vector, or list array X into a 2D numpy array with ncolumns = ns
    and nrows = ndt (default ndt = 1)
    """

    if isinstance(X, dict):
        for k in X.keys():
            X[k] = prep_array(X[k],ns)
    else:
        if (isinstance(X, list)):
            X = np.asarray(X)
        elif (np.isscalar(X)):
            X = np.asarray(X)

        if (len(X.shape)==0):
            #Scalar input: repeat over dimension (1)
            X = np.tile(X, (1,ns))
        elif (len(X.shape)==1):
            #Vector (1D) input
            if (asvector==True):
                #Leave it alone, only convert to 2D array (1*ns)
                X = np.tile(X, (1,1)) 
            else:
                if (np.isscalar(ndt)):
                    #ndt is given as well as ns -- use these to decide tiling
                    if (ndt==ns & ndt>1):
                        exit("Cannot infer correct tiling of vector input, orientation ambiguous since ndt==ns")
                    elif (len(X)==ns):
                        X = np.tile(X, (ndt,1))
                    elif (len(X)==ndt):
                        X = np.tile(X.reshape(-1,1), (1,ns))
                    elif (len(X)>0):
                        exit("Cannot infer correct tiling of vector input, since len(X)>0 & len(X)!=ns and len(X)!=ndt")

                else:        
                    X = np.tile(X.reshape(-1,1), (1,ns))
        elif (X.shape[1]==1):
            X = np.tile(X, (1,ns))

        if (np.isscalar(ndt) & (X.shape[0]==1)):
            #Repeat in t dimension (0) if necessary
            X = np.tile(X, (ndt,1))

    return(X)


def hadaptationUfn(So, UBtoto, i, ndt, hadaptationU, \
                   hUSgrid1, hUUgrid1, hUgrid, hUfun_inp=[], \
                   hurryup_par=0, cut_losses_Sc=11, cut_losses_rate=0.2):

    #Interpolate the harvest effort to the observed (S, UBtot)
    hUfun = []
    if (hadaptationU=='SUinterp'):
        if (len(hUgrid.shape)==2):
            if not hUfun_inp:
                hUfun = RectBivariateSpline(hUUgrid1,hUSgrid1,hUgrid)
            else:
                hUfun = hUfun_inp
        elif len(hUgrid.shape)==3:
            hUfun = RectBivariateSpline(hUUgrid1,hUSgrid1,hUgrid[:,:,i])
        hU = np.maximum(0,hUfun(UBtoto,So,grid=False))
    elif (hadaptationU=='Sinterp'):
        if len(hUgrid.shape)==2:
            hU = np.interp(So,hUSgrid1,hUgrid[:,i]) 
        else:
            hU = np.interp(So,hUSgrid1,hUgrid) 

    #Hurry-up effect if required
    if (hurryup_par>0):
        hU = hU*np.exp(hurryup_par/(ndt-i))

    #Cut losses if required growback is too large
    if (cut_losses_rate>0):
        #hU[np.log(cut_losses_Sc/So)/(ndt-i)>cut_losses_rate] = 0
        Scrit = cut_losses_Sc*(cut_losses_rate**(ndt-i))
        #Scrit = cut_losses_Sc*np.exp(-1*cut_losses_rate**(ndt-i))
        #print((0.5+0.5*np.tanh((So-Scrit)/0.1)))
        hU = hU*(0.5+0.5*np.tanh((So-Scrit)/0.1))

    return (hU,hUfun)
