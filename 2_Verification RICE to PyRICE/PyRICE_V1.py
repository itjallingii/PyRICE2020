import numpy as np
import pandas as pd
import math
from scipy.stats import norm, skewnorm, cauchy, lognorm
import logging
import json
import sys
import os

#pydice_folder = os.path.dirname(os.getcwd()) + "\\1_Model"

class PyRICE(object):
    """ RICE simulation model:
        tstep: time step/interval
        steps: amount of years looking into the future
        model_specification: model specification for 'EMA_det', 'EMA_dist' or 'Validation'  
    """
    def __init__(self, tstep=10, steps=31, model_specification="Validation_2",fdamage = 1):

        #setting up list for the total time period the model

        #validated
        self.tstep = tstep # (in years)
        self.steps = steps
        self.tperiod = []
        self.startYear = 2005
        self.model_specification = model_specification
        self.fdamage = fdamage

        #arrange simulation timeline
        for i in range(0, self.steps):
            self.tperiod.append((i*self.tstep)+self.startYear)

        #setup of json file to store model results
        with open(pydice_folder + '\\ecs_dist_v5.json') as f:
            d=json.load(f)

        #setting up three distributions for the climate sensitivity; normal lognormal and gauchy
        #is this for the damage function or only for the climate sensitivity parameter? 

        #creating a list from the dist of t2xC02
        np.random.seed(10)

        minb = 0
        maxb = 20
        nsamples = 1000

        samples_norm = np.zeros((0,))
        while samples_norm.shape[0] < nsamples:
            samples = (norm.rvs(d['norm'][0],d['norm'][1],nsamples))
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_norm = np.concatenate((samples_norm, accepted), axis=0)
        samples_norm = samples_norm[:nsamples]

        samples_lognorm = np.zeros((0,))
        while samples_lognorm.shape[0] < nsamples:
            samples = (lognorm.rvs(d['lognorm'][0],d['lognorm'][1],d['lognorm'][2],nsamples))
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_lognorm = np.concatenate((samples_lognorm, accepted), axis=0)
        samples_lognorm = samples_lognorm[:nsamples]

        samples_cauchy = np.zeros((0,))
        while samples_cauchy.shape[0] < nsamples:
            samples = (cauchy.rvs(d['cauchy'][0],d['cauchy'][1],nsamples))
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_cauchy = np.concatenate((samples_cauchy, accepted), axis=0)
        samples_cauchy = samples_cauchy[:nsamples]

        # extend array with the deterministic value of the nordhaus

        # ipv 2.9 from RICE2010
        samples_norm = np.append(samples_norm, 3.2)
        samples_lognorm = np.append(samples_lognorm, 3.2)
        samples_cauchy = np.append(samples_cauchy, 3.2)

        self.samples_t2xco2 = [samples_norm, samples_lognorm, samples_cauchy]


    def __call__(self, 

        #uncertainties from Nordhaus(2007) (first draft)
        #Check with RICE2010 paper
        #these are recognized uncertainties who are set up deterministicly 

        #DICE Uncertainties
        #t2xco2_index=-1,                    3
        #t2xco2_dist=0,      
        #tfp_gr=0.079,

        #sigma_gr=-0.01,
        #pop_gr=0.134,
        #fosslim=6000,
        #cback=344,
        #decl_back_gr=0.025,
        #limmiu=1.2,
        #fdamage=0,     
        #sr=0.249,
        #irstp=0.015,
        #periodfullpart=21,
        #miu_period=29

        #RICE Uncertainties
        t2xco2_index=-1,
        t2xco2_dist=0,
        fosslim=6000,

        #tfp_gr=0.079,
        #sigma_gr=-0.01,             #ALL OF THESE ARE REGION SPECIFIC IN RICE, HOW TO SAMPLE
        #pop_gr=0.134,        
        #cback=344,
        #decl_back_gr=0.025         
        #sr = 0.249           # Savings rate is very different for every region --> how to implement in non optimized RICE?

        periodfullpart=7,      #in OPT RICE period full part is 2075
        miu_period=13,          #2155 in RICE opt scenario when global emissions are near zero
        limmiu=1,    #Upper limit on control rate after 2150, in RICE 1 
        fdamage=1,   #0 is original damage function in RICE 1 is fitted SLR BUILT IN SHAJEE DAMAGE FUNCTIONS
        irstp = 0.015,   # Initial rate of social time preference (per year) (0.015) (RICE2010 OPT))        

        **kwargs):

        """
        ######################## INITIALIZE DATA ########################
        """
        RICE_DATA = pd.read_excel("RICE_2010_base_000.xlsm", sheet_name="Data")
        RICE_PARAMETER = pd.read_excel("RICE_2010_base_000.xlsm", sheet_name="Parameters")
        RICE_DAMAGE = pd.read_excel("RICE_2010_base_000.xlsm", sheet_name="Damage")
        RICE_POP_gr = pd.read_excel("RICE_2010_base_000.xlsm", sheet_name="Pop_gr")
        RICE_results = pd.read_excel("RICE_2010_opt_000.xlsm", sheet_name="Results")

        regions_list = ["US", "OECD-Europe","Japan","Russia","Non-Russia Eurasia","China","India","Middle East","Africa",
            "Latin America","OHI","Other non-OECD Asia"]

        #get some validation series
        self.emissions_series = RICE_results.iloc[3:4,1:32]

        """
        ####################### Population PARAMETERS and set up dataframe format #######################
        """
        #get population growth rates for each region
        self.a=[]
        for i in range(31):  
            if i == 0:
                self.a.append("region")
            k = 2005 + 10 * i
            k = str(k)
            self.a.append(k)    

        self.region_pop_gr = RICE_POP_gr.iloc[10:22,3:35]
        self.region_pop_gr.columns = self.region_pop_gr.columns=self.a
        self.region_pop_gr = self.region_pop_gr.set_index('region') 

        #get population data for 2005
        population2005 = RICE_DATA.iloc[19:31,0]
        regions = RICE_DATA.iloc[19:31,2]
        self.population2005 = pd.concat([population2005,regions],axis = 1)

        self.population2005.columns = ['2005', 'region']
        self.population2005 = self.population2005.set_index('region')
        self.region_pop = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        """
        ############################# LEVERS ###############################
        """

        #setting up model levers
        #if model is EMA the emission control rate miu is sampled 
        #if specification is DICE optimal, the optimal control rate range for every region is taken from Nordhaus 


        ###################### GET CONTROLS FROM RICE OPTIMAL RUN #########################
        #Savings rate (optlrsav = 0.2582781457) from the control file
        validation_series = pd.read_excel("RICE_2010_opt_000.xlsm", sheet_name="Validation series")

        #only sampling S tot nu toe ADJUST
        miu_opt_series = validation_series.iloc[6:18,3:34] 
        miu0 = miu_opt_series.iloc[:,0]
        miu0.index = self.region_pop_gr.index

        #set dataframes for MIU and S
        self.miu = pd.DataFrame(data=np.zeros([12, 31]), columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)          
        self.S = pd.DataFrame(data=np.zeros([12, 31]), columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)          

        self.miu.iloc[:,0] = miu0

        #Controls with random sampling
        if self.model_specification == "EMA":

            #generate random numbers for controls
            sampled_S = np.random.uniform(0.15, 0.45,12)
            sampled_limmiu = np.random.uniform(0.8, 1.2,12)
            sampled_miu_period = np.random.uniform(15, 25,12)

            #set start savings rate to converge from
            self.S.iloc[:,0] = sampled_S

            #set uncertainties that drive MIU
            self.limmiu= sampled_limmiu 
            self.irstp = irstp
            self.miu_period = sampled_miu_period

        #full RICE2010 replicating run
        if self.model_specification == "Validation_1":

            #set savings rate and control rate as optimized RICE 2010
            sr_opt_series = validation_series.iloc[21:33,34]
            self.sr = sr_opt_series           
            self.S = self.sr
            self.S = self.S.apply(pd.to_numeric)
            self.S.index = self.region_pop_gr.index

            #set emission control rate for the whole run according to RICE2010 opt.
            self.miu = miu_opt_series
            self.miu.index = self.region_pop_gr.index
            self.miu.columns = self.region_pop_gr.columns

            #set uncertainties that drive MIU
            self.limmiu= limmiu
            self.irstp = irstp
            self.miu_period = miu_period

        #EMA Deterministic
        if self.model_specification == "Validation_2":

            #set savings rate and control rate as optimized RICE 2010
            sr_opt_series = validation_series.iloc[21:33,3:34] 
            sr_opt_series.index = self.region_pop_gr.index
            sr_opt_series.columns = self.region_pop_gr.columns
            miu_opt_series.index = self.region_pop_gr.index
            miu_opt_series.columns = self.region_pop_gr.columns

            print(miu_opt_series.iloc[:,0:2])
            self.miu.iloc[:,0:2] = miu_opt_series.iloc[:,0:2]
            self.miu.index = self.region_pop_gr.index
            self.miu.columns = self.region_pop_gr.columns

            self.S.iloc[:,0:2] = sr_opt_series.iloc[:,0:2]
            #self.S = self.S.apply(pd.to_numeric)
            self.S.index = self.region_pop_gr.index

            print("savings initialized")
            print(self.S)

            print("emission control rate  initialized")
            print(self.miu)
            #set uncertainties that drive MIU
            self.limmiu= 1
            self.irstp = irstp
            self.miu_period = [12,15,15,10,10,11,13,13,13,14,13,14]

        """
        ######################## DEEP UNCERTAINTIES ########################
        """

        # Equilibrium temperature impact [dC per doubling CO2]/
        # CLimate sensitivity parameter (3.2 RICE OPT)
        self.t2xco2 = self.samples_t2xco2[t2xco2_dist][t2xco2_index]

        # Choice of the damage function (structural deep uncertainty)
        self.fdamage = fdamage

        print("used damage function is: ")
        print(self.fdamage)



        """
        ####################### Carbon cycle PARAMETERS #######################
        """            

        #RICE2010 INPUTS
        # Initial concentration in atmosphere 2000 [GtC]
        self.mat0 = 787 
        # Initial concentration in atmosphere 2010 [GtC]
        self.mat1 = 829
        # Initial concentration in upper strata [GtC]
        self.mu0 = 1600.0 #1600 in excel
        # Initial concentration in lower strata [GtC]
        self.ml0 = 10010.0
        # Equilibrium concentration in atmosphere [GtC]
        self.mateq = 588.0 
        # Equilibrium concentration in upper strata [GtC]
        self.mueq = 1500.0 
        # Equilibrium concentration in lower strata [GtC]
        self.mleq = 10000.0

        #parameters are scaled with 100, check with cllimate equations
        self.b11 = 0.088                                   #88 in excel
        self.b23 = 0.00500                                 #0.5 in excel
        self.b12 = 1 -  self.b11                           
        self.b21 =  self.b11 *  self.mateq /  self.mueq    
        self.b22 = 1 -  self.b21 -  self.b23               #good in excel       
        self.b32 =  self.b23 *  self.mueq /  self.mleq     #good in excel
        self.b33 = 1 -  self.b32                           #good in excel       

        # 2000 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
        self.fex0 = -0.06
        # 2100 forcings of non-CO2 GHG [Wm-2]
        self.fex1 = 0.30
        # Forcings of equilibrium CO2 doubling [Wm-2]
        self.fco22x = 3.8


        """
        ###################### CLIMATE INITIAL VALUES ######################
        """
        #RICE2010 INPUTS

        # Equilibrium temperature impact [dC per doubling CO2]
        # self.t2xco2 = t2xco2
        # Initial lower stratum temperature change [dC from 1900]
        self.tocean0 = 0.0068 
        # Initial atmospheric temperature change [dC from 1900]
        self.tatm0 = 0.83 


        # 2013 version and earlier:
        # Initial climate equation coefficient for upper level
        # self.c10 = 0.098
        # Regression slope coefficient (SoA~Equil TSC)
        # self.c1beta = 0.01243
        # Transient TSC Correction ("Speed of Adjustment Parameter")
        # self.c1 = self.c10+self.c1beta*(self.t2xco2-2.9)

        #DICE2013R
        # Climate equation coefficient for upper level
        #self.c1 = 0.098
        # Transfer coefficient upper to lower stratum
        #self.c3 = 0.088
        # Transfer coefficient for lower level
        #self.c4 = 0.025
        # Climate model parameter
        #self.lam =  self.fco22x /  self.t2xco2

        #RICE2010
        # Climate equation coefficient for upper level
        self.c1 = 0.208
        # Transfer coefficient upper to lower stratum
        self.c3 = 0.310
        # Transfer coefficient for lower level
        self.c4 = 0.05
        # Climate model parameter
        self.lam =  self.fco22x /  self.t2xco2

        """
        ######################### CARBON PARAMETERS ########################
        """

        self.mat = np.zeros((self.steps,))
        self.mu = np.zeros((self.steps,))
        self.ml = np.zeros((self.steps,))
        self.forcoth = np.zeros((self.steps,))
        self.forc = np.zeros((self.steps,))

        """
        ######################## CLIMATE PARAMETERS ########################
        """

        # Increase temperature of atmosphere [dC from 1900]
        self.temp_atm = np.zeros((self.steps,))
        # Increase temperature of lower oceans [dC from 1900]
        self.temp_ocean = np.zeros((self.steps,))


        """
        ######################## DAMAGE PARAMETERS ########################
        """


        #damage parameters excluding SLR from base file --> CHECKED WITH RICE OPT
        self.damage_parameters = RICE_DAMAGE.iloc[20:28,7:20]
        self.damage_parameters = self.damage_parameters.transpose()
        self.damage_parameters = self.damage_parameters.rename(columns=self.damage_parameters.iloc[0])
        self.damage_parameters = self.damage_parameters.drop("Unnamed: 7")
        self.damage_parameters.index = self.region_pop_gr.index

        #damage parameters INCLUDING SLR FIT Dennig et al without country specific
        self.damage_parameters_slr_fit = pd.read_excel("RICE_2010_opt_000.xlsm", sheet_name="SLR fitted parameters")
        self.damage_parameters_slr_fit = self.damage_parameters_slr_fit.iloc[27:39,12:15]
        self.damage_parameters_slr_fit.index = self.region_pop_gr.index

        #inclue SHAJEE stuff here
        """
        ####################### Capital and Economic PARAMETERS #######################
        """

        #get data for factor productivity growth
        self.tfpgr_region =  RICE_DATA.iloc[52:64,1:32]
        self.tfpgr_region.columns = self.region_pop_gr.columns
        self.tfpgr_region.index = self.region_pop_gr.index

        #get initial values for various parameters
        self.initails_par = RICE_PARAMETER.iloc[33:40,5:17]
        self.initials_par = self.initails_par.transpose()
        self.initials_par.index = self.region_pop_gr.index

        #setting up total factor productivity
        self.tfp_2005 = self.initials_par.iloc[:,5]
        self.tfp_region = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #setting up capital parameters
        self.k_2005 = self.initials_par.iloc[:,4]
        self.k_region = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.dk = 0.1
        self.gama = 0.3

        #setting up Y Gross
        self.Y_gross = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.ynet = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.damages = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.dam_frac = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #extra dataframe for calculating exponent
        self.Sigma_gr_tussenstap = pd.DataFrame(data=np.zeros([12, 1]),index=self.region_pop_gr.index)
        self.Sigma_gr_tussenstap["exp"] = ""

        #Dataframes for emissions, economy and utility
        self.Eind = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.E = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.Etree = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.cumetree = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.CCA = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.CCA_tot = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.Abetement_cost = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.Abetement_cost_RATIO = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.Mabetement_cost = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.CPRICE = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #economy parameters per rgeion
        self.Y = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.I = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.C = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.CPC = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #output metrics
        self.util_sdr = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.inst_util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.per_util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.cum_util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.reg_cum_util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.reg_util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        self.per_util_ww = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.cum_per_util = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.inst_util_ww = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #Output-to-Emission
        #Change in sigma: the cumulative improvement in energy efficiency)
        self.sigma_growth_data = RICE_DATA.iloc[70:82,1:6]
        self.Emissions_parameter = RICE_PARAMETER.iloc[65:70,5:17].transpose()

        self.sigma_growth_data.index = self.region_pop_gr.index
        self.Emissions_parameter.index = self.region_pop_gr.index

        #set up dataframe for saving CO2 to output ratio
        self.Sigma_gr = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #CO2-equivalent-emissions growth to output ratio in 2005
        self.Sigma_gr.iloc[:,0] = self.sigma_growth_data.iloc[:,0]

        #Period at which have full participation
        self.periodfullpart = periodfullpart #CHECK THIS WITH RICE

        # Fraction of emissions under control based on the Paris Agreement
        # US withdrawal would change the value to 0.7086 
        # https://climateanalytics.org/briefings/ratification-tracker/ (0.8875)
        self.partfract2005 = 1

        #Fraction of emissions under control at full time
        self.partfractfull = 1.0

        # Decline rate of decarbonization (per period)
        self.decl_sigma_gr = -0.001

        # Carbon emissions from land 2010 [GtCO2 per year]
        self.eland0 = 1.6
        # Decline rate of land emissions (per period) CHECKED
        self.ecl_land = 0.2

        # Elasticity of marginal utility of consumption (1.45) # CHECKED
        self.elasmu = 1.50

        #Emission data
        self.emission_factor = RICE_DATA.iloc[87:99,6]
        self.emission_factor.index = self.region_pop_gr.index
        self.Eland0 = 1.6 #(RICE2010 OPT)
        #Sigma_gr_tussenstap

        #get alpha data
        self.Alpha_data = RICE_DATA.iloc[357:369,1:60]
        self.additative_scaling_weights = RICE_DATA.iloc[167:179,14:17]
        self.multiplutacive_scaling_weights = RICE_DATA.iloc[232:244,1:2] / 1000

        self.Alpha_data.index = self.region_pop_gr.index
        self.additative_scaling_weights.index = self.region_pop_gr.index
        self.multiplutacive_scaling_weights.index = self.region_pop_gr.index  

        #Cost of abatement
        self.abatement_data = RICE_PARAMETER.iloc[56:60,5:17].transpose()
        self.abatement_data.index =self.region_pop_gr.index

        self.pbacktime = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.cost1 = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)

        #CO2 to economy ratio
        self.sigma_region = pd.DataFrame(data=np.zeros([12, 31])*np.nan, columns=self.region_pop_gr.columns,index=self.region_pop_gr.index)
        self.sigma_region.iloc[:,0] = self.Emissions_parameter.iloc[:,2] 

        #cback per region
        self.cback_region = self.abatement_data.iloc[:,0]
        self.cback_region.index = self.region_pop_gr.index

        self.ratio_asymptotic = self.abatement_data.iloc[:,2]
        self.decl_back_gr = self.abatement_data.iloc[:,3]
        self.expcost2 = 2.8   #RICE 2010 OPT


        """
        ####################### LIMITS OF THE MODEL ########################
        """

        #didnt check these yet

        # Output low (constraints of the model)
        self.y_lo = 0.0
        self.ygross_lo = 0.0
        self.i_lo = 0.0
        self.c_lo = 2.0
        self.cpc_lo = 0
        self.k_lo = 1.0
        # self.miu_up[0] = 1.0

        self.mat_lo = 10.0
        self.mu_lo = 100.0
        self.ml_lo = 1000.0
        self.temp_ocean_up = 20.0
        self.temp_ocean_lo = -1.0
        self.temp_atm_lo = 0.0

        #self.temp_atm_up = 20 or 12 for 2016 version
        self.temp_atm_up = 40.0      

        """
        ####################### INI CARBON and climate SUB-MODEL #######################
        """

        # Carbon pools
        self.mat[0] = self.mat0
        self.mat[1] = self.mat1

        if(self.mat[0] < self.mat_lo):
            self.mat[0] = self.mat_lo

        self.mu[0] = self.mu0
        if(self.mu[0] < self.mu_lo):
            self.mu[0] = self.mu_lo

        self.ml[0] = self.ml0
        if(self.ml[0] < self.ml_lo):
            self.ml[0] = self.ml_lo

        # Radiative forcing
        self.forcoth[0] = self.fex0
        self.forc[0] = self.fco22x*(np.log(((self.mat[0]+self.mat[1])/2)/596.40)/np.log(2.0)) + self.forcoth[0]

        """
        ################# CLIMATE PARAMETER INTITIALISATION ################
        """
        #checked with RICE2010

        # Atmospheric temperature
        self.temp_atm[0] = self.tatm0

        if(self.temp_atm[0] < self.temp_atm_lo):
            self.temp_atm[0] = self.temp_atm_lo
        if(self.temp_atm[0] > self.temp_atm_up):
            self.temp_atm[0] = self.temp_atm_up

        # Oceanic temperature
        self.temp_ocean[0] = 0.007

        if(self.temp_ocean[0] < self.temp_ocean_lo):
            self.temp_ocean[0] = self.temp_ocean_lo
        if(self.temp_ocean[0] > self.temp_ocean_up):
            self.temp_ocean[0] = self.temp_ocean_up

        """
        ################# ECONOMIC PARAMETER INTITIALISATION ################
        """

        #Insert population at 2005 for all regions
        self.region_pop.iloc[:,0] = self.population2005.iloc[:,0]

        #total factor production at 2005
        self.tfp_region.iloc[:,0] = self.tfp_2005

        #initial capital in 2005
        self.k_region.iloc[:,0] = self.k_2005

        # Gama: Capital elasticity in production function
        self.Y_gross.iloc[:,0] = (self.tfp_region.iloc[:,0]*((self.region_pop.iloc[:,0]/1000)**(1-self.gama)) * (self.k_region.iloc[:,0]**self.gama))

        #original RICE parameters dam_frac
        if self.fdamage == 0:
            self.dam_frac.iloc[:,0] =  (self.damage_parameters.iloc[:,0]*self.temp_atm[0] 
                            + self.damage_parameters.iloc[:,1]*(self.temp_atm[0]**self.damage_parameters.iloc[:,2])) * 0.01

        #Damage parameters RICE2010 fitted with extra SLR component
        if self.fdamage == 1:
            self.dam_frac.iloc[:,0] = 0.01 * (self.damage_parameters_slr_fit.iloc[:,1] * self.temp_atm[0] + 
                                              (self.damage_parameters_slr_fit.iloc[:,2] *
                                               (self.temp_atm[0]**self.damage_parameters.iloc[:,2])))

        #Net output damages
        self.ynet.iloc[:,0] = self.Y_gross.iloc[:,0]/(1.0+self.dam_frac.iloc[:,0])

        #Damages in 2005
        self.damages.iloc[:,0] = self.Y_gross.iloc[:,0] - self.ynet.iloc[:,0]

        #Cost of backstop
        self.pbacktime.iloc[:,0] = self.cback_region

        # Adjusted cost for backstop
        self.cost1.iloc[:,0] = self.pbacktime.iloc[:,0]*self.sigma_region.iloc[:,0]/self.expcost2

        #decline of backstop competitive year (RICE2010 OPT)
        self.periodfullpart = 2250

        #Emissions from land change use
        self.Etree.iloc[:,0] = self.Emissions_parameter.iloc[:,3]
        self.cumetree.iloc[:,0] = self.Emissions_parameter.iloc[:,3]

        #industrial emissions 2005
        self.Eind.iloc[:,0] =  self.sigma_region.iloc[:,0] * self.Y_gross.iloc[:,0] * (1 - self.miu.iloc[:,0])

        #initialize initial emissions
        self.E.iloc[:,0] = self.Eind.iloc[:,0] + self.Etree.iloc[:,0]
        self.CCA.iloc[:,0] = self.Eind.iloc[:,0]
        self.CCA_tot.iloc[:,0] = self.CCA.iloc[:,0] + self.cumetree.iloc[:,0]

        #doesnt do much here
        self.partfract = 1 


        """
        ####################### INIT NET ECONOMY SUB-MODEL ######################
        """                   

        #Cost of climate change to economy
        #Abettement cost ratio of output
        self.Abetement_cost_RATIO.iloc[:,0] = self.cost1.iloc[:,0]*(self.miu.iloc[:,0] ** self.expcost2)

        #Abettement cost total
        self.Abetement_cost.iloc[:,0] = self.Y_gross.iloc[:,0] * self.Abetement_cost_RATIO.iloc[:,0]

        #Marginal abetement cost
        self.Mabetement_cost.iloc[:,0] = self.pbacktime.iloc[:,0] * self.miu.iloc[:,0]**(self.expcost2-1)

        #Resulting carbon price
        self.CPRICE.iloc[:,0] = self.pbacktime.iloc[:,0] * 1000 * (self.miu.iloc[:,0]**(self.expcost2-1))     

        # Gross world product (net of abatement and damages)
        self.Y.iloc[:,0] = self.ynet.iloc[:,0]-self.Abetement_cost.iloc[:,0]           

        ##############  Investments & Savings  #########################

        #investments per region given the savings rate 
        self.I.iloc[:,0] = self.S.iloc[:,0] * self.Y.iloc[:,0]

        #consumption given the investments
        self.C.iloc[:,0] = self.Y.iloc[:,0] - self.I.iloc[:,0]

        #consumption per capita
        self.CPC.iloc[:,0] = (1000 * self.C.iloc[:,0]) / self.region_pop.iloc[:,0]

        #Utility

        #Initial rate of social time preference per year
        self.util_sdr.iloc[:,0] = 1

        #Instantaneous utility function equation 
        self.inst_util.iloc[:,0] = ((1 / (1 - self.elasmu)) * (self.CPC.iloc[:,0])**(1 - self.elasmu) + 1) * self.Alpha_data.iloc[:,0]           

        #CEMU period utilitity         
        self.per_util.iloc[:,0] = self.inst_util.iloc[:,0] * self.region_pop.iloc[:,0] * self.util_sdr.iloc[:,0]

        #Cummulativie period utilty without WW
        self.cum_per_util.iloc[:,0] = self.per_util.iloc[:,0] 

        #Instantaneous utility function with welfare weights
        self.inst_util_ww.iloc[:,0] = self.inst_util.iloc[:,0] * self.Alpha_data.iloc[:,0]

        #Period utility with welfare weights
        self.per_util_ww.iloc[:,0] = self.inst_util_ww.iloc[:,0] * self.region_pop.iloc[:,0] * self.util_sdr.iloc[:,0]

        #cummulative utility with ww
        self.reg_cum_util.iloc[:,0] =  self.per_util.iloc[:,0] 

        #scale utility with weights derived from the excel
        self.reg_util.iloc[:,0] = 10  * self.multiplutacive_scaling_weights.iloc[:,0] * self.reg_cum_util.iloc[:,0] + self.additative_scaling_weights.iloc[:,0] - self.additative_scaling_weights.iloc[:,2]  

        #calculate worldwide utility 
        self.utility = self.reg_util.sum()           

        """
        ########################################## RICE MODEL ###################################################    
        """   

        #Follows equations of notes #TOTAL OF 30 STEPS UNTIL 2305
        for t in range(1,31): 

            """
            ####################### GROSS ECONOMY SUB-MODEL ######################
            """
            #calculate population at time t
            self.region_pop.iloc[:,t] = self.region_pop.iloc[:,t-1] *  2.71828 **(self.region_pop_gr.iloc[:,t]*10)

            #TOTAL FACTOR PRODUCTIVITY level
            self.tfp_region.iloc[:,t] = self.tfp_region.iloc[:,t-1] * 2.71828 **(self.tfpgr_region.iloc[:,t]*10)

            #determine capital stock at time t
            self.k_region.iloc[:,t] = self.k_region.iloc[:,t-1]*(1-self.dk)**self.tstep + self.tstep*self.I.iloc[:,t-1]

            #lower bound capital


            #self.k_region[self.k_region < 1] = 1

            #determine Ygross at time t
            self.Y_gross.iloc[:,t] = self.tfp_region.iloc[:,t] * ((self.region_pop.iloc[:,t]/1000)**(1-self.gama))*(self.k_region.iloc[:,t]**self.gama)     

            #lower bound Y_Gross
            #self.Y_gross[self.Y_gross < 1] = 1


            #print(self.Y_gross.iloc[:,t])

            #capital and ygross show minor deviations after t =1 because of influence Y net
            #damage function is slidely different because of different damage functions
            #this influences the gross economy cycle as well as emissions, damages and welfare

            #calculate the sigma growth and the emission rate development          
            if t == 1:
                self.Sigma_gr.iloc[:,t] = self.sigma_growth_data.iloc[:,4] + (self.sigma_growth_data.iloc[:,2] - self.sigma_growth_data.iloc[:,4]  )

                self.sigma_region.iloc[:,t] = self.sigma_region.iloc[:,t-1] *  (2.71828 ** (self.Sigma_gr.iloc[:,t]*10)) * self.emission_factor

            if t > 1 :
                self.Sigma_gr.iloc[:,t] = self.sigma_growth_data.iloc[:,4] + (self.Sigma_gr.iloc[:,t-1] - self.sigma_growth_data.iloc[:,4]  ) * (1-self.sigma_growth_data.iloc[:,3] )

                self.sigma_region.iloc[:,t] = self.sigma_region.iloc[:,t-1] *  (2.71828 ** ( self.Sigma_gr.iloc[:,t]*10)) 


            #print("CO2 economy ratio = " + str(t))
            #print(self.sigma_region.iloc[:,t])

            if self.model_specification == "EMA":
                # control rate is maximum after target period, otherwise linearly increase towards that point from t[0]
                # Control rate limit
                if t > 1:
                        for index in range(0,12):            
                            calculated_miu = self.miu.iloc[index,t-1] + (self.limmiu - self.miu.iloc[index,1]) / self.miu_period[index]
                            self.miu.iloc[index, t]= min(calculated_miu,1.00)

            if self.model_specification == "Validation_2": 
                if t > 1:
                    for index in range(0,12):            
                        calculated_miu = self.miu.iloc[index,t-1] + (self.limmiu - self.miu.iloc[index,1]) / self.miu_period[index]
                        self.miu.iloc[index, t]= min(calculated_miu,1.00)

            print("current emmission control rate")       
            print(self.miu)
            #controlrate is werird output does not match --> this will cause CO2 emissions also not to match
            #print("Control rate = " + str(t))
            #print(self.miu.iloc[:,t])

            #Define function for EIND --> BIG STOP FROM t = 0 to t =1 something not right
            self.Eind.iloc[:,t] = self.sigma_region.iloc[:,t] * self.Y_gross.iloc[:,t] * (1 - self.miu.iloc[:,t])

            #yearly emissions from land change
            self.Etree.iloc[:,t] = self.Etree.iloc[:,t-1]*(1-self.Emissions_parameter.iloc[:,4])

            #print("emissions from change in land use: t = " + str(t))
            #print(self.Etree.iloc[:,t])

            #yearly combined emissions
            self.E.iloc[:,t] = self.Eind.iloc[:,t] + self.Etree.iloc[:,t]

            #cummulative emissions from land change
            self.cumetree.iloc[:,t] = self.cumetree.iloc[:,t-1] + self.Etree.iloc[:,t] * 10 

            #cummulative emissions from industry
            self.CCA.iloc[:,t] = self.CCA.iloc[:,t-1] + self.Eind.iloc[:,t] * 10

            #limits of CCA
            #self.CCA[self.CCA > fosslim] = fosslim + 1

            #total cummulative emissions
            self.CCA_tot = self.CCA.iloc[:,t] + self.cumetree.iloc[:,t]

            """
            ####################### CARBON SUB MODEL #######################
            """

            # Carbon concentration increase in atmosphere [GtC from 1750]

            E_worldwilde_per_year = self.E.sum()  #1    #2      #3

            E_worldwilde_per_year_placeholder = RICE_results.iloc[3:4,1:32]
            E_worldwilde_per_year_placeholder = E_worldwilde_per_year_placeholder.to_numpy()

            #print(" model based emisions ")
            #print(E_worldwilde_per_year[t])

            #print(" PLACE HOLDER EMISSIONS ")
            #print(E_worldwilde_per_year_placeholder[0][t])

            #print("diference at time t = " + str(t) + " is: ")
            #print(E_worldwilde_per_year[t] - E_worldwilde_per_year_placeholder[0][t])

            #parameters are scaled with 100, check with cllimate equations
            #self.b11 = 0.012                                 #88 in excel
            #self.b23 = 0.00500                                 #0.5 in excel
            #self.b12 = 1 -  self.b11                           
            #self.b21 =  self.b11 *  self.mateq /  self.mueq    
            #self.b22 = 1 -  self.b21 -  self.b23               #good in excel       
            #self.b32 =  self.b23 *  self.mueq /  self.mleq     #good in excel
            #self.b33 = 1 -  self.b32                           #good in excel       

            #calculate concentration in bioshpere and upper oceans
            self.mu[t] = 12/100 * self.mat[t-1] + 94.796/100*self.mu[t-1] + 0.075/100 *self.ml[t-1]

            #set lower constraint for shallow ocean concentration
            if(self.mu[t] < self.mu_lo):
                self.mu[t] = self.mu_lo

            # Carbon concentration increase in lower oceans [GtC from 1750]
            self.ml[t] = 99.925/100 *self.ml[t-1]+0.5/100 * self.mu[t-1]

            #set lower constraint for shallow ocean concentration
            if(self.ml[t] < self.ml_lo):
                self.ml[t] = self.ml_lo

            #calculate concentration in atmosphere for t + 1 (because of averaging in forcing formula
            if t < 30:
                self.mat[t+1] = 88/100 * self.mat[t] + 4.704/100 * self.mu[t] + E_worldwilde_per_year_placeholder[0][t]*10

            #set lower constraint for atmospheric concentration
            if(self.mat[t] < self.mat_lo):
                self.mat[t] = self.mat_lo

            # Radiative forcing

            #Exogenous forcings from other GHG
            #rises linearly from 2010 to 2100 from -0.060 to 0.3 then becomes stable in RICE -  UPDATE FOR DICE2016R

            exo_forcing_2000 = -0.060
            exo_forcing_2100 = 0.3000

            if (t < 11):
                self.forcoth[t] = self.fex0+0.1*(exo_forcing_2100 - exo_forcing_2000 )*(t)
            else:
                self.forcoth[t] = exo_forcing_2100


            # Increase in radiative forcing [Wm-2 from 1900]
            # print('atmospheric concentration t=0')

            print('atmospheric concentration t= ' + str(t))
            print(self.mat[t])

            #print('atmospheric concentration t + 1')
            #print(self.mat[t+1])

            #print('forcing external t = ' + str(t))
            #print(self.forcoth[t])

            #forcing = constant * Log2( current concentration / concentration of forcing in 1900 at a doubling of CO2 (η)[◦C/2xCO2] ) + external forcing    
            if t < 30:
                self.forc[t] = self.fco22x*(np.log(((self.mat[t]+self.mat[t+1])/2)/(280*2.13)) / np.log(2.0)) + self.forcoth[t]
            if t == 30:
                self.forc[t] = self.fco22x*(np.log((self.mat[t])/(280*2.13)) / np.log(2.0)) + self.forcoth[t]

            #print('forcing t = 0 ')
            #print(self.forc[0])
            #print('forcing t = ' + str(t))
            #print(self.forc[t])


            """
            ####################### CLIMATE SUB-MODEL ######################
            """
            #heating of oceans and atmospheric according to matrix equations
            if t == 1:
                self.temp_atm[t] = 0.980
            if t > 1:
                self.temp_atm[t] = (self.temp_atm[t-1]+self.c1
                                    * ((self.forc[t]-((self.fco22x/self.t2xco2)* self.temp_atm[t-1]))
                                       - (self.c3*(self.temp_atm[t-1] - self.temp_ocean[t-1]))))

            #setting up lower and upper bound for temperatures
            if (self.temp_atm[t] < self.temp_atm_lo):
                self.temp_atm[t] = self.temp_atm_lo

            if (self.temp_atm[t] > self.temp_atm_up):
                self.temp_atm[t] = self.temp_atm_up

            self.temp_ocean[t] = (self.temp_ocean[t-1]+self.c4 * (self.temp_atm[t-1]-self.temp_ocean[t-1]))

            #setting up lower and upper bound for temperatures
            if (self.temp_ocean[t] < self.temp_ocean_lo):
                self.temp_ocean[t] = self.temp_ocean_lo

            if (self.temp_ocean[t] > self.temp_ocean_up):
                self.temp_ocean[t] = self.temp_ocean_up

            #print("temperature of oceans and atmosphere")
            #print("oceans")
            #print(self.temp_ocean[t])
            print("atmosphere")
            print(self.temp_atm[t])

            """
            ####################### NET ECONOMY SUB-MODEL ######################
            """

            #print(self.damage_parameters.iloc[:,0])
            #print(self.damage_parameters.iloc[:,1])
            #print(self.damage_parameters.iloc[:,2])
            #print(self.temp_atm[t])

            #original RICE parameters dam_frac
            if self.fdamage == 0:
                self.dam_frac.iloc[:,t] =  (self.damage_parameters.iloc[:,0]*self.temp_atm[t] + self.damage_parameters.iloc[:,1]*(self.temp_atm[t]**self.damage_parameters.iloc[:,2])) * 0.01
                #self.dam_frac.iloc[:,t] =  (self.damage_parameters.iloc[:,0]*self.temp_atm[t] + self.damage_parameters.iloc[:,1]*(self.temp_atm[t]**self.damage_parameters.iloc[:,2])) * 0.01

            #Damage parameters RICE2010 fitted with extra SLR component
            if self.fdamage == 1:
                self.dam_frac.iloc[:,t] = (self.damage_parameters_slr_fit.iloc[:,1]*self.temp_atm[t] 
                                           + self.damage_parameters_slr_fit.iloc[:,2]*
                                           (self.temp_atm[t]**self.damage_parameters.iloc[:,2])) * 0.01
            #self.dam_frac.iloc[:,t] = 


            print("Dam frac time t = " + str(t))
            print(self.dam_frac.iloc[:,t])

            #Determine total damages
            self.damages.iloc[:,t] = self.Y_gross.iloc[:,t]*self.dam_frac.iloc[:,t]

            #print("damages in trillion $ at time is t = " + str(t))
            #print(self.damages.iloc[:,t])

            #determine net output damages with damfrac function chosen in previous step
            self.ynet.iloc[:,t] = self.Y_gross.iloc[:,t] - self.damages.iloc[:,t]

            #print("Y net at time t = " + str(t))
            #print(self.ynet.iloc[:,t])

            # Backstop price/cback: cost of backstop                
            self.pbacktime.iloc[:,t] = 0.10 * self.cback_region + (self.pbacktime.iloc[:,t-1]- 0.1 * self.cback_region) * (1-self.decl_back_gr)

            #print(self.pbacktime.iloc[:,t])

            # Adjusted cost for backstop
            self.cost1.iloc[:,t] = ((self.pbacktime.iloc[:,t]*self.sigma_region.iloc[:,t])/self.expcost2)

            #print("adjusted cost of backstop at t =  " + str(t))
            #print(self.cost1.iloc[:,t])

            #Abettement cost ratio of output
            self.Abetement_cost_RATIO.iloc[:,t] = self.cost1.iloc[:,t]*(self.miu.iloc[:,t]** self.expcost2)

            self.Abetement_cost.iloc[:,t] = self.Y_gross.iloc[:,t] * self.Abetement_cost_RATIO.iloc[:,t]

            #print("abatement  cost in trillion $ at time t = " + str(t))
            #print(self.Abetement_cost.iloc[:,t])

            #Marginal abetement cost
            self.Mabetement_cost.iloc[:,t] = self.pbacktime.iloc[:,t] * (self.miu.iloc[:,t]**(self.expcost2-1))

            #Resulting carbon price
            #goes wrong here miu not right --> different from excel ?
            self.CPRICE.iloc[:,t] = self.pbacktime.iloc[:,t] * 1000 * (self.miu.iloc[:,t]**(self.expcost2-1))             

            #print("carbon price  at t =  " + str(t))
            #print(self.CPRICE.iloc[:,t])

            # Gross world product (net of abatement and damages)
            self.Y.iloc[:,t] = self.ynet.iloc[:,t] - abs(self.Abetement_cost.iloc[:,t])

            print("Gross product at t = " + str(t))
            print(self.Y.iloc[:,t])

            #self.Y = self.Y[self.Y < 1] = 1                

            #if self.y[t] < self.y_lo:
            #    self.y[t] = self.y_lo

            ##############  Investments & Savings  #########################
            if self.model_specification == "EMA" or "Validation_2":

                # Optimal long-run savings rate used for transversality --> SEE THESIS SHAJEE
                optlrsav = ((self.dk + 0.004) / (self.dk+ 0.004 * self.elasmu + self.irstp) * self.gama)

                if t > 12:
                    self.S.iloc[:,t] = optlrsav

                else: 
                    if t > 1:
                        self.S.iloc[:,t] = (optlrsav - self.S.iloc[:,1]) * t / 12 + self.S.iloc[:,1] 

            #investments per region given the savings rate -
            self.I.iloc[:,t] = self.S.iloc[:,t] * self.Y.iloc[:,t]

            #print("Investment at t = " + str(t))
            #print(self.I.iloc[:,t])

            i_lo = 0

            #if(self.I.iloc[:,t] < i_lo):
            #    self.I.iloc[:,t] = i_lo

            #set up constraints
            self.c_lo = 0.001
            self.CPC_lo = 0.001
            year = str(t * 10 + 2005)

            #consumption given the investments
            self.C.iloc[:,t] = self.Y.iloc[:,t] - self.I.iloc[:,t]

            #print("Consumption at t = " + str(t))
            #print(self.C.iloc[:,t])

            self.C.loc[(self.C[year] < 0), year] = self.c_lo    

            #consumption per capita
            self.CPC.iloc[:,t] = (1000 * self.C.iloc[:,t]) / self.region_pop.iloc[:,t]

            #print("Consumption CPC at t = " + str(t))
            # print(self.CPC.iloc[:,t])

            #self.CPC.loc[(self.CPC[year] < 0), year] = self.CPC_lo

            #Utility
            # Average utility social discount rate
            # irstp: Initial rate of social time preference per year
            self.util_sdr.iloc[:,t] = 1/((1+self.irstp)**(self.tstep*(t)))

            #Instantaneous utility function equation ADOPTED OF RICE
            #if self.elasmu == 1:
            #    self.inst_util.iloc[:,t] = log(self.CPC.iloc[:,t]) * self.Alpha_data.iloc[:,t]
            #else:
            #    self.inst_util.iloc[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC.iloc[:,t])**(1 - self.elasmu) + 1) * self.Alpha_data.iloc[:,t]

            #instantaneous welfare without ww
            self.inst_util.iloc[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC.iloc[:,t])**(1 - self.elasmu) + 1) 

            #period utility 
            self.per_util.iloc[:,t] = self.inst_util.iloc[:,t] * self.region_pop.iloc[:,t] * self.util_sdr.iloc[:,t]

            #print("period utility at t = " + str(t))
            #print(self.per_util.iloc[:,t])

            #cummulativie period utilty without WW
            self.cum_per_util.iloc[:,0] = self.cum_per_util.iloc[:,t-1] + self.per_util.iloc[:,t] 

            #Instantaneous utility function with welfare weights
            self.inst_util_ww.iloc[:,t] = self.inst_util.iloc[:,t] * self.Alpha_data.iloc[:,t]

            #period utility with welfare weights
            self.per_util_ww.iloc[:,t] = self.inst_util_ww.iloc[:,t] * self.region_pop.iloc[:,t] * self.util_sdr.iloc[:,t]

            print("period utility with WW at t = " + str(t))
            print(self.per_util_ww.iloc[:,t])

            #cummulative utility with ww
            self.reg_cum_util.iloc[:,t] =  self.reg_cum_util.iloc[:,t-1] + self.per_util_ww.iloc[:,t]

            print("cummulative utility per region")
            print(self.reg_cum_util.iloc[:,t])

            #scale utility with weights derived from the excel
            if t == 30:
                self.reg_util.iloc[:,t] = 10  * self.multiplutacive_scaling_weights.iloc[:,0] * self.reg_cum_util.iloc[:,t] + self.additative_scaling_weights.iloc[:,0] - self.additative_scaling_weights.iloc[:,2]  

                print("total scaled cummulative regional utility")
                print(self.reg_util.iloc[:,t])

            #calculate worldwide utility 
            self.utility = self.reg_util.sum()


            print("####################################################################")
            print("######################    NEXT STEP        #######################")
            print("####################################################################")



        """
        ####################################################################
        ###################### OUTCOME OF INTEREST #########################
        ####################################################################
        """   

        self.data = {'Atmospheric Temperature 2005': self.temp_atm[0],
                     'Damages 2005': self.damages.iloc[:,0],
                     'Industrial Emission 2005': self.Eind.iloc[:,0],
                     'Utility 2005': self.utility[0],
                     'Total Output 2005': self.Y.iloc[:,0],

                     'Atmospheric Temperature 2055': self.temp_atm[5],
                     'Damages 2055': self.damages.iloc[:,5],
                     'Industrial Emission 2055': self.Eind.iloc[:,5],
                     'Utility 2055': self.utility[5],
                     'Total Output 2055': self.Y.iloc[:,5],

                     'Atmospheric Temperature 2105': self.temp_atm[10],
                     'Damages 2105': self.damages.iloc[:,10],
                     'Industrial Emission 2105': self.Eind.iloc[:,10],
                     'Utility 2105': self.utility[10],
                     'Total Output 2105': self.Y.iloc[:,10],

                     'Atmospheric Temperature 2155': self.temp_atm[15],
                     'Damages 2155': self.damages.iloc[:,20],
                     'Industrial Emission 2155': self.Eind.iloc[:,15],
                     'Utility 2155': self.utility[15],
                     'Total Output 2155': self.Y.iloc[:,15],

                     'Atmospheric Temperature 2205': self.temp_atm[20],
                     'Damages 2205': self.damages.iloc[:,20],
                     'Industrial Emission 2205': self.Eind.iloc[:,20], 
                     'Utility 2205': self.utility[20],
                     'Total Output 2205': self.Y.iloc[:,20],

                     'Atmospheric Temperature 2305': self.temp_atm[30],
                     'Damages 2305': self.damages.iloc[:,30],
                     'Industrial Emission 2305': self.Eind.iloc[:,30],
                     'Utility 2305': self.utility[30],
                     'Total Output 2305': self.Y.iloc[:,30]}
        return self.data