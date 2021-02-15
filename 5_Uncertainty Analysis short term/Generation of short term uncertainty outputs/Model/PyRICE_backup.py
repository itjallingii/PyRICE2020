import numpy as np
import pandas as pd
import math
from scipy.stats import norm, skewnorm, cauchy, lognorm
import logging
import json
import sys
import os

pydice_folder = os.path.dirname(os.getcwd())

print('local path in PyRICE = ')
print(pydice_folder)

sys.path.append(pydice_folder)


class PyRICE(object):
    """ RICE simulation model:
        tstep: time step/interval
        steps: amount of years looking into the future
        model_specification: model specification for 'Validation_1'(RICE2010 replicating), 
                                                     'Validation_2' (RICE2010 Deterministic) or 'EMA'  
    """
    def __init__(self, tstep=10, steps=31, model_specification="EMA",fdamage = 0, welfare_function="utilitarian",overwrite_f = True):
        self.tstep = tstep # (in years)
        self.steps = steps
        self.tperiod = []
        self.startYear = 2005
        self.model_specification = model_specification
        self.fdamage = fdamage #0 RICE Damage function #1
        self.welfare_function = welfare_function
        self.overwrite_fdamage = overwrite_f
        
        ########################## SAMPLING OF DAMAGE FUNCTIONS ##########################
        

        #arrange simulation timeline
        for i in range(0, self.steps):
            self.tperiod.append((i*self.tstep)+self.startYear)

        #setup of json file to store model results
        with open(pydice_folder + '\\ecs_dist_v5.json') as f:
            d=json.load(f)

        #setting up three distributions for the climate sensitivity; normal lognormal and gauchy
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
        samples_norm = np.append(samples_norm, 3.2)
        samples_lognorm = np.append(samples_lognorm, 3.2)
        samples_cauchy = np.append(samples_cauchy, 3.2)

        self.samples_t2xco2 = [samples_norm, samples_lognorm, samples_cauchy]


    def __call__(self,
        ###### alternative principles CONTROLS ########          
        #prioritarian controls 
        growth_factor_prio = 1,       #how much the worst-off consumption needs to grow each timestep to allow discounting
        prioritarian_discounting = 0,    #0 = no discounting or 1 = conditional_growth

        #sufficitarian controls
        sufficitarian_discounting = 1,   #0 = inheritance discounting , 1 = sustainable growth discounting, 
        growth_factor_suf = 1,
        ini_suf_treshold = 0.711,        #based on the poverty line of 1.95 dollar per day 
         
        #egalitarian controls
        egalitarian_discounting = 0,     #0 = no discouting , 1 = normal discounting,
        
        ###### RICE and SSP uncertainties ########          

        #uncertainties for climate components
        t2xco2_index = -1,                     #base RICE2010
        t2xco2_dist = 0,                       #base RICE2010
        fosslim =6000,                         #base RICE2010
        fdamage=0,                             #base RICE2010

        #SSP socio-economic uncertainties
        scenario_pop_gdp = 0,            #base RICE2010 scenario or ssp1 = 1 etc
        scenario_sigma = 0,              #base RICE2010 scenario or ssp1 = 1 etc
        scenario_cback = 0,              #base RICE2010 scenario or ssp1 = 1 etc
        scenario_elasticity_of_damages = 0,    #base RICE2010
        scenario_limmiu = 0,              # 0:base RICE or 1: negative emissions possible 
                 
        #additional scenario parameters for long run RICE uncertainty analysis
        #long run scenario check         
        longrun_scenario = 0,            #0: long run uncertainty switch off #1: switched on
        
        #uncertainty range for RICE inputs over long terms (seperate from SSP)
        long_run_nordhaus_tfp_gr = 1,    #range in DICE [0.07, 0.09] in RICE 0.85 - 1.15
        long_run_nordhaus_sigma = 1,     #range in DICE [-0.012, -0.008] 0.75 - 1.25
        long_run_nordhaus_pop_gr = 1,    #range in DICE [0.1 0.15]   0.75 - 1.25
        
        #controls
        sr = 0.248,              # BASE sr for the world                            
        miu_period=13,           # 2155 in RICE opt scenario when global emissions are near zero
        #limmiu=1,               # Upper limit on control rate after 2150, in RICE 1 in DICE 1.2 REPLACED WITH SCENARIO
        irstp = 0.015,           # Initial rate of social time preference  (RICE2010 OPT))     
                
        **kwargs):

        """
        ######################## INITIALIZE DATA IMPORTS ########################
        """    
        
        #Load in RICE input paramaters for all regions
        RICE_DATA = pd.read_excel("RICE_data.xlsx")
        RICE_PARAMETER = pd.read_excel("RICE_parameter.xlsx")
        RICE_input = pd.read_excel("input_data_RICE.xlsx")
        RICE_regional_damage_factor = pd.read_csv("regional damage frac factor RICE.csv")
        self.RICE_regional_damage_factor = RICE_regional_damage_factor.iloc[:,1:].to_numpy()
        
        #import World Bank income shares
        RICE_income_shares = pd.read_excel("RICE_income_shares.xlsx")
        self.RICE_income_shares = RICE_income_shares.iloc[:,1:6].to_numpy()

        #import dataframes for SSP (IPCC) uncertainty analysis
        self.RICE_GDP_SSP = pd.read_excel("Y_Gross_ssp.xlsx").to_numpy()
        POP_ssp = pd.read_excel("pop_ssp.xlsx")
        self.POP_ssp = POP_ssp.iloc[1:,:]        

        regions_list = ["US", "OECD-Europe","Japan","Russia","Non-Russia Eurasia","China","India","Middle East","Africa",
        "Latin America","OHI","Other non-OECD Asia"]
            
        """
        ############################# LEVERS ###############################
        """
        #Get controls from RICE optimal run
        miu_opt_series = RICE_input.iloc[15:27,1:].to_numpy()
        sr_opt_series = RICE_input.iloc[30:42,1:].to_numpy()
        
        #Controls with EMA sampling
        if self.model_specification == "EMA":
            
            #create frame for savings rate to be sampled
            self.S = np.zeros((12, self.steps))
            self.miu = np.zeros((12,self.steps))
            
            #set starting MIU for all runs
            self.miu[:,0:2] = miu_opt_series[:,0:2]
            self.S[:,0:2] = sr_opt_series[:,0:2]
            
            self.miu_period = np.full((12, 1), miu_period)
            self.sr = sr  
                

        #Get control from RICE2010 - full RICE2010 replicating run
        if self.model_specification == "Validation_1":
 
            #set savings rate and control rate as optimized RICE 2010          
            self.S =  sr_opt_series 
        
            #set emission control rate for the whole run according to RICE2010 opt.
            self.miu = miu_opt_series
            self.irstp = irstp
            

        #EMA Deterministic controls
        if self.model_specification == "Validation_2":

            #create dataframes for control rate and savings rate
            self.miu = np.zeros((12,self.steps))
            self.S = np.zeros((12, self.steps))
            
            #set savings rate and control rate as optimized RICE 2010 for the first two timesteps
            self.miu[:,0:2] = miu_opt_series[:,0:2]
            self.S[:,0:2] = sr_opt_series[:,0:2]

            #set uncertainties that drive MIU
            self.limmiu= 1
            self.irstp = irstp
            self.miu_period = [12,15,15,10,10,11,13,13,13,14,13,14]
        
        #define other uncertainties 
        self.irstp = irstp
        self.fosslim = fosslim
        
        #SSP CCS and negative emissions possibilities        
        if scenario_limmiu == 0:
             self.limmiu = 1
        
        if scenario_limmiu == 1:
             self.limmiu = 1.2
                
        #Uncertainty of elasticity of damages to consumption  
        if scenario_elasticity_of_damages == 0:
            elasticity_of_damages = 1
        
        if  scenario_elasticity_of_damages == 1:
             elasticity_of_damages = 0
        
        if  scenario_elasticity_of_damages == 2:
             elasticity_of_damages = -1
  
        #overwrite IRSTP for non discounting levers
        if self.welfare_function == "prioritarian":
            if prioritarian_discounting == 0:
                self.irstp = 0
                
        if self.welfare_function == "egalitarian":
            if egalitarian_discounting == 0:
                self.irstp = 0  
        
        """
        ######################## DEEP UNCERTAINTIES ########################
        """

        # Equilibrium temperature impact [dC per doubling CO2]/(3.2 RICE OPT)
        self.t2xco2 = self.samples_t2xco2[t2xco2_dist][t2xco2_index]

        # Choice of the damage function (structural deep uncertainty) #derived from Lingerewan (2020
        if self.overwrite_fdamage == True:
            self.fdamage = fdamage
        
        print("damage function used:" + str(self.fdamage))
        """
        ######################## SOCIO-ECONONMIC UNCERTAINTIES FROM SSPs ########################
        """
        #define growth factor uncertainties for sampling
        self.scenario_pop_gdp =scenario_pop_gdp
        self.scenario_sigma = scenario_sigma
        self.scenario_cback = scenario_cback

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

        self.b11 = 0.088                                   
        self.b23 = 0.00500                               
        self.b12 = 1 -  self.b11                           
        self.b21 =  self.b11 *  self.mateq /  self.mueq    
        self.b22 = 1 -  self.b21 -  self.b23                     
        self.b32 =  self.b23 *  self.mueq /  self.mleq    
        self.b33 = 1 -  self.b32                                 

        # 2000 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
        self.fex0 = -0.06
        # 2100 forcings of non-CO2 GHG [Wm-2]
        self.fex1 = 0.30
        # Forcings of equilibrium CO2 doubling [Wm-2]
        self.fco22x = 3.8


        """
        ###################### CLIMATE INITIAL VALUES ######################
        """
        # Initial lower stratum temperature change [dC from 1900]
        self.tocean0 = 0.0068 
        # Initial atmospheric temperature change [dC from 1900]
        self.tatm0 = 0.83 

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
        self.temp_atm = np.zeros((self.steps))
        # Increase temperature of lower oceans [dC from 1900]
        self.temp_ocean = np.zeros((self.steps))


        """
        ######################## DAMAGE FUNCTION PARAMETERS ########################
        """
        #damage parameters excluding SLR from RICE2010 
        self.damage_parameters =  RICE_input.iloc[47:55,1:13]
        self.damage_parameters = self.damage_parameters.transpose().to_numpy()

        #damage parameters INCLUDING SLR FIT Dennig et 
        self.damage_parameters_slr_fit =  RICE_input.iloc[61:73,1:3]
        self.damage_parameters_slr_fit = self.damage_parameters_slr_fit.to_numpy()
        self.dam_frac_global = np.zeros((self.steps))

        """
        ####################### Capital and Economic PARAMETERS #######################
        """
        #Population parameters
        self.region_pop_gr = RICE_input.iloc[0:12,1:].to_numpy()
        
        #Get population data for 2005
        population2005 = RICE_DATA.iloc[19:31,0].to_numpy()
        self.region_pop = np.zeros((12,self.steps))

        #get regional series for factor productivity growth
        self.tfpgr_region =  RICE_DATA.iloc[52:64,1:32].to_numpy()

        #get initial values for various parameters
        self.initails_par = RICE_PARAMETER.iloc[33:40,5:17].to_numpy()
        self.initials_par = self.initails_par.transpose()

        #setting up total factor productivity
        self.tfp_2005 = self.initials_par[:,5]
        self.tfp_region = np.zeros((12, self.steps))
        
        #setting up Capital Stock parameters
        self.k_2005 = self.initials_par[:,4]
        self.k_region = np.zeros((12, self.steps))
        self.dk = 0.1
        self.gama = 0.3

        #setting up total output dataframes
        self.Y_gross = np.zeros((12, self.steps))
        self.ynet = np.zeros((12, self.steps))
        self.damages = np.zeros((12, self.steps))
        self.dam_frac = np.zeros((12, self.steps))

        #Dataframes for emissions, economy and utility
        self.Eind = np.zeros((12, self.steps))
        self.E = np.zeros((12, self.steps))
        self.Etree = np.zeros((12, self.steps))
        self.cumetree = np.zeros((12, self.steps))
        self.CCA = np.zeros((12, self.steps))
        self.CCA_tot = np.zeros((12, self.steps))
        self.Abetement_cost = np.zeros((12, self.steps))
        self.Abetement_cost_RATIO = np.zeros((12, self.steps))
        self.Mabetement_cost = np.zeros((12, self.steps))
        self.CPRICE =np.zeros((12, self.steps))

        #economy parameters per region
        self.Y = np.zeros((12, self.steps))
        self.I = np.zeros((12, self.steps))
        self.C = np.zeros((12, self.steps))
        self.CPC = np.zeros((12, self.steps))
        
        """
        ####################### Utility and alternative principle output #######################
        """
        
        #output metrics
        self.util_sdr = np.zeros((12, self.steps))
        self.inst_util = np.zeros((12, self.steps))
        self.per_util = np.zeros((12, self.steps))
        
        self.cum_util = np.zeros((12, self.steps))
        self.reg_cum_util = np.zeros((12, self.steps))
        self.reg_util = np.zeros((12))
        self.util = np.zeros((12, self.steps))

        self.per_util_ww =  np.zeros((12, self.steps))
        self.cum_per_util = np.zeros((12, self.steps))
        self.inst_util_ww = np.zeros((12, self.steps))
        
        #alternative SWF output arrays
        self.sufficitarian_treshold = np.zeros((self.steps))
        self.inst_util_tres = np.zeros((self.steps))
        self.inst_util_tres_ww = np.zeros((12,self.steps))

        #Output-to-Emission
        #Change in sigma: the cumulative improvement in energy efficiency)
        self.sigma_growth_data = RICE_DATA.iloc[70:82,1:6].to_numpy()
        self.Emissions_parameter = RICE_PARAMETER.iloc[65:70,5:17].to_numpy().transpose()

        #set up dataframe for saving CO2 to output ratio
        self.Sigma_gr = np.zeros((12, self.steps))
        
        #CO2-equivalent-emissions growth to output ratio in 2005
        self.Sigma_gr[:,0] = self.sigma_growth_data[:,0]
        
        #RICE asssumes full participation therefore 1 here, optional parameter
        
        #Period at which have full participation
        #self.periodfullpart = periodfullpart 

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

        # Emission data
        self.emission_factor = RICE_DATA.iloc[87:99,6].to_numpy()
        self.Eland0 = 1.6 #(RICE2010 OPT)

        # Get scaling data for welfare weights and aggregated utility
        self.Alpha_data = RICE_DATA.iloc[357:369,1:60].to_numpy()
        self.additative_scaling_weights = RICE_DATA.iloc[167:179,14:17].to_numpy()
        self.multiplutacive_scaling_weights = RICE_DATA.iloc[232:244,1:2].to_numpy() / 1000

        # Cost of abatement
        self.abatement_data = RICE_PARAMETER.iloc[56:60,5:17].to_numpy().transpose()
        self.pbacktime = np.zeros((12, self.steps))
        self.cost1 =  np.zeros((12, self.steps))

        # CO2 to economy ratio
        self.sigma_region =  np.zeros((12, self.steps))
        self.sigma_region[:,0] = self.Emissions_parameter[:,2] 

        # Cback per region
        ratio_backstop_world = np.array(([0.9,1.4,1.4,0.6,0.6,0.7,1.1,1.0,1.1,1.3,1.1,1.2]))
        
        if scenario_cback == 0:           #SSP LOW SCENARIO
            cback = 1.260
        
        if scenario_cback == 1:            #SSP HIGH SCENARIO
            cback = 1.260 * 1.5
            
        self.cback_region = cback * ratio_backstop_world
               
        # Constations for backstop costs
        self.ratio_asymptotic = self.abatement_data[:,2]
        self.decl_back_gr = self.abatement_data[:,3]
        self.expcost2 = 2.8   #RICE 2010 OPT
        
        # Disaggregated consumption tallys
        self.CPC_post_damage = {}
        self.CPC_pre_damage = {}
        self.pre_damage_total__region_consumption = np.zeros((12, self.steps))
        
        # Dictionaries for quintile outputs
        self.quintile_inst_util = {}
        self.quintile_inst_util_ww = {}
        self.quintile_inst_util_concave = {}
        self.quintile_per_util_ww = {}
        
        # Utilitarian outputs
        self.global_damages = np.zeros((self.steps))
        self.global_ouput = np.zeros((self.steps))
        self.global_per_util_ww = np.zeros((self.steps))
        self.regional_cum_util = np.zeros((self.steps))
        
        # Prioritarian outputs
        self.inst_util_worst_off = np.zeros((12,self.steps))
        self.inst_util_worst_off_condition = np.zeros((12,self.steps))
        self.worst_off_income_class = np.zeros((self.steps))
        self.worst_off_income_class_index = np.zeros((self.steps))
        self.worst_off_climate_impact = np.zeros((self.steps))
        self.worst_off_climate_impact_index = np.zeros((self.steps))
        self.climate_impact_relative_to_capita = {}
        
        # Sufficitarian outputs
        self.average_world_CPC = np.zeros((self.steps))
        self.average_growth_CPC = np.zeros((self.steps))
        self.sufficitarian_treshold = np.zeros((self.steps))
        self.inst_util_tres = np.zeros((self.steps))
        self.inst_util_tres_ww = np.zeros((12,self.steps))
        self.quintile_inst_util = {}
        self.quintile_inst_util_ww = {}
        self.population_under_treshold = np.zeros((self.steps))
        self.utility_distance_treshold = np.zeros((12,self.steps))
        self.max_utility_distance_treshold = np.zeros((self.steps))
        self.regions_under_treshold = [None] * self.steps
        self.largest_distance_under_treshold = np.zeros((self.steps))
        self.growth_frontier= np.zeros((self.steps))

        # Egalitarian outputs
        self.CPC_intra_gini = np.zeros((self.steps))
        self.average_world_CPC = np.zeros((self.steps))
        self.average_regional_impact = np.zeros((self.steps))
        self.climate_impact_per_dollar_consumption = np.zeros((12,self.steps))
        self.climate_impact_per_dollar_gini = np.zeros((self.steps))
        
        """
        ####################### LIMITS OF THE MODEL ########################
        """

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
        self.temp_atm[1] = 0.980

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
        ################# SLR PARAMETER INTITIALISATION ################
        """
        
        #define inputs
        self.SLRTHERM = np.zeros((31))
        self.THERMEQUIL = np.zeros((31))

        self.GSICREMAIN = np.zeros((31))
        self.GSICCUM = np.zeros((31))
        self.GSICMELTRATE = np.zeros((31))
        self.GISREMAIN = np.zeros((31))
        self.GISMELTRATE = np.zeros((31))
        self.GISEXPONENT = np.zeros((31))
        self.GISCUM = np.zeros((31))
        self.AISREMAIN = np.zeros((31))
        self.AISMELTRATE = np.zeros((31))
        self.AISCUM = np.zeros((31))
        self.TOTALSLR = np.zeros((31))

        #inputs
        self.therm0 = 0.092066694
        self.thermadj = 0.024076141
        self.thermeq = 0.5

        self.gsictotal = 0.26
        self.gsicmelt= 0.0008
        self.gsicexp = 1
        self.gsieq = -1

        self.gis0 = 7.3
        self.gismelt0 = 0.6
        self.gismeltabove = 1.118600816
        self.gismineq = 0
        self.gisexp = 1

        self.aismelt0 = 0.21
        self.aismeltlow = -0.600407185
        self.aismeltup = 2.225420209
        self.aisratio = 1.3
        self.aisinflection = 0
        self.aisintercept = 0.770332789
        self.aiswais = 5
        self.aisother = 51.6
       
        self.THERMEQUIL[0] = self.temp_atm[0] * self.thermeq
        self.SLRTHERM[0] = self.therm0 + self.thermadj * (self.THERMEQUIL[0] - self.therm0)

        self.GSICREMAIN[0] = self.gsictotal

        self.GSICMELTRATE[0] = self.gsicmelt * 10 * (self.GSICREMAIN[0] / self.gsictotal)**(self.gsicexp) * (self.temp_atm[0] - self.gsieq )
        self.GSICCUM[0] = self.GSICMELTRATE[0] 
        self.GISREMAIN[0] = self.gis0
        self.GISMELTRATE[0] = self.gismelt0
        self.GISCUM[0] = self.gismelt0 / 100
        self.GISEXPONENT[0] = 1
        self.AISREMAIN[0] = self.aiswais + self.aisother
        self.AISMELTRATE[0] = 0.1225
        self.AISCUM[0] = self.AISMELTRATE[0] / 100

        self.TOTALSLR[0] = self.SLRTHERM[0] + self.GSICCUM[0] + self.GISCUM[0] + self.AISCUM[0]
        
        self.slrmultiplier = 2
        self.slrelasticity = 4

        self.SLRDAMAGES = np.zeros((12,self.steps))
        self.slrdamlinear = np.array([0,0.00452, 0.00053 ,0, 0.00011 , 0.01172 ,0, 0.00138 , 0.00351, 0, 0.00616,0])
        self.slrdamquadratic = np.array([0.000255,0,0.000053,0.000042,0,0.000001,0.000255,0,0,0.000071,0,0.001239])
        
        self.SLRDAMAGES[:,0] = 0

        """
        ################# ECONOMIC PARAMETER INTITIALISATION ################
        """

        #Insert population at 2005 for all regions
        self.region_pop[:,0] = population2005

        #total factor production at 2005
        self.tfp_region[:,0] = self.tfp_2005

        #initial capital in 2005
        self.k_region[:,0] = self.k_2005
        
        # Gama: Capital elasticity in production function
        self.Y_gross[:,0] = (self.tfp_region[:,0]*((self.region_pop[:,0]/1000)**(1-self.gama)) * (self.k_region[:,0]**self.gama))
        
        #original RICE parameters dam_frac with SLR
        if self.fdamage == 0:
            self.dam_frac[:,0] =  (self.damage_parameters[:,0]*self.temp_atm[0] 
                            + self.damage_parameters[:,1]*(self.temp_atm[0]**self.damage_parameters[:,2])) * 0.01

        #Damage function Newbold & Daigneault
        elif self.fdamage == 1:
            self.dam_frac_global[0] = (1-(np.exp(-0.0025*self.temp_atm[0]*2.45)))
            
            #translate global damage frac to regional damage frac with factor as used in RICE
            self.dam_frac[:,0] = self.dam_frac_global[0] * self.RICE_regional_damage_factor[:,0]
                        
        #Damage function Weitzman
        elif self.fdamage == 2:
            self.dam_frac_global[0] = (1-1/(1+0.0028388**2+0.0000050703*(self.temp_atm[0]*6.754)))
            
            #translate global damage frac to regional damage frac with factor as used in RICE
            self.dam_frac[:,0] = self.dam_frac_global[0] * self.RICE_regional_damage_factor[:,0]

        #Net output damages
        self.ynet[:,0] = self.Y_gross[:,0]/(1.0+self.dam_frac[:,0])

        #Damages in 2005
        self.damages[:,0] = self.Y_gross[:,0] - self.ynet[:,0]

        #Cost of backstop
        self.pbacktime[:,0] = self.cback_region

        # Adjusted cost for backstop
        self.cost1[:,0] = self.pbacktime[:,0]*self.sigma_region[:,0]/self.expcost2

        #decline of backstop competitive year (RICE2010 OPT)
        self.backstopcompetitiveyear = 2250

        #Emissions from land change use
        self.Etree[:,0] = self.Emissions_parameter[:,3]
        self.cumetree[:,0] = self.Emissions_parameter[:,3]

        #industrial emissions 2005
        self.Eind[:,0] =  self.sigma_region[:,0] * self.Y_gross[:,0] * (1 - self.miu[:,0])

        #initialize initial emissions
        self.E[:,0] = self.Eind[:,0] + self.Etree[:,0]
        self.CCA[:,0] = self.Eind[:,0]
        self.CCA_tot[:,0] = self.CCA[:,0] + self.cumetree[:,0]

        #doesnt do much here
        self.partfract = 1 


        """
        ####################### INIT NET ECONOMY SUB-MODEL ######################
        """                   

        #Cost of climate change to economy
        #Abettement cost ratio of output
        self.Abetement_cost_RATIO[:,0] = self.cost1[:,0]*(self.miu[:,0] ** self.expcost2)

        #Abettement cost total
        self.Abetement_cost[:,0] = self.Y_gross[:,0] * self.Abetement_cost_RATIO[:,0]

        #Marginal abetement cost
        self.Mabetement_cost[:,0] = self.pbacktime[:,0] * self.miu[:,0]**(self.expcost2-1)

        #Resulting carbon price
        self.CPRICE[:,0] = self.pbacktime[:,0] * 1000 * (self.miu[:,0]**(self.expcost2-1))     

        # Gross world product (net of abatement and damages)
        self.Y[:,0] = self.ynet[:,0]-self.Abetement_cost[:,0]           

        ##############  Investments & Savings  #########################

        #investments per region given the savings rate 
        self.I[:,0] = self.S[:,0] * self.Y[:,0]
        
        #consumption given the investments
        self.C[:,0] = self.Y[:,0] - self.I[:,0]
        
        #placeholder for different damagefactor per quintile
        quintile_damage_factor = 1

        #calculate pre damage consumption aggregated per region
        self.pre_damage_total__region_consumption[:,0] = self.C[:,0] + self.damages[:,0]

        #damage share elasticity function derived from Denig et al 2015
        self.damage_share = self.RICE_income_shares**elasticity_of_damages
        sum_damage = np.sum(self.damage_share,axis=1)

        for i in range(0,12):
            self.damage_share[i,:] = self.damage_share[i,:]/sum_damage[i]    

        #calculate disaggregated per capita consumption based on income shares BEFORE damages
        self.CPC_pre_damage[2005] = ((self.pre_damage_total__region_consumption[:,0] * self.RICE_income_shares.transpose() )  / (self.region_pop[:,0] * (1 / 5))) * 1000

        #calculate disaggregated per capita consumption based on income shares AFTER damages
        self.CPC_post_damage[2005] = self.CPC_pre_damage[2005]  - (((self.damages[:,0] *  self.damage_share.transpose() ) / (self.region_pop[:,0] * (1 / 5))) * 1000)
        
        #calculate damage per consumpion in thousands of US dollarsa
        self.climate_impact_relative_to_capita[2005] = ((self.damages[:,0] *  self.damage_share.transpose() * 10**12)/(0.2* self.region_pop[:,0]*10**6)) /(self.CPC_post_damage[2005] * 1000)

        #consumption per capita
        self.CPC[:,0] = (1000 * self.C[:,0]) / self.region_pop[:,0]

        ######################################### Utility #########################################

        #Initial rate of social time preference per year
        self.util_sdr[:,0] = 1

        #Instantaneous utility function equation 
        self.inst_util[:,0] = ((1 / (1 - self.elasmu)) * (self.CPC[:,0])**(1 - self.elasmu) + 1)        

        #CEMU period utilitity         
        self.per_util[:,0] = self.inst_util[:,0] * self.region_pop[:,0] * self.util_sdr[:,0]

        #Cummulativie period utilty without WW
        self.cum_per_util[:,0] = self.per_util[:,0] 

        #Instantaneous utility function with welfare weights
        self.inst_util_ww[:,0] = self.inst_util[:,0] * self.Alpha_data[:,0]

        #Period utility with welfare weights
        self.per_util_ww[:,0] = self.inst_util_ww[:,0] * self.region_pop[:,0] * self.util_sdr[:,0]

        #cummulative utility with ww
        self.reg_cum_util[:,0] =  self.per_util[:,0] 
        self.global_per_util_ww[0] = self.per_util_ww[:,0].sum(axis = 0) 
        
        #initialise objectives for principles
        #objective for the worst-off region in terms of consumption per capita
        self.worst_off_income_class[0] = self.CPC_post_damage[2005][0].min()

        array_worst_off_income = self.CPC_post_damage[2005][0]
        self.worst_off_income_class_index[0] = np.argmin(array_worst_off_income)

        #objective for the worst-off region in terms of climate impact
        self.worst_off_climate_impact[0] = self.climate_impact_relative_to_capita[2005][0].min()

        array_worst_off_share = self.climate_impact_relative_to_capita[2005][0]
        self.worst_off_climate_impact_index[0] = np.argmin(array_worst_off_share)

        #objectives sufficitarian
        self.average_world_CPC[0] = self.CPC[:,0].sum() / 12
        self.average_growth_CPC[0] = 0.250 #average growth over 10 years World Bank Data
        
        #calculate instantaneous welfare equivalent of minimum capita per head 
        self.sufficitarian_treshold[0] = ini_suf_treshold  #specified in consumption per capita thousand/year 

        self.inst_util_tres[0] = ((1 / (1 - self.elasmu)) * (self.sufficitarian_treshold[0])**(1 - self.elasmu) + 1) 

        #calculate instantaneous welfare equivalent of minimum capita per head with PPP
        self.inst_util_tres_ww[:,0] = self.inst_util_tres[0] * self.Alpha_data[:,0]

        #calculate utility equivalent for every income quintile and scale with welfare weights for comparison
        self.quintile_inst_util[2005] = ((1 / (1 - self.elasmu)) * (self.CPC_post_damage[2005])**(1 - self.elasmu) + 1)
        self.quintile_inst_util_ww[2005] = self.quintile_inst_util[2005] * self.Alpha_data[:,0]       

        utility_per_income_share = self.quintile_inst_util_ww[2005]

        list_timestep = []

        for quintile in range(0,5):
            for region in range(0,12):
                if utility_per_income_share[quintile,region] < self.inst_util_tres_ww[region,0]:                            
                    self.population_under_treshold[0] = self.population_under_treshold[0] + self.region_pop[region,0] * 1/5
                    self.utility_distance_treshold[region,0] = self.inst_util_tres_ww[region,0]-utility_per_income_share[quintile,region]

                    list_timestep.append(regions_list[region])    

        self.regions_under_treshold[0]= list_timestep
        self.max_utility_distance_treshold[0] = self.utility_distance_treshold[:,0].max()       

        #calculate gini as measure of current inequality in consumption (intragenerational)
        input_gini = self.CPC[:,0]

        diffsum = 0
        for i, xi in enumerate(input_gini[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini[i:]))

        self.CPC_intra_gini[0] = diffsum / ((len(input_gini)**2)* np.mean(input_gini))


        #calculate gini as measure of current inequality in climate impact (per dollar consumption)  (intragenerational
        self.climate_impact_per_dollar_consumption[:,0] = self.damages[:,0] / self.CPC[:,0]

        input_gini = self.climate_impact_per_dollar_consumption[:,0]

        diffsum = 0
        for i, xi in enumerate(input_gini[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini[i:]))

        self.climate_impact_per_dollar_gini[0] = diffsum / ((len(input_gini)**2)* np.mean(input_gini))
        

        """
        ########################################## RICE MODEL ###################################################    
        """    

        
        #Follows equations of notes #TOTAL OF 30 STEPS UNTIL 2305
        for t in range(1,31): 
            
            #keep track of year per timestep for dicts used
            year = 2005 + 10 * t

            """
            ####################### GROSS ECONOMY SUB-MODEL ######################
            """
            
            #use ssp population projections if not base with right SSP scenario (SSP1, SSP2 etc.)
            if scenario_pop_gdp !=0:
                print("Shared socio economic pathways are used")                
                #load population and gdp projections from SSP scenarios on first timestep
                if t == 1:
                    for region in range(0,12):
                        self.region_pop[region,:] = POP_ssp.iloc[:,scenario_pop_gdp + (region * 5)]
                      
                        self.Y_gross[region,:] = RICE_GDP_SSP.iloc[:,scenario_pop_gdp + (region * 5)] / 1000
                
                self.Y_gross[:,t] = np.where(self.Y_gross[:,t]  > 0, self.Y_gross[:,t], 0)
                
                self.k_region[:,t] = self.k_region[:,t-1]*((1-self.dk)**self.tstep) + self.tstep*self.I[:,t-1]
                
                #calculate tfp based on GDP projections by SSP's
                self.tfp_region[:,t] = self.Y_gross[:,t] / ((self.k_region[:,t]**self.gama)*(self.region_pop[:,t]/1000)**(1-self.gama))
                                
            #Use base projections for population and TFP and sigma growth 
            if scenario_pop_gdp == 0 and longrun_scenario == 0:
                
                print("RICE Reference scenario is used")
                
                #calculate population at time t
                self.region_pop[:,t] = self.region_pop[:,t-1] *  2.71828 **(self.region_pop_gr[:,t]*10)

                #TOTAL FACTOR PRODUCTIVITY level according to RICE base
                self.tfp_region[:,t] = self.tfp_region[:,t-1] * 2.71828 **(self.tfpgr_region[:,t]*10)
                                                        
                #determine capital stock at time t
                self.k_region[:,t] = self.k_region[:,t-1]*((1-self.dk)**self.tstep) + self.tstep*self.I[:,t-1]

                #lower bound capital
                self.k_region[:,t] = np.where(self.k_region[:,t]  > 1, self.k_region[:,t] ,1)

                #determine Ygross at time t
                self.Y_gross[:,t] = self.tfp_region[:,t] * ((self.region_pop[:,t]/1000)**(1-self.gama))*(self.k_region[:,t]**self.gama)   
                
                #lower bound Y_Gross
                self.Y_gross[:,t] = np.where(self.Y_gross[:,t]  > 0, self.Y_gross[:,t], 0)
            
            #LONG RUN EXPLORATORY ANALAYSIS USING RICE REFERENCE SCENARIO
            if longrun_scenario == 1:
                print("RICE long run scenario is used")

                #calculate population at time t adjust with uncertainty range 
                self.region_pop[:,t] = self.region_pop[:,t-1] *  2.71828 **(self.region_pop_gr[:,t]*long_run_nordhaus_pop_gr*10)

                #TOTAL FACTOR PRODUCTIVITY level according to RICE base adjust with uncertainty range
                self.tfp_region[:,t] = self.tfp_region[:,t-1] * 2.71828 **(self.tfpgr_region[:,t]*long_run_nordhaus_tfp_gr*10)
                                                        
                #determine capital stock at time t
                self.k_region[:,t] = self.k_region[:,t-1]*((1-self.dk)**self.tstep) + self.tstep*self.I[:,t-1]

                #lower bound capital
                self.k_region[:,t] = np.where(self.k_region[:,t]  > 1, self.k_region[:,t] ,1)

                #determine Ygross at time t
                self.Y_gross[:,t] = self.tfp_region[:,t] * ((self.region_pop[:,t]/1000)**(1-self.gama))*(self.k_region[:,t]**self.gama)   
                
                #lower bound Y_Gross
                self.Y_gross[:,t] = np.where(self.Y_gross[:,t]  > 0, self.Y_gross[:,t], 0)
                
                #calculate the sigma growth adjust with uncertainty range and the emission rate development
                if t == 1:
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.sigma_growth_data[:,2] - self.sigma_growth_data[:,4] )) 

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** (self.Sigma_gr[:,t]*10)) * self.emission_factor

                if t > 1 :
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.Sigma_gr[:,t-1] - self.sigma_growth_data[:,4]  ) * (1-self.sigma_growth_data[:,3] ))* long_run_nordhaus_sigma
                    
                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** ( self.Sigma_gr[:,t]*10))
                        
            print('ygross at t = ' + str(t))
            print(self.Y_gross[:,t])
            
            if scenario_sigma == 0:   #medium SSP AEEI (base RICE)
                #calculate the sigma growth and the emission rate development
                if t == 1:
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.sigma_growth_data[:,2] - self.sigma_growth_data[:,4] )) 

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** (self.Sigma_gr[:,t]*10)) * self.emission_factor

                if t > 1 :
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.Sigma_gr[:,t-1] - self.sigma_growth_data[:,4]  ) * (1-self.sigma_growth_data[:,3] )) 

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** ( self.Sigma_gr[:,t]*10)) 
            
            if scenario_sigma == 1:   #low SSP AEEI
                #calculate the sigma growth and the emission rate development
                if t == 1:
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.sigma_growth_data[:,2] - self.sigma_growth_data[:,4] )) 

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** (self.Sigma_gr[:,t]*10)) * self.emission_factor

                if t > 1 :
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.Sigma_gr[:,t-1] - self.sigma_growth_data[:,4]  ) * (1-self.sigma_growth_data[:,3] )) * 0.5

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** ( self.Sigma_gr[:,t]*10)) 
            
            if scenario_sigma == 2:   #high SSP AEEI 
                #calculate the sigma growth and the emission rate development
                if t == 1:
                    self.Sigma_gr[:,t] = ((self.sigma_growth_data[:,4] + (self.sigma_growth_data[:,2] - self.sigma_growth_data[:,4] ))) 

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** (self.Sigma_gr[:,t]*10)) * self.emission_factor

                if t > 1 :
                    self.Sigma_gr[:,t] = (self.sigma_growth_data[:,4] + (self.Sigma_gr[:,t-1] - self.sigma_growth_data[:,4]  ) * (1-self.sigma_growth_data[:,3] )) * 1.50

                    self.sigma_region[:,t] = self.sigma_region[:,t-1] *  (2.71828 ** ( self.Sigma_gr[:,t]*10)) 
            
            print('ygross at t = ' + str(t))
            print(self.Y_gross[:,t])
                
            #calculate emission control rate under EMA
            if self.model_specification == "EMA":
                # control rate is maximum after target period, otherwise linearly increase towards that point from t[0]
                if t > 1:
                        for index in range(0,12):            
                            calculated_miu = self.miu[index,t-1] + (self.limmiu - self.miu[index,1]) / self.miu_period[index]
                            self.miu[index, t]= min(calculated_miu, self.limmiu)

            if self.model_specification == "Validation_2": 
                if t > 1:
                    for index in range(0,12):            
                        calculated_miu = self.miu[index,t-1] + (self.limmiu - self.miu[index,1]) / self.miu_period[index]
                        self.miu[index, t]= min(calculated_miu, self.limmiu)

            #Define function for EIND --> BIG STOP FROM t = 0 to t =1 something not right
            self.Eind[:,t] = self.sigma_region[:,t] * self.Y_gross[:,t] * (1 - self.miu[:,t])

            #yearly emissions from land change
            self.Etree[:,t] = self.Etree[:,t-1]*(1-self.Emissions_parameter[:,4])

            #yearly combined emissions
            self.E[:,t] = self.Eind[:,t] + self.Etree[:,t]

            #cummulative emissions from land change
            self.cumetree[:,t] = self.cumetree[:,t-1] + self.Etree[:,t] * 10 

            #cummulative emissions from industry
            self.CCA[:,t] = self.CCA[:,t-1] + self.Eind[:,t] * 10
            
            self.CCA[:,t] = np.where(self.CCA[:,t]  < self.fosslim, self.CCA[:,t] ,self.fosslim)

            #total cummulative emissions
            self.CCA_tot = self.CCA[:,t] + self.cumetree[:,t]
                                                         

            """
            ####################### CARBON SUB MODEL #######################
            """

            # Carbon concentration increase in atmosphere [GtC from 1750]

            self.E_worldwilde_per_year = self.E.sum(axis=0)  #1    #2      #3

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
                self.mat[t+1] = 88/100 * self.mat[t] + 4.704/100 * self.mu[t] + self.E_worldwilde_per_year[t]*10

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
            #forcing = constant * Log2( current concentration / concentration of forcing in 1900 at a doubling of CO2 (η)[◦C/2xCO2] ) + external forcing    
            if t < 30:
                self.forc[t] = self.fco22x*(np.log(((self.mat[t]+self.mat[t+1])/2)/(280*2.13)) / np.log(2.0)) + self.forcoth[t]
            if t == 30:
                self.forc[t] = self.fco22x*(np.log((self.mat[t])/(280*2.13)) / np.log(2.0)) + self.forcoth[t]

            """
            ####################### CLIMATE SUB-MODEL ######################
            """
            #heating of oceans and atmospheric according to matrix equations
            
            if t > 1:
                self.temp_atm[t] = (self.temp_atm[t-1]+self.c1
                                    * ((self.forc[t]-((self.fco22x/self.t2xco2)* self.temp_atm[t-1]))
                                       - (self.c3*(self.temp_atm[t-1] - self.temp_ocean[t-1]))))

            #setting up lower and upper bound for temperatures
            if self.temp_atm[t] < self.temp_atm_lo:
                self.temp_atm[t] = self.temp_atm_lo

            if self.temp_atm[t] > self.temp_atm_up:
                self.temp_atm[t] = self.temp_atm_up

            self.temp_ocean[t] = self.temp_ocean[t-1]+self.c4 * (self.temp_atm[t-1]-self.temp_ocean[t-1])

            #setting up lower and upper bound for temperatures
            if self.temp_ocean[t] < self.temp_ocean_lo:
                self.temp_ocean[t] = self.temp_ocean_lo

            if self.temp_ocean[t] > self.temp_ocean_up:
                self.temp_ocean[t] = self.temp_ocean_up

            #thermal expansion
            self.THERMEQUIL[t] = self.temp_atm[t] * self.thermeq

            self.SLRTHERM[t] = self.SLRTHERM[t-1] + self.thermadj * (self.THERMEQUIL[t] - self.SLRTHERM[t-1])

            #glacier ice cap
            self.GSICREMAIN[t] = self.gsictotal - self.GSICCUM[t-1]

            self.GSICMELTRATE[t] = self.gsicmelt * 10 * (self.GSICREMAIN[t] / self.gsictotal)**(self.gsicexp) * self.temp_atm[t]

            self.GSICCUM[t] = self.GSICCUM[t-1] + self.GSICMELTRATE[t]    

            #greenland
            self.GISREMAIN[t] = self.GISREMAIN[t-1] - (self.GISMELTRATE[t-1] / 100)

            if t > 1:
                self.GISMELTRATE[t] = (self.gismeltabove * (self.temp_atm[t] - self.gismineq) + self.gismelt0) * self.GISEXPONENT[t-1]
            else:
                self.GISMELTRATE[1] = 0.60

            self.GISCUM[t] = self.GISCUM[t-1] + self.GISMELTRATE[t] / 100

            if t > 1:
                self.GISEXPONENT[t] = 1 - (self.GISCUM[t] / self.gis0)**self.gisexp
            else:
                self.GISEXPONENT[t] = 1

            #antartica ice cap
            if t <=11:
                if self.temp_atm[t]< 3:
                    self.AISMELTRATE[t] = self.aismeltlow * self.temp_atm[t] * self.aisratio + self.aisintercept
                else:
                    self.AISMELTRATE[t] = self.aisinflection * self.aismeltlow + self.aismeltup * (self.temp_atm[t] - 3.) + self.aisintercept
            else:
                if self.temp_atm[t] < 3:
                    self.AISMELTRATE[t] = self.aismeltlow * self.temp_atm[t] * self.aisratio + self.aismelt0
                else:
                    self.AISMELTRATE[t] = self.aisinflection * self.aismeltlow + self.aismeltup * (self.temp_atm[t] - 3) + self.aismelt0

            self.AISCUM[t] = self.AISCUM[t-1] + self.AISMELTRATE[t] / 100

            self.AISREMAIN[t] = self.AISREMAIN[0] - self.AISCUM[t]

            self.TOTALSLR[t] = self.SLRTHERM[t] + self.GSICCUM[t] + self.GISCUM[t] + self.AISCUM[t]
            
            self.SLRDAMAGES[:,t] =  100 * self.slrmultiplier * (self.TOTALSLR[t-1] * self.slrdamlinear + (self.TOTALSLR[t-1]**2) * self.slrdamquadratic) * (self.Y_gross[:,t-1] / self.Y_gross[:,0])**(1/self.slrelasticity)


            """
            ####################### NET ECONOMY SUB-MODEL ######################
            """

            #original RICE parameters dam_frac
            if self.fdamage == 0:
                print("Nordhaus is used")
                self.dam_frac[:,t] =  (self.damage_parameters[:,0]*self.temp_atm[t] + self.damage_parameters[:,1]*(self.temp_atm[t]**self.damage_parameters[:,2])) * 0.01
                
                #Determine total damages
                self.damages[:,t] = self.Y_gross[:,t]*(self.dam_frac[:,t] + (self.SLRDAMAGES[:,t] / 100))
            
            #Damage function Newbold & Daigneault
            elif self.fdamage == 1:
                print("Newbold is used")
                self.dam_frac_global[t] = (1-(np.exp(-0.0025*self.temp_atm[t]**2.45)))
                
                #translate global damage frac to regional damage frac with factor as used in RICE
                self.dam_frac[:,t] = self.dam_frac_global[t] * self.RICE_regional_damage_factor[:,t]
                
                #calculate damages to economy
                self.damages[:,t] = self.Y_gross[:,t]*self.dam_frac[:,t]

            #Damage function Weitzman
            elif self.fdamage == 2:
                print("weitzman is used")
                
                self.dam_frac_global[t] = (1-1/(1+0.0028388**2+0.0000050703*(self.temp_atm[t]**6.754)))
                
                #translate global damage frac to regional damage frac with factor as used in RICE
                self.dam_frac[:,t] = self.dam_frac_global[t] * self.RICE_regional_damage_factor[:,t]
                
                #calculate damages to economy
                self.damages[:,t] = self.Y_gross[:,t]*self.dam_frac[:,t]

            #determine net output damages with damfrac function chosen in previous step
            self.ynet[:,t] = self.Y_gross[:,t] - self.damages[:,t]

            # Backstop price/cback: cost of backstop   
            
            if year > self.backstopcompetitiveyear:
                self.pbacktime[:,t] = self.pbacktime[:,t-1] * 0.5
            else:
                self.pbacktime[:,t] = 0.10 * self.cback_region + (self.pbacktime[:,t-1]- 0.1 * self.cback_region) * (1-self.decl_back_gr)

            # Adjusted cost for backstop
            self.cost1[:,t] = ((self.pbacktime[:,t]*self.sigma_region[:,t])/self.expcost2)

            #Abettement cost ratio of output
            self.Abetement_cost_RATIO[:,t] = self.cost1[:,t]*(self.miu[:,t]** self.expcost2)
            self.Abetement_cost[:,t] = self.Y_gross[:,t] * self.Abetement_cost_RATIO[:,t]

            #Marginal abetement cost
            self.Mabetement_cost[:,t] = self.pbacktime[:,t] * (self.miu[:,t]**(self.expcost2-1))

            #Resulting carbon price
            self.CPRICE[:,t] = self.pbacktime[:,t] * 1000 * (self.miu[:,t]**(self.expcost2-1))             

            # Gross world product (net of abatement and damages)
            self.Y[:,t] = self.ynet[:,t] - abs(self.Abetement_cost[:,t])

            print("Gross product at t = " + str(t))
            print(self.Y[:,t])

            self.Y[:,t] = np.where(self.Y[:,t] > 0, self.Y[:,t], 0)

            ##############  Investments & Savings  #########################
            if self.model_specification != 'Validation_1':
                # Optimal long-run savings rate used for transversality --> SEE THESIS SHAJEE
                optlrsav = ((self.dk + 0.004) / (self.dk+ 0.004 * self.elasmu + self.irstp) * self.gama)
                
                if self.model_specification == 'Validation_2':
                        if t > 12:
                            self.S[:,t] = optlrsav
                        else: 
                            if t > 1: 
                                    self.S[:,t] = (optlrsav - self.S[:,1]) * t / 12 + self.S[:,1]
                                      
                if self.model_specification == 'EMA':
                        if t > 25:
                            self.S[:,t] = optlrsav
                        else: 
                            if t > 1: 
                                    self.S[:,t] = (self.sr - self.S[:,1]) * t / 12 + self.S[:,1]
                            if t > 12:
                                self.S[:,t] = self.sr

            #investments per region given the savings rate
            self.I[:,t] = self.S[:,t]* self.Y[:,t]
            
            #check lower bound investments
            self.I[:,t] = np.where(self.I[:,t] > 0, self.I[:,t], 0)    
            
            #set up constraints
            c_lo = 2
            CPC_lo = 0.01
                   
            #consumption given the investments
            self.C[:,t] = self.Y[:,t] - self.I[:,t]
            
            #check for lower bound on C
            self.C[:,t] = np.where(self.C[:,t]  > c_lo, self.C[:,t] , c_lo)
        
            #calculate pre damage consumption aggregated per region
            self.pre_damage_total__region_consumption[:,t] = self.C[:,t] + self.damages[:,t]
            
            #damage share elasticity function derived from Denig et al 2015
            self.damage_share = self.RICE_income_shares**elasticity_of_damages
            sum_damage = np.sum(self.damage_share,axis=1)

            for i in range(0,12):
                self.damage_share[i,:] = self.damage_share [i,:]/sum_damage[i]           

            #calculate disaggregated per capita consumption based on income shares BEFORE damages
            self.CPC_pre_damage[year] = ((self.pre_damage_total__region_consumption[:,t] * self.RICE_income_shares.transpose() )  / (self.region_pop[:,t] * (1 / 5))) * 1000

            #calculate disaggregated per capita consumption based on income shares AFTER damages
            self.CPC_post_damage[year] = self.CPC_pre_damage[year]  - (((self.damages[:,t] *  self.damage_share.transpose() ) / (self.region_pop[:,t] * (1 / 5))) * 1000)
            
            #check for lower bound on C
            self.CPC_pre_damage[year] = np.where(self.CPC_pre_damage[year]  > CPC_lo, self.CPC_pre_damage[year], CPC_lo)
            self.CPC_post_damage[year] = np.where(self.CPC_post_damage[year]  > CPC_lo, self.CPC_post_damage[year], CPC_lo)
            
            #calculate damage per quintile equiv
            self.climate_impact_relative_to_capita[year] = ((self.damages[:,t] *  self.damage_share.transpose() * 10**12)/(0.2* self.region_pop[:,t]*10**6)) /(self.CPC_post_damage[year] * 1000)
            
            #average consumption per capita per region
            self.CPC[:,t] = (1000 * self.C[:,t]) / self.region_pop[:,t]

            self.CPC[:,t] = np.where(self.CPC[:,t]  > CPC_lo, self.CPC[:,t] , CPC_lo)
            


            ################################## Utility ##################################
                       
            if self.welfare_function == "utilitarian":
                print("utilitarian SWF is used")
                
                print(self.irstp)
                
                # irstp: Initial rate of social time preference per year
                self.util_sdr[:,t] = 1/((1+self.irstp)**(self.tstep*(t)))
                
                print(self.util_sdr[:,t])
                print(1/((1+self.irstp)**(self.tstep*(t))))

                #instantaneous welfare without ww
                self.inst_util[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC[:,t])**(1 - self.elasmu) + 1) 

                #period utility 
                self.per_util[:,t] = self.inst_util[:,t] * self.region_pop[:,t] * self.util_sdr[:,t]

                #cummulativie period utilty without WW
                self.cum_per_util[:,0] = self.cum_per_util[:,t-1] + self.per_util[:,t] 

                #Instantaneous utility function with welfare weights
                self.inst_util_ww[:,t] = self.inst_util[:,t] * self.Alpha_data[:,t]

                #period utility with welfare weights
                self.per_util_ww[:,t] = self.inst_util_ww[:,t] * self.region_pop[:,t] * self.util_sdr[:,t]

                print("period utility with WW at t = " + str(t))
                print(self.per_util_ww[:,t])

                #cummulative utility with ww 
                self.reg_cum_util[:,t] =  self.reg_cum_util[:,t-1] + self.per_util_ww[:,t]
                
                self.regional_cum_util[t] =  self.reg_cum_util[:,t].sum()

                #scale utility with weights derived from the excel
                if t == 30:
                    self.reg_util = 10  * self.multiplutacive_scaling_weights[:,0] * self.reg_cum_util[:,t] + self.additative_scaling_weights[:,0] - self.additative_scaling_weights[:,2]  

                    print("total scaled cummulative regional utility")
                    print(self.reg_util)

                #calculate worldwide utility 
                self.utility = self.reg_util.sum()
                
                #additional per time step aggregated objectives utilitarian case
                self.global_damages[t] = self.damages[:,t].sum(axis = 0)
                self.global_ouput[t] = self.Y[:,t].sum(axis = 0)
                self.global_per_util_ww[t] = self.per_util_ww[:,t].sum(axis = 0)
                
                """
                ####################### calculate alternative principle objectives for reference scenario ######################
                """
                                
                ####### GINI calculations INTERTEMPORAL #########
                self.average_world_CPC[t] = (self.CPC[:,t].sum() / 12)
                input_gini_inter = self.average_world_CPC
                    
                diffsum = 0
                for i, xi in enumerate(input_gini_inter[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini_inter[i:]))
                
                if t ==30:                
                    self.intertemporal_utility_gini = diffsum / ((len(input_gini_inter)**2)* np.mean(input_gini_inter))
                    
                #intertemporal climate impact GINI
                self.average_regional_impact[t] = (self.damages[:,t].sum() / 12)   
                input_gini = self.average_regional_impact
                    
                diffsum = 0
                for i, xi in enumerate(input_gini[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini[i:]))
                
                if t ==30:   
                    self.intertemporal_impact_gini = diffsum / ((len(input_gini)**2)* np.mean(input_gini))
                
                ####### GINI calculations INTRATEMPORAL #########
                #calculate gini as measure of current inequality in welfare (intragenerational)
                input_gini_intra = self.CPC[:,t]
                
                diffsum = 0
                for i, xi in enumerate(input_gini_intra[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini_intra[i:]))
                        
                self.CPC_intra_gini[t] = diffsum / ((len(input_gini_intra)**2)* np.mean(input_gini_intra))
                
                #calculate gini as measure of current inequality in climate impact (per dollar consumption)
                self.climate_impact_per_dollar_consumption[:,t] = self.damages[:,t] / self.CPC[:,t]
                
                input_gini_intra_impact = self.climate_impact_per_dollar_consumption[:,t]

                diffsum = 0
                for i, xi in enumerate(input_gini_intra_impact[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini_intra_impact[i:]))
                        
                self.climate_impact_per_dollar_gini[t] = diffsum / ((len(input_gini_intra_impact)**2)* np.mean(input_gini_intra_impact))
                  
                #prioritarian objectives
                self.worst_off_income_class[t] = self.CPC_post_damage[year][0].min()
                self.worst_off_climate_impact[t] = self.climate_impact_relative_to_capita[year][0].max()
                
                #sufficitarian objectives
                #growth by the world
                self.average_world_CPC[t] = self.CPC[:,t].sum() / 12
                self.average_growth_CPC[t] = (self.average_world_CPC[t] - self.average_world_CPC[t-1]) / (self.average_world_CPC[t-1])
                
                #sufficitarian treshold adjusted by the growth of the average world economy 
                self.sufficitarian_treshold[t] = self.sufficitarian_treshold[t-1] * (1+self.average_growth_CPC[t])
                                  
                #calculate instantaneous welfare equivalent of minimum capita per head 
                self.inst_util_tres[t] = ((1 / (1 - self.elasmu)) * (self.sufficitarian_treshold[t])**(1 - self.elasmu) + 1) 
                
                #calculate instantaneous welfare equivalent of treshold
                self.inst_util_tres_ww[:,t] = self.inst_util_tres[t] * self.Alpha_data[:,t]
                                
                #calculate utility equivalent for every income quintile and scale with welfare weights for comparison
                self.quintile_inst_util[year] = ((1 / (1 - self.elasmu)) * (self.CPC_post_damage[year])**(1 - self.elasmu) + 1)
                self.quintile_inst_util_ww[year] = self.quintile_inst_util[year] * self.Alpha_data[:,t]       
                
                utility_per_income_share = self.quintile_inst_util_ww[year]
                                                
                for quintile in range(0,5):
                    for region in range(0,12):
                        if utility_per_income_share[quintile,region] < self.inst_util_tres_ww[region,t]:                            
                            self.population_under_treshold[t] = self.population_under_treshold[t] + self.region_pop[region,t] * 1/5
                            self.utility_distance_treshold[region,t] = self.inst_util_tres_ww[region,t] - utility_per_income_share[quintile,region]
                                            
                #minimize max distance to treshold
                self.max_utility_distance_treshold[t] = self.utility_distance_treshold[:,t].max()    
                
           
            if self.welfare_function == "prioritarian":
                print("prioritarian SWF is used")
                
                #specify growth factor for conditional discounting
                self.growth_factor = growth_factor_prio ** 10
                self.prioritarian_discounting = prioritarian_discounting
                    
                # irstp: Initial rate of social time preference per year
                self.util_sdr[:,t] = 1/((1+self.irstp)**(self.tstep*(t)))

                #instantaneous welfare without ww
                self.inst_util[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC[:,t])**(1 - self.elasmu) + 1) 

                #period utility withouw ww
                self.per_util[:,t] = self.inst_util[:,t] * self.region_pop[:,t] * self.util_sdr[:,t]

                #cummulativie period utilty without WW
                self.cum_per_util[:,0] = self.cum_per_util[:,t-1] + self.per_util[:,t] 

                #Instantaneous utility function with welfare weights
                self.inst_util_ww[:,t] = self.inst_util[:,t] * self.Alpha_data[:,t]
                
                #check for discounting prioritarian
                
                #no discounting used
                if self.prioritarian_discounting == 0:
                    self.per_util_ww[:,t] = self.inst_util_ww[:,t] * self.region_pop[:,t]
                    
                #only execute discounting when the lowest income groups experience consumption level growth 
                if self.prioritarian_discounting == 1:
                    #utility worst-off
                    self.inst_util_worst_off[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC_post_damage[year][0])**(1 - self.elasmu) + 1)     
                    
                    self.inst_util_worst_off_condition[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC_post_damage[year-10][0] * self.growth_factor)**(1 - self.elasmu) + 1)     
                    
                    #apply discounting when all regions experience enough growth
 
                    for region in range(0,12):
                        if self.inst_util_worst_off[region,t] >= self.inst_util_worst_off_condition[region,t]:
                            self.per_util_ww[region,t] = self.inst_util_ww[region,t] * self.region_pop[region,t] * self.util_sdr[region,t]
                        
                        #no discounting when lowest income groups do not experience enough growth
                        else:
                            self.per_util_ww[region,t] = self.inst_util_ww[region,t]* self.region_pop[region,t]                        
                
                #objective for the worst-off region in terms of consumption per capita
                self.worst_off_income_class[t] = self.CPC_post_damage[year][0].min()
                
                array_worst_off_income = self.CPC_post_damage[year][0]
                self.worst_off_income_class_index[t] = np.argmin(array_worst_off_income)

                #objective for the worst-off region in terms of climate impact
                self.worst_off_climate_impact[t] = self.climate_impact_relative_to_capita[year][0].max()
                
                array_worst_off_share = self.climate_impact_relative_to_capita[year][0]
                self.worst_off_climate_impact_index[t] = np.argmax(array_worst_off_share)

                #cummulative utility with ww
                self.reg_cum_util[:,t] =  self.reg_cum_util[:,t-1] + self.per_util_ww[:,t]

                #scale utility with weights derived from the excel
                if t == 30:
                    self.reg_util = 10  * self.multiplutacive_scaling_weights[:,0] * self.reg_cum_util[:,t] + self.additative_scaling_weights[:,0] - self.additative_scaling_weights[:,2]  

                    print("total scaled cummulative regional utility")
                    print(self.reg_util)

                #calculate worldwide utility 
                self.utility = self.reg_util.sum()
                
            if self.welfare_function == "sufficitarian":
                print("sufficitarian SWF is used")
                                
                #sufficitarian controls
                self.sufficitarian_discounting = sufficitarian_discounting
                
                #ten year growth factor to be met to discount
                self.temporal_growth_factor = growth_factor_suf **10
                
                #growth by technology frontier
                self.growth_frontier[t] = (np.max(self.CPC[:,t]) - np.max(self.CPC[:,t-1]))/np.max(self.CPC[:,t-1])
                
                #growth by the world
                self.average_world_CPC[t] = self.CPC[:,t].sum() / 12
                self.average_growth_CPC[t] = (self.average_world_CPC[t] - self.average_world_CPC[t-1]) / (self.average_world_CPC[t-1])
                
                #sufficitarian treshold adjusted by the growth of the average world economy 
                self.sufficitarian_treshold[t] = self.sufficitarian_treshold[t-1] * (1+self.average_growth_CPC[t])
                  
                #irstp: Initial rate of social time preference per year
                self.util_sdr[:,t] = 1/((1+self.irstp)**(self.tstep*(t)))

                #instantaneous welfare without ww
                self.inst_util[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC[:,t])**(1 - self.elasmu) + 1) 
                
                #calculate instantaneous welfare equivalent of minimum capita per head 
                self.inst_util_tres[t] = ((1 / (1 - self.elasmu)) * (self.sufficitarian_treshold[t])**(1 - self.elasmu) + 1) 

                #period utility 
                self.per_util[:,t] = self.inst_util[:,t] * self.region_pop[:,t] * self.util_sdr[:,t]

                #cummulativie period utilty without WW
                self.cum_per_util[:,0] = self.cum_per_util[:,t-1] + self.per_util[:,t] 

                #Instantaneous utility function with welfare weights
                self.inst_util_ww[:,t] = self.inst_util[:,t] * self.Alpha_data[:,t]
                
                #calculate instantaneous welfare equivalent of minimum capita per head with PPP
                self.inst_util_tres_ww[:,t] = self.inst_util_tres[t] * self.Alpha_data[:,t]
                
                print("sufficitarian treshold in utility")
                print(self.inst_util_tres_ww[:,t])
                
                #calculate utility equivalent for every income quintile and scale with welfare weights for comparison
                self.quintile_inst_util[year] = ((1 / (1 - self.elasmu)) * (self.CPC_post_damage[year])**(1 - self.elasmu) + 1)
                self.quintile_inst_util_ww[year] = self.quintile_inst_util[year] * self.Alpha_data[:,t]       
                
                utility_per_income_share = self.quintile_inst_util_ww[year]
                
                list_timestep = []
                                
                for quintile in range(0,5):
                    for region in range(0,12):
                        if utility_per_income_share[quintile,region] < self.inst_util_tres_ww[region,t]:                            
                            self.population_under_treshold[t] = self.population_under_treshold[t] + self.region_pop[region,t] * 1/5
                            self.utility_distance_treshold[region,t] = self.inst_util_tres_ww[region,t] - utility_per_income_share[quintile,region]
                            
                            list_timestep.append(regions_list[region])   

                self.regions_under_treshold[t]= list_timestep
                
                #minimize max distance to treshold
                self.max_utility_distance_treshold[t] = self.utility_distance_treshold[:,t].max()            
                            
                
                #sufficitarian discounting
                #only discount when economy situations is as good as timestep before in every region
                if sufficitarian_discounting == 0:
                    for region in range(0,12):
                        if self.CPC[region,t] < self.CPC[region,t-1]:
                            self.per_util_ww[:,t] = self.inst_util_ww[:,t] * self.region_pop[:,t]
                            break
                        else:
                            self.per_util_ww[region,t] = self.inst_util_ww[region,t] * self.region_pop[region,t] * self.util_sdr[region,t]
                        

                #only discount when next generation experiences certain growth in every region
                if sufficitarian_discounting == 1:
                    for region in range(0,12):
                        if self.CPC[region,t] < self.CPC[region,t-1] * self.temporal_growth_factor:
                            self.per_util_ww[:,t] = self.inst_util_ww[:,t] * self.region_pop[:,t]
                            break
                        else:
                            self.per_util_ww[region,t] = self.inst_util_ww[region,t] * self.region_pop[region,t] * self.util_sdr[region,t]
    
                print("period utility with WW at t = " + str(t))
                print(self.per_util_ww[:,t])

                #cummulative utility with ww
                self.reg_cum_util[:,t] =  self.reg_cum_util[:,t-1] + self.per_util_ww[:,t]

                #scale utility with weights derived from the excel
                if t == 30:
                    self.reg_util = 10  * self.multiplutacive_scaling_weights[:,0] * self.reg_cum_util[:,t] + self.additative_scaling_weights[:,0] - self.additative_scaling_weights[:,2]  

                    print("total scaled cummulative regional utility")
                    print(self.reg_util)

                #calculate worldwide utility 
                self.utility = self.reg_util.sum()
                
                self.global_per_util_ww[t] = self.per_util_ww[:,t].sum(axis = 0)
    
            if self.welfare_function == "egalitarian":
                print("egalitarian SWF is used")
                
                #controls for egalitarian principles
                self.egalitarian_discounting = egalitarian_discounting
                                
                #calculate IRSTP
                self.util_sdr[:,t] = 1/((1+self.irstp)**(self.tstep*(t)))
                
                #instantaneous welfare without ww
                self.inst_util[:,t] = ((1 / (1 - self.elasmu)) * (self.CPC[:,t])**(1 - self.elasmu) + 1) 
                
                #period utility without ww
                self.per_util[:,t] = self.inst_util[:,t] * self.region_pop[:,t] * self.util_sdr[:,t]
                
                #cummulativie period utilty without WW
                self.cum_per_util[:,0] = self.cum_per_util[:,t-1] + self.per_util[:,t]
                
                #Instantaneous utility function with welfare weights
                self.inst_util_ww[:,t] = self.inst_util[:,t] * self.Alpha_data[:,t]
                
                #apply no discounting
                if self.egalitarian_discounting == 1:
                    self.per_util_ww[:,t] = self.inst_util_ww[:,t] * self.region_pop[:,t]
                
                else:
                    self.per_util_ww[:,t] = self.inst_util_ww[:,t] * self.region_pop[:,t] * self.util_sdr[:,t]
                
                ####### GINI calculations INTERTEMPORAL #########                
                self.average_world_CPC[t] = (self.CPC[:,t].sum() / 12)

                input_gini_inter = self.average_world_CPC
                    
                diffsum = 0
                for i, xi in enumerate(input_gini_inter[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini_inter[i:]))
                
                if t == 30:                
                    self.intertemporal_utility_gini = diffsum / ((len(input_gini_inter)**2)* np.mean(input_gini_inter))
                    
                #intertemporal climate impact GINI

                self.average_regional_impact[t] = (self.damages[:,t].sum() / 12)
                    
                input_gini = self.average_regional_impact
                    
                diffsum = 0
                for i, xi in enumerate(input_gini[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini[i:]))
                
                if t == 30:   
                    self.intertemporal_impact_gini = diffsum / ((len(input_gini)**2)* np.mean(input_gini))
                
                ####### GINI calculations INTRATEMPORAL #########
                #calculate gini as measure of current inequality in welfare (intragenerational)
                input_gini_intra = self.CPC[:,t]
                
                diffsum = 0
                for i, xi in enumerate(input_gini_intra[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini_intra[i:]))
                        
                self.CPC_intra_gini[t] = diffsum / ((len(input_gini_intra)**2)* np.mean(input_gini_intra))
                
                #calculate gini as measure of current inequality in climate impact (per dollar consumption)  (intragenerational
                self.climate_impact_per_dollar_consumption[:,t] = self.damages[:,t] / self.CPC[:,t]
                
                input_gini_intra_impact = self.climate_impact_per_dollar_consumption[:,t]

                diffsum = 0
                for i, xi in enumerate(input_gini_intra_impact[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini_intra_impact[i:]))
                        
                self.climate_impact_per_dollar_gini[t] = diffsum / ((len(input_gini_intra_impact)**2)* np.mean(input_gini_intra_impact))
                
                print("gini set")
                
                #cummulative utility with ww
                self.reg_cum_util[:,t] =  self.reg_cum_util[:,t-1] + self.per_util_ww[:,t]
                
                #scale utility with weights derived from the excel
                if t == 30:
                    self.reg_util = 10  * self.multiplutacive_scaling_weights[:,0] * self.reg_cum_util[:,t] + self.additative_scaling_weights[:,0] - self.additative_scaling_weights[:,2]  

                    print("total scaled cummulative regional utility")
                    print(self.reg_util)

                #calculate worldwide utility 
                self.utility = self.reg_util.sum()
                
                self.global_per_util_ww[t] = self.per_util_ww[:,t].sum(axis = 0)

            print("####################################################################")
            print("######################      NEXT STEP      #########################")
            print("####################################################################")


        """
        ####################################################################
        ###################### OUTCOME OF INTEREST #########################
        ####################################################################
        """         
                
        if self.welfare_function == "utilitarian":
                      
            self.data = {'Damages 2005': self.global_damages[0],
                         'Utility 2005': self.global_per_util_ww[0],
                         
                         'Damages 2055': self.global_damages[5],
                         'Utility 2055': self.global_per_util_ww[5],
                         
                         'Damages 2105': self.global_damages[10],
                         'Utility 2105': self.global_per_util_ww[10],
                         
                         'Damages 2155': self.global_damages[15],
                         'Utility 2155': self.global_per_util_ww[15],
                         
                         'Damages 2205': self.global_damages[20],
                         'Utility 2205': self.global_per_util_ww[20],
                         
                         'Damages 2305': self.global_damages[30],
                         'Utility 2305': self.global_per_util_ww[30],
                         
                         'Total Aggregated Utility': self.utility
                        }
        
        if self.welfare_function == "prioritarian":
            
            self.data = {'Lowest income per capita 2005': self.worst_off_income_class[0],
                         'Highest climate impact per capita 2005': self.worst_off_climate_impact[0],
                         
                         'Lowest income per capita 2055': self.worst_off_income_class[5],
                         'Highest climate impact per capita 2055': self.worst_off_climate_impact[5],
                         
                         'Lowest income per capita 2105': self.worst_off_income_class[10],
                         'Highest climate impact per capita 2105': self.worst_off_climate_impact[10],
                         
                         'Lowest income per capita 2155': self.worst_off_income_class[15],
                         'Highest climate impact per capita 2155': self.worst_off_climate_impact[15],
                         
                         'Lowest income per capita 2205': self.worst_off_income_class[20],
                         'Highest climate impact per capita 2205': self.worst_off_climate_impact[20],
                         
                         'Lowest income per capita 2305': self.worst_off_income_class[30],
                         'Highest climate impact per capita 2305': self.worst_off_climate_impact[30]
                         }
            
            
        if self.welfare_function == "sufficitarian":

            self.data = {'Distance to treshold 2005': self.max_utility_distance_treshold[0],
                         'Population under treshold 2005':  self.population_under_treshold[0],
                         
                         'Distance to treshold 2055': self.max_utility_distance_treshold[5],
                         'Population under treshold 2055':  self.population_under_treshold[5],
                         
                         'Distance to treshold 2105': self.max_utility_distance_treshold[10],
                         'Population under treshold 2105':  self.population_under_treshold[10],
                         
                         'Distance to treshold 2155': self.max_utility_distance_treshold[15],
                         'Population under treshold 2155':  self.population_under_treshold[15],
                         
                         'Distance to treshold 2205': self.max_utility_distance_treshold[20],
                         'Population under treshold 2205':  self.population_under_treshold[20],
                         
                         'Distance to treshold 2305': self.max_utility_distance_treshold[30],
                         'Population under treshold 2305':  self.population_under_treshold[30],
                         'Total Aggregated Utility': self.utility                         
                        }
            
        if self.welfare_function == "egalitarian":
            
            self.data = {'Intratemporal utility GINI 2005': self.CPC_intra_gini[0],
                         'Intratemporal impact GINI 2005': self.climate_impact_per_dollar_gini[0],
                
                         'Intratemporal utility GINI 2055': self.CPC_intra_gini[5],
                         'Intratemporal impact GINI 2055': self.climate_impact_per_dollar_gini[5],
                
                         'Intratemporal utility GINI 2105': self.CPC_intra_gini[10],
                         'Intratemporal impact GINI 2105': self.climate_impact_per_dollar_gini[10],
                
                         'Intratemporal utility GINI 2155': self.CPC_intra_gini[15],
                         'Intratemporal impact GINI 2155': self.climate_impact_per_dollar_gini[15],
                
                         'Intratemporal utility GINI 2205': self.CPC_intra_gini[20],
                         'Intratemporal impact GINI 2205': self.climate_impact_per_dollar_gini[20],
                
                         'Intratemporal utility GINI 2305': self.CPC_intra_gini[30],
                         'Intratemporal impact GINI 2305': self.climate_impact_per_dollar_gini[30],
                         
                         'Intertemporal utility GINI': self.intertemporal_utility_gini,
                         'Intertemporal impact GINI': self.intertemporal_impact_gini
                        }
                        
        return self.data