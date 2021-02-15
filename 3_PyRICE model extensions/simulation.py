
import numpy as np
import pandas as pd
import math
from scipy.stats import norm, skewnorm, cauchy, lognorm
import logging
import json
import sys
import os


def run(tstep=10, steps=31, model_specification="Validation_1",fdamage = 0,welfare_function="utilitarian"):
    
    pydice_folder = os.path.dirname(os.getcwd())


    sys.path.append(pydice_folder)


    tstep = tstep # (in years)
    steps = steps
    tperiod = []
    startYear = 2005
    model_specification = model_specification
    fdamage = fdamage
    welfare_function = welfare_function


    ########################## SAMPLING OF DAMAGE FUNCTIONS ##########################


    #arrange simulation timeline
    for i in range(0, steps):
        tperiod.append((i*tstep)+startYear)

    #setup of json file to store model results
    with open(pydice_folder + '\\ecs_dist_v5.json') as f:
        d=json.load(f)

    #setting up three distributions for the climate sensitivity; normal lognormal and gauchy

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

    # extend array with the deterministic value of the nordhau
    samples_norm = np.append(samples_norm, 3.2)
    samples_lognorm = np.append(samples_lognorm, 3.2)
    samples_cauchy = np.append(samples_cauchy, 3.2)

    samples_t2xco2 = [samples_norm, samples_lognorm, samples_cauchy]

    #controls for distributive principles
    #prioritarian controls 
    growth_factor_prio = 1.02**10        #how much the worst-off consumption needs to grow each timestep to allow discounting
    prioritarian_discounting = "conditional discounting"   # no discounting or conditional_growth


    #sufficitarian controls
    sufficitarian_discounting = "inheritance discounting"
    growth_factor_suf = 1.02
    ini_suf_treshold = 711.75  #based on the poverty line of 1.95 dollar per day 

    #egalitarian controls
    egalitarian_temporal = ""
    egalitarian_discounting = "temporal egalitarity"


    #uncertainties from Nordhaus(2010) (first draft)
    t2xco2_index = -1
    t2xco2_dist = 0
    fosslim =6000

    #SSP uncertainties
    scenario_pop_gdp = 0    #base RICE2010 scenario
    scenario_sigma = 0  #base RICE2010 scenario
    scenario_cback = 0  #base RICE2010 scenario

    #additonal uncertainty for backstop technology to zero emmissions               
    #cback_to_zero',0,1),


    #decl_back_gr=0.025         

    sr = 0.249          # Savings rate is very different for every region --> how to implement in non optimized RICE?

    periodfullpart=7      #in OPT RICE period full part is 2075
    miu_period=13          #2155 in RICE opt scenario when global emissions are near zero
    limmiu=1   #Upper limit on control rate after 2150, in RICE 1 
    fdamage=0   #0 is original damage function in RICE 1 is fitted SLR BUILT IN SHAJEE DAMAGE FUNCTIONS
    irstp = 0.015   # Initial rate of social time preference (per year) (0.015) (RICE2010 OPT))     

    """
    ######################## INITIALIZE DATA IMPORTS ########################
    """
    RICE_DATA = pd.read_excel("RICE_data.xlsx")
    RICE_PARAMETER = pd.read_excel("RICE_parameter.xlsx")
    RICE_input = pd.read_excel("input_data_RICE.xlsx")
    
    RICE_income_shares = pd.read_excel("RICE_income_shares.xlsx")
    RICE_GDP_SSP = pd.read_excel("Y_Gross_ssp.xlsx")

    RICE_income_shares = RICE_income_shares.iloc[:,1:6].to_numpy()

    #import dataframes for SSP uncertainty analysis
    POP_ssp = pd.read_excel("pop_ssp.xlsx")
    POP_ssp = POP_ssp.iloc[1:,:]        

    regions_list = ["US", "OECD-Europe","Japan","Russia","Non-Russia Eurasia","China","India","Middle East","Africa",
        "Latin America","OHI","Other non-OECD Asia"]

    """
    ############################# LEVERS ###############################
    """

    #setting up model levers
    #if model is EMA the emission control rate miu is sampled 
    #if specification is DICE optimal, the optimal control rate range for every region is taken from Nordhaus 


    ###################### GET CONTROLS FROM RICE OPTIMAL RUN #########################

    #welfare_function = welfare_function

    #get lever series for RICE optimal run
    miu_opt_series = RICE_input.iloc[15:27,1:].to_numpy()
    sr_opt_series = RICE_input.iloc[30:42,1:].to_numpy()


    #Controls with random sampling
    if model_specification == "EMA":

        #create frame for savings rate to be sampled
        S = np.zeros((12, steps))
        miu = np.zeros((12,steps))

        #set starting MIU for all runs
        miu[:,0:2] = miu_opt_series[:,0:2]
        S[:,0:2] = sr_opt_series[:,0:2]

        miu_period = np.full((12, 1), miu_period)
        sr = sr



    #full RICE2010 replicating run
    if model_specification == "Validation_1":

        #set savings rate and control rate as optimized RICE 2010          
        S =  sr_opt_series 

        #set emission control rate for the whole run according to RICE2010 opt.
        miu = miu_opt_series
        irstp = irstp


    #EMA Deterministic
    if model_specification == "Validation_2":

        #create dataframes for control rate and savings rate
        miu = np.zeros((12,steps))
        S = np.zeros((12, steps))

        #set savings rate and control rate as optimized RICE 2010 for the first two timesteps
        miu[:,0:2] = miu_opt_series[:,0:2]
        S[:,0:2] = sr_opt_series[:,0:2]

        #set uncertainties that drive MIU
        limmiu= 1
        irstp = irstp
        miu_period = [12,15,15,10,10,11,13,13,13,14,13,14]

    #define other uncertainties same over all instances
    irstp = irstp
    limmiu = limmiu
    fosslim = fosslim

    """
    ######################## DEEP UNCERTAINTIES ########################
    """

    # Equilibrium temperature impact [dC per doubling CO2]/
    # CLimate sensitivity parameter (3.2 RICE OPT)
    t2xco2 = samples_t2xco2[t2xco2_dist][t2xco2_index]

    # Choice of the damage function (structural deep uncertainty)
    fdamage = fdamage

    """
    ######################## OTHER UNCERTAINTIES ########################
    """
    #define growth factor uncertainties for sampling
    scenario_pop_gdp =scenario_pop_gdp
    scenario_sigma = scenario_sigma
    scenario_cback = scenario_cback

    """
    ####################### Carbon cycle PARAMETERS #######################
    """            

    #RICE2010 INPUTS
    # Initial concentration in atmosphere 2000 [GtC]
    mat0 = 787 
    # Initial concentration in atmosphere 2010 [GtC]
    mat1 = 829
    # Initial concentration in upper strata [GtC]
    mu0 = 1600.0 #1600 in excel
    # Initial concentration in lower strata [GtC]
    ml0 = 10010.0
    # Equilibrium concentration in atmosphere [GtC]
    mateq = 588.0 
    # Equilibrium concentration in upper strata [GtC]
    mueq = 1500.0 
    # Equilibrium concentration in lower strata [GtC]
    mleq = 10000.0

    #parameters are scaled with 100, check with cllimate equations
    b11 = 0.088                                   #88 in excel
    b23 = 0.00500                                 #0.5 in excel
    b12 = 1 -  b11                           
    b21 =  b11 *  mateq /  mueq    
    b22 = 1 -  b21 -  b23               #good in excel       
    b32 =  b23 *  mueq /  mleq     #good in excel
    b33 = 1 -  b32                           #good in excel       

    # 2000 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
    fex0 = -0.06
    # 2100 forcings of non-CO2 GHG [Wm-2]
    fex1 = 0.30
    # Forcings of equilibrium CO2 doubling [Wm-2]
    fco22x = 3.8


    """
    ###################### CLIMATE INITIAL VALUES ######################
    """
    #RICE2010 INPUTS

    # Equilibrium temperature impact [dC per doubling CO2]
    # t2xco2 = t2xco2
    # Initial lower stratum temperature change [dC from 1900]
    tocean0 = 0.0068 
    # Initial atmospheric temperature change [dC from 1900]
    tatm0 = 0.83 


    # 2013 version and earlier:
    # Initial climate equation coefficient for upper level
    # c10 = 0.098
    # Regression slope coefficient (SoA~Equil TSC)
    # c1beta = 0.01243
    # Transient TSC Correction ("Speed of Adjustment Parameter")
    # c1 = c10+c1beta*(t2xco2-2.9)

    #DICE2013R
    # Climate equation coefficient for upper level
    #c1 = 0.098
    # Transfer coefficient upper to lower stratum
    #c3 = 0.088
    # Transfer coefficient for lower level
    #c4 = 0.025
    # Climate model parameter
    #lam =  fco22x /  t2xco2

    #RICE2010
    # Climate equation coefficient for upper level
    c1 = 0.208
    # Transfer coefficient upper to lower stratum
    c3 = 0.310
    # Transfer coefficient for lower level
    c4 = 0.05
    # Climate model parameter
    lam =  fco22x /  t2xco2

    """
    ######################### CARBON PARAMETERS ########################
    """

    mat = np.zeros((steps,))
    mu = np.zeros((steps,))
    ml = np.zeros((steps,))
    forcoth = np.zeros((steps,))
    forc = np.zeros((steps,))

    """
    ######################## CLIMATE PARAMETERS ########################
    """

    # Increase temperature of atmosphere [dC from 1900]
    temp_atm = np.zeros((steps,))
    # Increase temperature of lower oceans [dC from 1900]
    temp_ocean = np.zeros((steps,))


    """
    ######################## DAMAGE PARAMETERS ########################
    """

    #damage parameters excluding SLR from base file 
    damage_parameters =  RICE_input.iloc[47:55,1:13]
    damage_parameters = damage_parameters.transpose().to_numpy()

    #damage parameters INCLUDING SLR FIT Dennig et 
    damage_parameters_slr_fit =  RICE_input.iloc[61:73,1:3]
    damage_parameters_slr_fit = damage_parameters_slr_fit.to_numpy()

    #include SHAJEE stuff here
    """
    ####################### Capital and Economic PARAMETERS #######################
    """
    #population parameteers
    region_pop_gr = RICE_input.iloc[0:12,1:].to_numpy()

    #get population data for 2005
    population2005 = RICE_DATA.iloc[19:31,0].to_numpy()
    region_pop = np.zeros((12,steps))

    #get data for factor productivity growth
    tfpgr_region =  RICE_DATA.iloc[52:64,1:32].to_numpy()

    #get initial values for various parameters
    initails_par = RICE_PARAMETER.iloc[33:40,5:17].to_numpy()
    initials_par = initails_par.transpose()

    #setting up total factor productivity
    tfp_2005 = initials_par[:,5]
    tfp_region = np.zeros((12, steps))

    #setting up capital parameters
    k_2005 = initials_par[:,4]
    k_region = np.zeros((12, steps))
    dk = 0.1
    gama = 0.3

    #setting up Y Gross
    Y_gross = np.zeros((12, steps))
    ynet = np.zeros((12, steps))
    damages = np.zeros((12, steps))
    dam_frac = np.zeros((12, steps))

    #extra dataframe for calculating exponent - waarschijnlijk overbodig
    Sigma_gr_tussenstap = pd.DataFrame(data=np.zeros([12, 1]))
    Sigma_gr_tussenstap["exp"] = ""

    #Dataframes for emissions, economy and utility
    Eind = np.zeros((12, steps))
    E = np.zeros((12, steps))
    Etree = np.zeros((12, steps))
    cumetree = np.zeros((12, steps))
    CCA = np.zeros((12, steps))
    CCA_tot = np.zeros((12, steps))
    Abetement_cost = np.zeros((12, steps))
    Abetement_cost_RATIO = np.zeros((12, steps))
    Mabetement_cost = np.zeros((12, steps))
    CPRICE =np.zeros((12, steps))

    #economy parameters per region
    Y = np.zeros((12, steps))
    I = np.zeros((12, steps))
    C = np.zeros((12, steps))
    CPC = np.zeros((12, steps))

    #output metrics
    util_sdr = np.zeros((12, steps))
    inst_util = np.zeros((12, steps))
    per_util = np.zeros((12, steps))

    cum_util = np.zeros((12, steps))
    reg_cum_util = np.zeros((12, steps))
    reg_util = np.zeros((12, steps))
    util = np.zeros((12, steps))

    per_util_ww =  np.zeros((12, steps))
    cum_per_util = np.zeros((12, steps))
    inst_util_ww = np.zeros((12, steps))

    #alternative SWF output arrays
    sufficitarian_treshold = np.zeros((steps))
    inst_util_tres = np.zeros((steps))
    inst_util_tres_ww = np.zeros((12,steps))

    #Output-to-Emission
    #Change in sigma: the cumulative improvement in energy efficiency)
    sigma_growth_data = RICE_DATA.iloc[70:82,1:6].to_numpy()
    Emissions_parameter = RICE_PARAMETER.iloc[65:70,5:17].to_numpy().transpose()

    #set up dataframe for saving CO2 to output ratio
    Sigma_gr = np.zeros((12, steps))

    #CO2-equivalent-emissions growth to output ratio in 2005
    Sigma_gr[:,0] = sigma_growth_data[:,0]

    #Period at which have full participation
    periodfullpart = periodfullpart 

    # Fraction of emissions under control based on the Paris Agreement
    # US withdrawal would change the value to 0.7086 
    # https://climateanalytics.org/briefings/ratification-tracker/ (0.8875)
    partfract2005 = 1

    #Fraction of emissions under control at full time
    partfractfull = 1.0

    # Decline rate of decarbonization (per period)
    decl_sigma_gr = -0.001

    # Carbon emissions from land 2010 [GtCO2 per year]
    eland0 = 1.6
    # Decline rate of land emissions (per period) CHECKED
    ecl_land = 0.2

    # Elasticity of marginal utility of consumption (1.45) # CHECKED
    elasmu = 1.50

    #Emission data
    emission_factor = RICE_DATA.iloc[87:99,6].to_numpy()
    Eland0 = 1.6 #(RICE2010 OPT)
    #Sigma_gr_tussenstap

    #get alpha data
    Alpha_data = RICE_DATA.iloc[357:369,1:60].to_numpy()
    additative_scaling_weights = RICE_DATA.iloc[167:179,14:17].to_numpy()
    multiplutacive_scaling_weights = RICE_DATA.iloc[232:244,1:2].to_numpy() / 1000

    #Cost of abatement
    abatement_data = RICE_PARAMETER.iloc[56:60,5:17].to_numpy().transpose()

    pbacktime = np.zeros((12, steps))
    cost1 =  np.zeros((12, steps))

    #CO2 to economy ratio
    sigma_region =  np.zeros((12, steps))
    sigma_region[:,0] = Emissions_parameter[:,2] 

    #cback per region
    cback_region = abatement_data[:,0]

    #constations for backstop costs
    ratio_asymptotic = abatement_data[:,2]
    decl_back_gr = abatement_data[:,3]
    expcost2 = 2.8   #RICE 2010 OPT

    #disaggregated consumption tallys
    CPC_post_damage = {}
    CPC_pre_damage = {}
    pre_damage_total__region_consumption = np.zeros((12, steps))

    #dictionaries for quintile outputs
    quintile_inst_util = {}
    quintile_inst_util_ww = {}
    quintile_inst_util_concave = {}
    quintile_per_util_ww = {}

    #prioritarian outputs
    inst_util_worst_off = np.zeros((12,steps))
    inst_util_worst_off_condition = np.zeros((12,steps))
    worst_off_income_class = np.zeros((steps))
    worst_off_income_class_index = np.zeros((steps))
    worst_off_climate_impact = np.zeros((steps))
    worst_off_climate_impact_index = np.zeros((steps))
    climate_impact_per_income_share = {}


    #sufficitarian outputs
    sufficitarian_treshold = np.zeros((12,steps))
    inst_util_tres = np.zeros((12,steps))
    inst_util_tres_ww = np.zeros((12,steps))
    quintile_inst_util = {}
    quintile_inst_util_ww = {}
    population_under_treshold = np.zeros((12,steps))
    utility_distance_treshold = np.zeros((12,steps))
    regions_under_treshold_index = np.zeros((12,steps))
    largest_distance_under_treshold = np.zeros((12,steps))

    #egalitarian outputs
    utility_intra_gini = np.zeros((steps))
    regional_period_utility_sum = np.zeros((steps))
    intertemporal_gini = np.zeros((steps))
    climate_impact_per_dollar_consumption = np.zeros((12,steps))
    climate_impact_per_dollar_gini = np.zeros((steps))

    """
    ####################### LIMITS OF THE MODEL ########################
    """

    # Output low (constraints of the model)
    y_lo = 0.0
    ygross_lo = 0.0
    i_lo = 0.0
    c_lo = 2.0
    cpc_lo = 0
    k_lo = 1.0
    # miu_up[0] = 1.0

    mat_lo = 10.0
    mu_lo = 100.0
    ml_lo = 1000.0
    temp_ocean_up = 20.0
    temp_ocean_lo = -1.0
    temp_atm_lo = 0.0

    #temp_atm_up = 20 or 12 for 2016 version
    temp_atm_up = 40.0      

    """
    ####################### INI CARBON and climate SUB-MODEL #######################
    """

    # Carbon pools
    mat[0] = mat0
    mat[1] = mat1

    if(mat[0] < mat_lo):
        mat[0] = mat_lo

    mu[0] = mu0
    if(mu[0] < mu_lo):
        mu[0] = mu_lo

    ml[0] = ml0
    if(ml[0] < ml_lo):
        ml[0] = ml_lo

    # Radiative forcing
    forcoth[0] = fex0
    forc[0] = fco22x*(np.log(((mat[0]+mat[1])/2)/596.40)/np.log(2.0)) + forcoth[0]

    """
    ################# CLIMATE PARAMETER INTITIALISATION ################
    """
    #checked with RICE2010

    # Atmospheric temperature
    temp_atm[0] = tatm0

    if(temp_atm[0] < temp_atm_lo):
        temp_atm[0] = temp_atm_lo
    if(temp_atm[0] > temp_atm_up):
        temp_atm[0] = temp_atm_up

    # Oceanic temperature
    temp_ocean[0] = 0.007

    if(temp_ocean[0] < temp_ocean_lo):
        temp_ocean[0] = temp_ocean_lo
    if(temp_ocean[0] > temp_ocean_up):
        temp_ocean[0] = temp_ocean_up

    """
    ################# SLR PARAMETER INTITIALISATION ################
    """

    #define inputs
    SLRTHERM = np.zeros((31))
    THERMEQUIL = np.zeros((31))

    GSICREMAIN = np.zeros((31))
    GSICCUM = np.zeros((31))
    GSICMELTRATE = np.zeros((31))
    GISREMAIN = np.zeros((31))
    GISMELTRATE = np.zeros((31))
    GISEXPONENT = np.zeros((31))
    GISCUM = np.zeros((31))
    AISREMAIN = np.zeros((31))
    AISMELTRATE = np.zeros((31))
    AISCUM = np.zeros((31))
    TOTALSLR = np.zeros((31))

    #inputs
    therm0 = 0.092066694
    thermadj = 0.024076141
    thermeq = 0.5

    gsictotal = 0.26
    gsicmelt= 0.0008
    gsicexp = 1
    gsieq = -1

    gis0 = 7.3
    gismelt0 = 0.6
    gismeltabove = 1.118600816
    gismineq = 0
    gisexp = 1

    aismelt0 = 0.21
    aismeltlow = -0.600407185
    aismeltup = 2.225420209
    aisratio = 1.3
    aisinflection = 0
    aisintercept = 0.770332789
    aiswais = 5
    aisother = 51.6

    THERMEQUIL[0] = temp_atm[0] * thermeq
    SLRTHERM[0] = therm0 + thermadj * (THERMEQUIL[0] - therm0)

    GSICREMAIN[0] = gsictotal

    GSICMELTRATE[0] = gsicmelt * 10 * (GSICREMAIN[0] / gsictotal)**(gsicexp) * (temp_atm[0] - gsieq )
    GSICCUM[0] = GSICMELTRATE[0] 
    GISREMAIN[0] = gis0
    GISMELTRATE[0] = gismelt0
    GISCUM[0] = gismelt0 / 100
    GISEXPONENT[0] = 1
    AISREMAIN[0] = aiswais + aisother
    AISMELTRATE[0] = 0.1225
    AISCUM[0] = AISMELTRATE[0] / 100

    TOTALSLR[0] = SLRTHERM[0] + GSICCUM[0] + GISCUM[0] + AISCUM[0]

    slrmultiplier = 2
    slrelasticity = 4

    SLRDAMAGES = np.zeros((12,steps))
    slrdamlinear = np.array([0,0.00452, 0.00053 ,0, 0.00011 , 0.01172 ,0, 0.00138 , 0.00351, 0, 0.00616,0])
    slrdamquadratic = np.array([0.000255,0,0.000053,0.000042,0,0.000001,0.000255,0,0,0.000071,0,0.001239])

    SLRDAMAGES[:,0] = 0

    """
    ################# ECONOMIC PARAMETER INTITIALISATION ################
    """

    #Insert population at 2005 for all regions
    region_pop[:,0] = population2005

    #total factor production at 2005
    tfp_region[:,0] = tfp_2005

    #initial capital in 2005
    k_region[:,0] = k_2005

    # Gama: Capital elasticity in production function
    Y_gross[:,0] = (tfp_region[:,0]*((region_pop[:,0]/1000)**(1-gama)) * (k_region[:,0]**gama))

    #original RICE parameters dam_frac with SLR
    if fdamage == 0:
        dam_frac[:,0] =  (damage_parameters[:,0]*temp_atm[0] 
                        + damage_parameters[:,1]*(temp_atm[0]**damage_parameters[:,2])) * 0.01

    #Damage parameters RICE2010 fitted with extra SLR component
    if fdamage == 1:
        dam_frac[:,0] = 0.01 * (damage_parameters_slr_fit[:,0] * temp_atm[0] + 
                                          (damage_parameters_slr_fit[:,1] *
                                           (temp_atm[0]**damage_parameters[:,2])))

    #Net output damages
    ynet[:,0] = Y_gross[:,0]/(1.0+dam_frac[:,0])

    #Damages in 2005
    damages[:,0] = Y_gross[:,0] - ynet[:,0]

    #Cost of backstop
    pbacktime[:,0] = cback_region

    # Adjusted cost for backstop
    cost1[:,0] = pbacktime[:,0]*sigma_region[:,0]/expcost2

    #decline of backstop competitive year (RICE2010 OPT)
    periodfullpart = 2250

    #Emissions from land change use
    Etree[:,0] = Emissions_parameter[:,3]
    cumetree[:,0] = Emissions_parameter[:,3]

    #industrial emissions 2005
    Eind[:,0] =  sigma_region[:,0] * Y_gross[:,0] * (1 - miu[:,0])

    #initialize initial emissions
    E[:,0] = Eind[:,0] + Etree[:,0]
    CCA[:,0] = Eind[:,0]
    CCA_tot[:,0] = CCA[:,0] + cumetree[:,0]

    #doesnt do much here
    partfract = 1 


    """
    ####################### INIT NET ECONOMY SUB-MODEL ######################
    """                   

    #Cost of climate change to economy
    #Abettement cost ratio of output
    Abetement_cost_RATIO[:,0] = cost1[:,0]*(miu[:,0] ** expcost2)

    #Abettement cost total
    Abetement_cost[:,0] = Y_gross[:,0] * Abetement_cost_RATIO[:,0]

    #Marginal abetement cost
    Mabetement_cost[:,0] = pbacktime[:,0] * miu[:,0]**(expcost2-1)

    #Resulting carbon price
    CPRICE[:,0] = pbacktime[:,0] * 1000 * (miu[:,0]**(expcost2-1))     

    # Gross world product (net of abatement and damages)
    Y[:,0] = ynet[:,0]-Abetement_cost[:,0]           

    ##############  Investments & Savings  #########################

    #investments per region given the savings rate 
    I[:,0] = S[:,0] * Y[:,0]

    #consumption given the investments
    C[:,0] = Y[:,0] - I[:,0]

    #placeholder for different damagefactor per quintile
    quintile_damage_factor = 1

    #calculate pre damage consumption aggregated per region
    pre_damage_total__region_consumption[:,0] = C[:,0] + damages[:,0]

    #calculate damage share with damage factor per quintile
    damage_share = RICE_income_shares.transpose() * quintile_damage_factor

    #calculate disaggregated per capita consumption based on income shares BEFORE damages
    CPC_pre_damage[2005] = ((pre_damage_total__region_consumption[:,0] * RICE_income_shares.transpose() )  / (region_pop[:,0] * (1 / 5))) * 1000

    #calculate disaggregated per capita consumption based on income shares AFTER damages
    CPC_post_damage[2005] = CPC_pre_damage[2005]  - (((damages[:,0] *  damage_share ) / (region_pop[:,0] * (1 / 5))) * 1000)

    #consumption per capita
    CPC[:,0] = (1000 * C[:,0]) / region_pop[:,0]

    ######################################### Utility #########################################

    #Initial rate of social time preference per year
    util_sdr[:,0] = 1

    #Instantaneous utility function equation 
    inst_util[:,0] = ((1 / (1 - elasmu)) * (CPC[:,0])**(1 - elasmu) + 1) * Alpha_data[:,0]           

    #CEMU period utilitity         
    per_util[:,0] = inst_util[:,0] * region_pop[:,0] * util_sdr[:,0]

    #Cummulativie period utilty without WW
    cum_per_util[:,0] = per_util[:,0] 

    #Instantaneous utility function with welfare weights
    inst_util_ww[:,0] = inst_util[:,0] * Alpha_data[:,0]

    #Period utility with welfare weights
    per_util_ww[:,0] = inst_util_ww[:,0] * region_pop[:,0] * util_sdr[:,0]

    #cummulative utility with ww
    reg_cum_util[:,0] =  per_util[:,0] 

    #scale utility with weights derived from the excel
    reg_util[:,0] = 10  * multiplutacive_scaling_weights[:,0] * reg_cum_util[:,0] + additative_scaling_weights[:,0] - additative_scaling_weights[:,2]  

    #calculate worldwide utility 
    utility = reg_util.sum()            

    """
    ########################################## RICE MODEL ###################################################    
    """    


    #Follows equations of notes #TOTAL OF 30 STEPS UNTIL 2305
    for t in range(1,31): 

        """
        ####################### GROSS ECONOMY SUB-MODEL ######################
        """

        #use ssp population projections if not base with right SSP scenario (SSP1, SSP2 etc.)
        if scenario_pop_gdp !=0:

            #load population and gdp projections from SSP scenarios on first timestep
            if t == 1:
                for region in range(0,12):
                    region_pop[region,:] = POP_ssp.iloc[:,scenario_pop_gdp + (region * 5)]

                    Y_gross[region,:] = RICE_GDP_SSP.iloc[:,scenario_pop_gdp + (region * 5)] / 1000

            Y_gross[:,t] = np.where(Y_gross[:,t]  > 0, Y_gross[:,t], 0)

            k_region[:,t] = k_region[:,t-1]*((1-dk)**tstep) + tstep*I[:,t-1]

            #calculate tfp based on gdp projections by SSP's
            tfp_region[:,t] = Y_gross[:,t] / ((k_region[:,t]**gama)*(region_pop[:,t]/1000)**(1-gama))

        #base tfp projections RICE2010
        else:
            #calculate population at time t
            region_pop[:,t] = region_pop[:,t-1] *  2.71828 **(region_pop_gr[:,t]*10)

            #TOTAL FACTOR PRODUCTIVITY level according to RICE base
            tfp_region[:,t] = tfp_region[:,t-1] * 2.71828 **(tfpgr_region[:,t]*10)

            #determine capital stock at time t
            k_region[:,t] = k_region[:,t-1]*((1-dk)**tstep) + tstep*I[:,t-1]

            #lower bound capital
            k_region[:,t] = np.where(k_region[:,t]  > 1, k_region[:,t] ,1)

            #determine Ygross at time t
            Y_gross[:,t] = tfp_region[:,t] * ((region_pop[:,t]/1000)**(1-gama))*(k_region[:,t]**gama)   

            #lower bound Y_Gross
            Y_gross[:,t] = np.where(Y_gross[:,t]  > 0, Y_gross[:,t], 0)

        #capital and ygross show minor deviations after t =1 because of influence Y net
        #damage function is slidely different because of different damage functions
        #this influences the gross economy cycle as well as emissions, damages and welfare

        #calculate the sigma growth and the emission rate development          
        if t == 1:
            Sigma_gr[:,t] = (sigma_growth_data[:,4] + (sigma_growth_data[:,2] - sigma_growth_data[:,4] )) 

            sigma_region[:,t] = sigma_region[:,t-1] *  (2.71828 ** (Sigma_gr[:,t]*10)) * emission_factor

        if t > 1 :
            Sigma_gr[:,t] = (sigma_growth_data[:,4] + (Sigma_gr[:,t-1] - sigma_growth_data[:,4]  ) * (1-sigma_growth_data[:,3] )) 

            sigma_region[:,t] = sigma_region[:,t-1] *  (2.71828 ** ( Sigma_gr[:,t]*10)) 


        #print("CO2 economy ratio = " + str(t))
        #print(sigma_region.iloc[:,t])

        if model_specification == "EMA":
            # control rate is maximum after target period, otherwise linearly increase towards that point from t[0]
            # Control rate limit
            if t > 1:
                    for index in range(0,12):            
                        calculated_miu = miu[index,t-1] + (limmiu - miu[index,1]) / miu_period[index]
                        miu[index, t]= min(calculated_miu, 1.00)

        if model_specification == "Validation_2": 
            if t > 1:
                for index in range(0,12):            
                    calculated_miu = miu[index,t-1] + (limmiu - miu[index,1]) / miu_period[index]
                    miu[index, t]= min(calculated_miu, 1.00)


        #controlrate is werird output does not match --> this will cause CO2 emissions also not to match
        #print("Control rate = " + str(t))
        #print(miu.iloc[:,t])

        #Define function for EIND --> BIG STOP FROM t = 0 to t =1 something not right
        Eind[:,t] = sigma_region[:,t] * Y_gross[:,t] * (1 - miu[:,t])

        #yearly emissions from land change
        Etree[:,t] = Etree[:,t-1]*(1-Emissions_parameter[:,4])

        #print("emissions from change in land use: t = " + str(t))
        #print(Etree.iloc[:,t])

        #yearly combined emissions
        E[:,t] = Eind[:,t] + Etree[:,t]

        #cummulative emissions from land change
        cumetree[:,t] = cumetree[:,t-1] + Etree[:,t] * 10 

        #cummulative emissions from industry
        CCA[:,t] = CCA[:,t-1] + Eind[:,t] * 10

        CCA[:,t] = np.where(CCA[:,t]  < fosslim, CCA[:,t] ,fosslim)

        #total cummulative emissions
        CCA_tot = CCA[:,t] + cumetree[:,t]


        """
        ####################### CARBON SUB MODEL #######################
        """

        # Carbon concentration increase in atmosphere [GtC from 1750]

        E_worldwilde_per_year = E.sum(axis=0)  #1    #2      #3

        #parameters are scaled with 100, check with cllimate equations
        #b11 = 0.012                                 #88 in excel
        #b23 = 0.00500                                 #0.5 in excel
        #b12 = 1 -  b11                           
        #b21 =  b11 *  mateq /  mueq    
        #b22 = 1 -  b21 -  b23               #good in excel       
        #b32 =  b23 *  mueq /  mleq     #good in excel
        #b33 = 1 -  b32                           #good in excel       

        #calculate concentration in bioshpere and upper oceans
        mu[t] = 12/100 * mat[t-1] + 94.796/100*mu[t-1] + 0.075/100 *ml[t-1]

        #set lower constraint for shallow ocean concentration
        if(mu[t] < mu_lo):
            mu[t] = mu_lo

        # Carbon concentration increase in lower oceans [GtC from 1750]
        ml[t] = 99.925/100 *ml[t-1]+0.5/100 * mu[t-1]

        #set lower constraint for shallow ocean concentration
        if(ml[t] < ml_lo):
            ml[t] = ml_lo

        #calculate concentration in atmosphere for t + 1 (because of averaging in forcing formula
        if t < 30:
            mat[t+1] = 88/100 * mat[t] + 4.704/100 * mu[t] + E_worldwilde_per_year[t]*10

        #set lower constraint for atmospheric concentration
        if(mat[t] < mat_lo):
            mat[t] = mat_lo

        # Radiative forcing

        #Exogenous forcings from other GHG
        #rises linearly from 2010 to 2100 from -0.060 to 0.3 then becomes stable in RICE -  UPDATE FOR DICE2016R

        exo_forcing_2000 = -0.060
        exo_forcing_2100 = 0.3000

        if (t < 11):
            forcoth[t] = fex0+0.1*(exo_forcing_2100 - exo_forcing_2000 )*(t)
        else:
            forcoth[t] = exo_forcing_2100


        # Increase in radiative forcing [Wm-2 from 1900]
        #forcing = constant * Log2( current concentration / concentration of forcing in 1900 at a doubling of CO2 (η)[◦C/2xCO2] ) + external forcing    
        if t < 30:
            forc[t] = fco22x*(np.log(((mat[t]+mat[t+1])/2)/(280*2.13)) / np.log(2.0)) + forcoth[t]
        if t == 30:
            forc[t] = fco22x*(np.log((mat[t])/(280*2.13)) / np.log(2.0)) + forcoth[t]


        """
        ####################### CLIMATE SUB-MODEL ######################
        """
        #heating of oceans and atmospheric according to matrix equations
        if t == 1:
            temp_atm[t] = 0.980
        if t > 1:
            temp_atm[t] = (temp_atm[t-1]+c1
                                * ((forc[t]-((fco22x/t2xco2)* temp_atm[t-1]))
                                   - (c3*(temp_atm[t-1] - temp_ocean[t-1]))))

        #setting up lower and upper bound for temperatures
        if (temp_atm[t] < temp_atm_lo):
            temp_atm[t] = temp_atm_lo

        if (temp_atm[t] > temp_atm_up):
            temp_atm[t] = temp_atm_up

        temp_ocean[t] = (temp_ocean[t-1]+c4 * (temp_atm[t-1]-temp_ocean[t-1]))

        #setting up lower and upper bound for temperatures
        if (temp_ocean[t] < temp_ocean_lo):
            temp_ocean[t] = temp_ocean_lo

        if (temp_ocean[t] > temp_ocean_up):
            temp_ocean[t] = temp_ocean_up

        #thermal expansion
        THERMEQUIL[t] = temp_atm[t] * thermeq

        SLRTHERM[t] = SLRTHERM[t-1] + thermadj * (THERMEQUIL[t] - SLRTHERM[t-1])

        #glacier ice cap
        GSICREMAIN[t] = gsictotal - GSICCUM[t-1]

        GSICMELTRATE[t] = gsicmelt * 10 * (GSICREMAIN[t] / gsictotal)**(gsicexp) * temp_atm[t]

        GSICCUM[t] = GSICCUM[t-1] + GSICMELTRATE[t]    

        #greenland
        GISREMAIN[t] = GISREMAIN[t-1] - (GISMELTRATE[t-1] / 100)

        if t > 1:
            GISMELTRATE[t] = (gismeltabove * (temp_atm[t] - gismineq) + gismelt0) * GISEXPONENT[t-1]
        else:
            GISMELTRATE[1] = 0.60

        GISCUM[t] = GISCUM[t-1] + GISMELTRATE[t] / 100

        if t > 1:
            GISEXPONENT[t] = 1 - (GISCUM[t] / gis0)**gisexp
        else:
            GISEXPONENT[t] = 1

        #antartica ice cap
        if t <=11:
            if temp_atm[t]< 3:
                AISMELTRATE[t] = aismeltlow * temp_atm[t] * aisratio + aisintercept
            else:
                AISMELTRATE[t] = aisinflection * aismeltlow + aismeltup * (temp_atm[t] - 3.) + aisintercept
        else:
            if temp_atm[t] < 3:
                AISMELTRATE[t] = aismeltlow * temp_atm[t] * aisratio + aismelt0
            else:
                AISMELTRATE[t] = aisinflection * aismeltlow + aismeltup * (temp_atm[t] - 3) + aismelt0

        AISCUM[t] = AISCUM[t-1] + AISMELTRATE[t] / 100

        AISREMAIN[t] = AISREMAIN[0] - AISCUM[t]

        TOTALSLR[t] = SLRTHERM[t] + GSICCUM[t] + GISCUM[t] + AISCUM[t]

        SLRDAMAGES[:,t] =  100 * slrmultiplier * (TOTALSLR[t-1] * slrdamlinear + (TOTALSLR[t-1]**2) * slrdamquadratic) * (Y_gross[:,t-1] / Y_gross[:,0])**(1/slrelasticity)


        """
        ####################### NET ECONOMY SUB-MODEL ######################
        """

        #original RICE parameters dam_frac
        if fdamage == 0:
            dam_frac[:,t] =  (damage_parameters[:,0]*temp_atm[t] + damage_parameters[:,1]*(temp_atm[t]**damage_parameters[:,2])) * 0.01

            #Determine total damages
            damages[:,t] = Y_gross[:,t]*(dam_frac[:,t] + (SLRDAMAGES[:,t] / 100))

        #Damage parameters RICE2010 fitted with extra SLR component
        if fdamage == 1:
            dam_frac[:,t] = (damage_parameters_slr_fit[:,0]*temp_atm[t] 
                                       + damage_parameters_slr_fit[:,1]*
                                       (temp_atm[t]**damage_parameters[:,2])) * 0.01

            #determine total damages
            damages[:,t] = Y_gross[:,t]*dam_frac[:,t]

        #determine net output damages with damfrac function chosen in previous step
        ynet[:,t] = Y_gross[:,t] - damages[:,t]

        #print("Y net at time t = " + str(t))
        #print(ynet.iloc[:,t])

        # Backstop price/cback: cost of backstop                
        pbacktime[:,t] = 0.10 * cback_region + (pbacktime[:,t-1]- 0.1 * cback_region) * (1-decl_back_gr)

        #print(pbacktime.iloc[:,t])

        # Adjusted cost for backstop
        cost1[:,t] = ((pbacktime[:,t]*sigma_region[:,t])/expcost2)

        #print("adjusted cost of backstop at t =  " + str(t))
        #print(cost1.iloc[:,t])

        #Abettement cost ratio of output
        Abetement_cost_RATIO[:,t] = cost1[:,t]*(miu[:,t]** expcost2)

        Abetement_cost[:,t] = Y_gross[:,t] * Abetement_cost_RATIO[:,t]

        #print("abatement  cost in trillion $ at time t = " + str(t))
        #print(Abetement_cost.iloc[:,t])

        #Marginal abetement cost
        Mabetement_cost[:,t] = pbacktime[:,t] * (miu[:,t]**(expcost2-1))

        #Resulting carbon price
        #goes wrong here miu not right --> different from excel ?
        CPRICE[:,t] = pbacktime[:,t] * 1000 * (miu[:,t]**(expcost2-1))             

        #print("carbon price  at t =  " + str(t))
        #print(CPRICE.iloc[:,t])

        # Gross world product (net of abatement and damages)
        Y[:,t] = ynet[:,t] - abs(Abetement_cost[:,t])

        Y[:,t] = np.where(Y[:,t] > 0, Y[:,t], 0)

        ##############  Investments & Savings  #########################
        if model_specification != 'Validation_1':
            # Optimal long-run savings rate used for transversality --> SEE THESIS SHAJEE
            optlrsav = ((dk + 0.004) / (dk+ 0.004 * elasmu + irstp) * gama)

            if model_specification == 'Validation_2':
                    if t > 12:
                        S[:,t] = optlrsav
                    else: 
                        if t > 1: 
                                S[:,t] = (optlrsav - S[:,1]) * t / 12 + S[:,1]

            if model_specification == 'EMA':
                    if t > 25:
                        S[:,t] = optlrsav
                    else: 
                        if t > 1: 
                                S[:,t] = (sr - S[:,1]) * t / 12 + S[:,1]
                        if t > 12:
                            S[:,t] = sr

        #investments per region given the savings rate -

        I[:,t] = S[:,t]* Y[:,t]

        #check lower bound investments
        I[:,t] = np.where(I[:,t] > 0, I[:,t], 0)

        #set up constraints
        c_lo = 2
        CPC_lo = 0.01

        #consumption given the investments
        C[:,t] = Y[:,t] - I[:,t]

        #check for lower bound on C
        C[:,t] = np.where(C[:,t]  > c_lo, C[:,t] , c_lo)

        #keep track of year for storing in dict
        year = 2005 + 10 * t

        #calculate pre damage consumption aggregated per region
        pre_damage_total__region_consumption[:,t] = C[:,t] + damages[:,t]

        #damage spread equally across every person
        #damage_share = (model.RICE_income_shares**0 ) * 0.2

        #damage share according to Denig et al 2015
        damage_share = RICE_income_shares**-1
        sum_damage = np.sum(damage_share,axis=1)

        for i in range(0,12):
            damage_share[i,:] = damage_share [i,:]/sum_damage[i]           

        #calculate disaggregated per capita consumption based on income shares BEFORE damages
        CPC_pre_damage[year] = ((pre_damage_total__region_consumption[:,t] * RICE_income_shares.transpose() )  / (region_pop[:,t] * (1 / 5))) * 1000

        #calculate disaggregated per capita consumption based on income shares AFTER damages
        CPC_post_damage[year] = CPC_pre_damage[year]  - (((damages[:,t] *  damage_share.transpose() ) / (region_pop[:,t] * (1 / 5))) * 1000)

        #calculate damage per quintile equiv
        climate_impact_per_income_share[year] = damages[:,t] *  damage_share.transpose()

        #average consumption per capita per region
        CPC[:,t] = (1000 * C[:,t]) / region_pop[:,t]

        CPC[:,t] = np.where(CPC[:,t]  > CPC_lo, CPC[:,t] , CPC_lo)

        ################################## Utility ##################################

        #set up df to check swfs

        if welfare_function == "utilitarian":
            print("utilitarian SWF is used")

            # irstp: Initial rate of social time preference per year
            util_sdr[:,t] = 1/((1+irstp)**(tstep*(t)))

            #instantaneous welfare without ww
            inst_util[:,t] = ((1 / (1 - elasmu)) * (CPC[:,t])**(1 - elasmu) + 1) 

            #period utility 
            per_util[:,t] = inst_util[:,t] * region_pop[:,t] * util_sdr[:,t]

            #cummulativie period utilty without WW
            cum_per_util[:,0] = cum_per_util[:,t-1] + per_util[:,t] 

            #Instantaneous utility function with welfare weights
            inst_util_ww[:,t] = inst_util[:,t] * Alpha_data[:,t]

            #period utility with welfare weights
            per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t] * util_sdr[:,t]
            #cummulative utility with ww
            reg_cum_util[:,t] =  reg_cum_util[:,t-1] + per_util_ww[:,t]

            #scale utility with weights derived from the excel
            if t == 30:
                reg_util[:,t] = 10  * multiplutacive_scaling_weights[:,0] * reg_cum_util[:,t] + additative_scaling_weights[:,0] - additative_scaling_weights[:,2]  

                print("total scaled cummulative regional utility")
                print(reg_util[:,t])

            #calculate worldwide utility 
            utility = reg_util.sum()



        if welfare_function == "prioritarian":
            print("prioritarian SWF is used")

            #specify growth factor for conditional discounting
            growth_factor = growth_factor_prio
            prioritarian_discounting = prioritarian_discounting


            # irstp: Initial rate of social time preference per year
            util_sdr[:,t] = 1/((1+irstp)**(tstep*(t)))

            #instantaneous welfare without ww
            inst_util[:,t] = ((1 / (1 - elasmu)) * (CPC[:,t])**(1 - elasmu) + 1) 

            #period utility withouw ww
            per_util[:,t] = inst_util[:,t] * region_pop[:,t] * util_sdr[:,t]

            #cummulativie period utilty without WW
            cum_per_util[:,0] = cum_per_util[:,t-1] + per_util[:,t] 

            #Instantaneous utility function with welfare weights
            inst_util_ww[:,t] = inst_util[:,t] * Alpha_data[:,t]

            #check for discounting prioritarian

            #no discounting used
            if prioritarian_discounting == "no discounting":
                per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t]

            #only execute discounting when the lowest income groups experience consumption level growth 
            if prioritarian_discounting == "conditional discounting":
                #utility worst-off
                inst_util_worst_off[:,t] = ((1 / (1 - elasmu)) * (CPC_post_damage[year][0])**(1 - elasmu) + 1)     

                inst_util_worst_off_condition[:,t] = ((1 / (1 - elasmu)) * (CPC_post_damage[year-10][0] * growth_factor)**(1 - elasmu) + 1)     

                #apply discounting when all regions experience enough growth

                for region in range(0,12):
                    if inst_util_worst_off[region,t] >= inst_util_worst_off_condition[region,t]:
                        per_util_ww[region,t] = inst_util_ww[region,t] * region_pop[region,t] * util_sdr[region,t]

                    #no discounting when lowest income groups do not experience enough growth
                    else:
                        per_util_ww[region,t] = inst_util_ww[region,t]* region_pop[region,t]                        

            #objective for the worst-off region in terms of consumption per capita
            worst_off_income_class[t] = CPC_post_damage[year][0].min()

            array_worst_off_income = CPC_post_damage[year][0]
            worst_off_income_class_index[t] = np.argmin(array_worst_off_income)

            #objective for the worst-off region in terms of climate impact
            worst_off_climate_impact[t] = climate_impact_per_income_share[year][0].min()

            array_worst_off_share = climate_impact_per_income_share[year][0]
            worst_off_climate_impact_index[t] = np.argmin(array_worst_off_share)

            #cummulative utility with ww
            reg_cum_util[:,t] =  reg_cum_util[:,t-1] + per_util_ww[:,t]

            #scale utility with weights derived from the excel
            if t == 30:
                reg_util[:,t] = 10  * multiplutacive_scaling_weights[:,0] * reg_cum_util[:,t] + additative_scaling_weights[:,0] - additative_scaling_weights[:,2]  

                print("total scaled cummulative regional utility")
                print(reg_util[:,t])

            #calculate worldwide utility 
            utility = reg_util.sum()



        if welfare_function == "sufficitarian":
            print("sufficitarian SWF is used")

            #sufficitarian controls
            sufficitarian_discounting = sufficitarian_discounting
            growth_factor = growth_factor_suf
            ini_suf_treshold = ini_suf_treshold,  #specified in consumption per capita thousand/year 

            #growth by technology frontier
            growth_frontier = (np.max(CPC[:,t]) - np.max(CPC[:,t-1]))/np.max(CPC[:,t-1])

            sufficitarian_treshold[t] = ini_suf_treshold * growth_frontier

            #irstp: Initial rate of social time preference per year
            util_sdr[:,t] = 1/((1+irstp)**(tstep*(t)))

            #instantaneous welfare without ww
            inst_util[:,t] = ((1 / (1 - elasmu)) * (CPC[:,t])**(1 - elasmu) + 1) 

            #calculate instantaneous welfare equivalent of minimum capita per head 
            inst_util_tres[t] = ((1 / (1 - elasmu)) * (sufficitarian_treshold[t])**(1 - elasmu) + 1) 

            #period utility 
            per_util[:,t] = inst_util[:,t] * region_pop[:,t] * util_sdr[:,t]

            #cummulativie period utilty without WW
            cum_per_util[:,0] = cum_per_util[:,t-1] + per_util[:,t] 

            #Instantaneous utility function with welfare weights
            inst_util_ww[:,t] = inst_util[:,t] * Alpha_data[:,t]

            #calculate instantaneous welfare equivalent of minimum capita per head with PPP
            inst_util_tres_ww[:,t] = inst_util_tres[t] * Alpha_data[:,t]

            print("sufficitarian treshold in utility")
            print(inst_util_tres_ww[:,t])

            #calculate utility equivalent for every income quintile and scale with welfare weights for comparison
            quintile_inst_util[year] = ((1 / (1 - elasmu)) * (CPC_post_damage[year])**(1 - elasmu) + 1)
            quintile_inst_util_ww[year] = quintile_inst_util[year] * Alpha_data[:,t]       

            utility_per_income_share = quintile_inst_util_ww[year]

            index = 0

            for quintile in range(0,5):
                for region in range(0,12):
                    if utility_per_income_share[quintile,region] < inst_util_tres_ww[region,t]:                            
                        population_under_treshold[t] = population_under_treshold[t] + region_pop[region,t] * 1/5
                        utility_distance_treshold[region,t] = inst_util_tres_ww[region,t] - utility_per_income_share[quintile,region]

                        regions_under_treshold_index[index,t] = region

                        index = index + 1

            #objective: minimize distance under treshold           
            largest_distance_under_treshold[t] = np.max(utility_distance_treshold[:,t])         

            #sufficitarian discounting

            #only discount when economy situations is as good as timestep before in every region
            if sufficitarian_discounting == "inheritance discounting":
                for region in range(0,12):
                    if inst_util_ww[region,t] < inst_util_ww[region,t-1]:
                        per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t]
                        break
                    else:
                        per_util_ww[region,t] = inst_util_ww[region,t] * region_pop[region,t] * util_sdr[region,t]


            #only discount when next generation experiences certain growth in every region
            if sufficitarian_discounting == "sustainable growth discounting":
                for region in range(0,12):
                    if inst_util_ww[region,t] < inst_util_ww[region,t-1] * growth_factor:
                        per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t]
                        break
                    else:
                        per_util_ww[region,t] = inst_util_ww[region,t] * region_pop[region,t] * util_sdr[region,t]

            #cummulative utility with ww
            reg_cum_util[:,t] =  reg_cum_util[:,t-1] + per_util_ww[:,t]

            #scale utility with weights derived from the excel
            if t == 30:
                reg_util[:,t] = 10  * multiplutacive_scaling_weights[:,0] * reg_cum_util[:,t] + additative_scaling_weights[:,0] - additative_scaling_weights[:,2]  

                print("total scaled cummulative regional utility")
                print(reg_util[:,t])

            #calculate worldwide utility 
            utility = reg_util.sum()


        if welfare_function == "egalitarian":
            print("egalitarian SWF is used")

            #controls for egalitarian principles
            egalitarian_discounting = egalitarian_discounting
            egalitarian_temporal = egalitarian_temporal

            #calculate IRSTP
            util_sdr[:,t] = 1/((1+irstp)**(tstep*(t)))

            #instantaneous welfare without ww
            inst_util[:,t] = ((1 / (1 - elasmu)) * (CPC[:,t])**(1 - elasmu) + 1) 

            #period utility without ww
            per_util[:,t] = inst_util[:,t] * region_pop[:,t] * util_sdr[:,t]

            #cummulativie period utilty without WW
            cum_per_util[:,0] = cum_per_util[:,t-1] + per_util[:,t]

            #Instantaneous utility function with welfare weights
            inst_util_ww[:,t] = inst_util[:,t] * Alpha_data[:,t]

            #apply no discounting
            if egalitarian_discounting == "no discounting":
                per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t]

            else:
                per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t] * util_sdr[:,t]

            #only execute discounting when the lowest income groups experience consumption level growth 
            if egalitarian_temporal == "temporal egalitarity":
                per_util_ww[:,t] = inst_util_ww[:,t] * region_pop[:,t]

                regional_period_utility_sum[t] = per_util_ww[:,t].sum()

                input_gini = regional_period_utility_sum

                diffsum = 0
                for i, xi in enumerate(input_gini[:-1], 1):
                    diffsum += np.sum(np.abs(xi - input_gini[i:]))

                    intertemporal_gini[t] = diffsum / ((len(input_gini)**2)* np.mean(input_gini))

            #calculate gini as measure of current inequality in welfare
            input_gini = inst_util_ww[:,t]

            diffsum = 0
            for i, xi in enumerate(input_gini[:-1], 1):
                diffsum += np.sum(np.abs(xi - input_gini[i:]))

                utility_intra_gini[t] = diffsum / ((len(input_gini)**2)* np.mean(input_gini))


            #calculate gini as measure of current inequality in climate impact (per dollar consumption)  
            climate_impact_per_dollar_consumption[:,t] = damages[:,t] / CPC[:,t]

            input_gini = climate_impact_per_dollar_consumption[:,t]

            diffsum = 0
            for i, xi in enumerate(input_gini[:-1], 1):
                diffsum += np.sum(np.abs(xi - input_gini[i:]))

                climate_impact_per_dollar_gini[t] = diffsum / ((len(input_gini)**2)* np.mean(input_gini))


            #cummulative utility with ww
            reg_cum_util[:,t] =  reg_cum_util[:,t-1] + per_util_ww[:,t]

            #scale utility with weights derived from the excel
            if t == 30:
                reg_util[:,t] = 10  * multiplutacive_scaling_weights[:,0] * reg_cum_util[:,t] + additative_scaling_weights[:,0] - additative_scaling_weights[:,2]  

                print("total scaled cummulative regional utility")
                print(reg_util[:,t])

            #calculate worldwide utility 
            utility = reg_util.sum()


        print("####################################################################")
        print("######################    NEXT STEP        #########################")
        print("####################################################################")



    """
    ####################################################################
    ###################### OUTCOME OF INTEREST #########################
    ####################################################################
    """   

    data = {'Atmospheric Temperature 2005': temp_atm[0],
                 'Damages 2005': damages[:,0],
                 'Industrial Emission 2005': Eind[:,0],
                 'Utility 2005': per_util_ww[:,0],
                 'Total Output 2005': Y[:,0],

                 'Atmospheric Temperature 2055': temp_atm[5],
                 'Damages 2055': damages[:,5],
                 'Industrial Emission 2055': Eind[:,5],
                 'Utility 2055': per_util_ww[:,5],
                 'Total Output 2055': Y[:,5],

                 'Atmospheric Temperature 2105': temp_atm[10],
                 'Damages 2105': damages[:,10],
                 'Industrial Emission 2105': Eind[:,10],
                 'Utility 2105': per_util_ww[:,10],
                 'Total Output 2105': Y[:,10],

                 'Atmospheric Temperature 2155': temp_atm[15],
                 'Damages 2155': damages[:,15],
                 'Industrial Emission 2155': Eind[:,15],
                 'Utility 2155': per_util_ww[:,15],
                 'Total Output 2155': Y[:,15],

                 'Atmospheric Temperature 2205': temp_atm[20],
                 'Damages 2205': damages[:,20],
                 'Industrial Emission 2205': Eind[:,20], 
                 'Utility 2205': per_util_ww[:,20],
                 'Total Output 2205': Y[:,20],

                 'Atmospheric Temperature 2305': temp_atm[30],
                 'Damages 2305': damages[:,30],
                 'Industrial Emission 2305': Eind[:,30],
                 'Utility 2305': per_util_ww[:,30],
                 'Total Output 2305': Y[:,30]}
    return data
