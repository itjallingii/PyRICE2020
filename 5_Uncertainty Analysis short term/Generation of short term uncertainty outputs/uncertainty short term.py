import pandas as pd
import time
import os
import numpy as np
import sys
import itertools

from ema_workbench import save_results, load_results

pydice_folder = os.path.dirname(os.getcwd()) + '//server//model_server'
save_folder = "//root//util_gen//server//output//"
sys.path.append(pydice_folder)
print(save_folder)

from ema_workbench import (Model, MultiprocessingEvaluator, Policy, Scenario)
from ema_workbench import (Constraint, RealParameter, IntegerParameter, ScalarOutcome, ArrayOutcome)
from ema_workbench.util import ema_logging

from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume)

ema_logging.log_to_stderr(ema_logging.INFO)
BaseEvaluator.reporting_frequency = 0.1

from PyRICE_V9_short_term_uncertainty import PyRICE
from model_outcomes_uncertainty_v2 import get_all_model_outcomes_uncertainty_search

all_policies = pd.read_csv("example_policys_principles.csv",index_col=0)

#define principle list
principles_list =["prioritarian","egalitarian","sufficitarian","utilitarian"]

total_policy_list = []

for principle in principles_list:
    policies = all_policies[all_policies['principle']==principle]
    policies = policies.dropna(axis='columns')
    policies = policies.iloc[:,:-1]
    policy_list_principle = []

    #get list of policies as input for uncertainty sampling
    for i in range(0,len(policies)):
        policy_dict = policies.iloc[i].to_dict()
        policy_list_principle.append(Policy(policies.index[i], **policy_dict)) 
    total_policy_list.append(policy_list_principle)

nfe = 35000

#IMPORTANT: principle done one by one because generation was split over various servers for computational burden
principle_index =0

if __name__ == "__main__":
    for principle in principles_list:
        print("uncertainty analysis started for: " + principles_list[principle_index] + " case for " + str(nfe) + " scenario's")

        model = PyRICE(model_specification="EMA", welfare_function = principle)
        RICE = Model('RICE', function = model)

        RICE.uncertainties =[IntegerParameter('fdamage',0,2),
                             IntegerParameter('t2xco2_index',0,999),
                             IntegerParameter('t2xco2_dist',0,2),
                             RealParameter('fosslim', 4000, 13649),

                             IntegerParameter('scenario_pop_gdp',0,5),
                             IntegerParameter('scenario_sigma',0,2),
                             IntegerParameter('scenario_cback',0,1),
                             IntegerParameter('scenario_elasticity_of_damages',0,2),
                             IntegerParameter('scenario_limmiu',0,1)]   
        
        #same for all formulations
        RICE.outcomes = get_all_model_outcomes_uncertainty_search(optimization_formulation = "utilitarian",horizon = 2105, precision = 10)

        ema_logging.log_to_stderr(ema_logging.INFO)
        
        #only needed on IPython console within Anaconda
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"     
        
        total_policies = len(total_policy_list[principle_index])
        policy_list = total_policy_list[principle_index]
        
        with MultiprocessingEvaluator(RICE) as evaluator:
            results = evaluator.perform_experiments(scenarios=nfe, policies=policy_list[:len(policy_list)//2])
           
            file_name =  save_folder + "results_uncertainty_analsysis_short_term_policies_4-8" + principles_list[principle_index] +"_runs_" + str(nfe) + ".tar.gz"
            save_results(results, file_name)

            print("uncertainty_analysis_short_term_first_half" + principles_list[principle_index] + " cycle completed")        
        
        with MultiprocessingEvaluator(RICE) as evaluator:
            results = evaluator.perform_experiments(scenarios=nfe, policies=policy_list[len(policy_list)//2:])
           
            file_name =  save_folder + "results_uncertainty_analsysis_short_term_policies_8-8" + principles_list[principle_index] +"_runs_" + str(nfe) + ".tar.gz"
            save_results(results, file_name)

            print("uncertainty_analysis_short_term " + principles_list[principle_index] + " cycle completed")
        principle_index = principle_index + 1

