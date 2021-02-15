import pandas as pd
import time
import os
import numpy as np
import sys
import itertools

from ema_workbench import save_results, load_results

save_folder = os.path.dirname(os.getcwd()) + '\\6_Uncertainty Analysis\\output\\'
pydice_folder = os.path.dirname(os.getcwd()) + '\\6_Uncertainty Analysis\\model_server'
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

principles_list =["utilitarian","prioritarian","egalitarian","sufficitarian","nordhaus"]

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
#principle_index =0

if __name__ == "__main__":
    print("short term uncertainty analysis started for: " + "nordhaus" + " case for " + str(nfe) + " scenario's")

    model = PyRICE(model_specification="EMA", welfare_function = "utilitarian")
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

    with MultiprocessingEvaluator(RICE) as evaluator:
        results = evaluator.perform_experiments(scenarios=nfe, policies=total_policy_list[4])

        file_name =  save_folder + "results_uncertainty_analsysis_short_term_" + "nordhaus" +"_runs_" + str(nfe) + ".tar.gz"
        save_results(results, file_name)

        print("uncertainty_analysis_short_term " + "nordhaus" + " cycle completed")
    #principle_index = principle_index + 1


