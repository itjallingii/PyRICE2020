import pandas as pd
import time
import os
import numpy as np
import sys
import itertools
# insert at 1, 0 is the script path (or '' in REPL)
#os.chdir('/root/my_project_dr/utilitarian_opti/server/1_Model')
#path = os.getcwd()
#print(path)

pydice_folder = os.path.dirname(os.getcwd()) + '//server//1_Model'
save_folder = os.path.dirname(os.getcwd()) + '//server//output'
sys.path.append(pydice_folder)

print(pydice_folder)

from ema_workbench import (Model, Constraint, Scenario, RealParameter, IntegerParameter, ScalarOutcome, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging

from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume)

ema_logging.log_to_stderr(ema_logging.INFO)
BaseEvaluator.reporting_frequency = 0.1
# ema_logging.log_to_stderr(ema_logging.DEBUG)

from PyRICE_V8 import PyRICE

model = PyRICE(model_specification="EMA",welfare_function="sufficitarian")
RICE = Model('RICE', function = model)

RICE.uncertainties =[IntegerParameter('fdamage',0,1),
                     IntegerParameter('scenario_pop_tfp',0,5),
                     IntegerParameter('scenario_sigma',0,5),
                     IntegerParameter('scenario_cback',0,2),
                         
                     IntegerParameter('cback_to_zero',0,1),
                     RealParameter('fosslim', 4000, 13649),
                     RealParameter('limmiu',0.8,1.2)] 
                        
RICE.levers = [RealParameter('sr', 0.1, 0.5),
               RealParameter('irstp',  0.001, 0.015),
               IntegerParameter('miu_period', 5, 30),
    
               IntegerParameter('sufficitarian_discounting', 0,1)]

RICE.outcomes =[ScalarOutcome('Distance to treshold 2055', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Population under treshold 2055', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Distance to treshold 2105', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Population under treshold 2105', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Distance to treshold 2155', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Population under treshold 2155', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Distance to treshold 2205', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Population under treshold 2205', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Distance to treshold 2305', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Population under treshold 2305', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Total Aggregated Utility',ScalarOutcome.MAXIMIZE)
               ]
range_of_levers = 5

range_growth_factor = np.arange(1, 1.044, (1.04-1)/range_of_levers)
range_treshold_level = np.arange(0.7, 2.57, (2.4-0.7)/range_of_levers)
range_growth_factor = list(range_growth_factor)
range_treshold_level = list(range_treshold_level)

reference_combinations = list(itertools.product(range_growth_factor, range_treshold_level))

nfe = 10000

epsilon_list = [0.0001,0.1,0.0001,0.1,0.0001,0.1,0.0001,0.1,0.0001,0.1,1]

convergence_metrics = [EpsilonProgress()]

constraints = [Constraint('Total Aggregated Utility', outcome_names='Total Aggregated Utility',
                          function=lambda x:max(0, -x))]

if __name__ == "__main__":
    for i in range(0,len(reference_combinations)):    
        reference_scenario = Scenario('reference', **{'growth_factor_suf' : reference_combinations[i][1], 
                                                      'sufficitarian_discounting' : reference_combinations[i][0]})

        start = time.time()
        print("starting search for reference scenario: " + str(i))

        #only needed on IPython console within Anaconda
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

        with MultiprocessingEvaluator(RICE) as evaluator:
            results, convergence = evaluator.optimize(nfe=nfe,
                                                      searchover='levers',
                                                      epsilons=epsilon_list,
                                                      reference = reference_scenario,
                                                      convergence=convergence_metrics,
                                                      constraints=constraints)

            results.to_csv("//root//util_gen//server//output//results_suffici_policy_gen_reference_scenario_" + str(i) + ".csv")
            convergence.to_csv("//root//util_gen//server//output//con_suffici_policy_gen_reference_scenario_" + str(i) + ".csv")

        end = time.time()
        print('sufficitarian policy generation time for scenario: ' + str(i) + " - " + str(round((end - start)/60)) + ' minutes')
