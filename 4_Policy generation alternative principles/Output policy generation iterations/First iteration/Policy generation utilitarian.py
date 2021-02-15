import pandas as pd
import time
import os
import numpy as np

os.chdir(os.getcwd())
import sys

# insert at 1, 0 is the script path (or '' in REPL)
pydice_folder = os.path.dirname(os.getcwd())+"\\1_Model"

print(pydice_folder)
sys.path.insert(1,pydice_folder)

from ema_workbench import (Model, Constraint, Scenario, RealParameter, IntegerParameter, ScalarOutcome, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging

from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume)

ema_logging.log_to_stderr(ema_logging.INFO)
BaseEvaluator.reporting_frequency = 0.1
# ema_logging.log_to_stderr(ema_logging.DEBUG)

from PyRICE_V8 import PyRICE

model = PyRICE(model_specification="EMA",welfare_function="utilitarian")
RICE = Model('RICE', function = model)

RICE.uncertainties =[IntegerParameter('fdamage',0,1),
                     IntegerParameter('scenario_pop_tfp',0,5),
                     IntegerParameter('scenario_sigma',0,5),
                     IntegerParameter('scenario_cback',0,2),
                         
                     IntegerParameter('cback_to_zero',0,1),
                     RealParameter('fosslim', 4000.0, 13649),
                     RealParameter('limmiu',0.8,1.2)] 
                        
RICE.levers = [RealParameter('sr', 0.1, 0.5),
               RealParameter('irstp', 0.001, 0.015),
               IntegerParameter('miu_period', 5, 30)]

RICE.outcomes =[ScalarOutcome('Damages 2055', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Utility 2055', ScalarOutcome.MAXIMIZE),
                
                ScalarOutcome('Damages 2105', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Utility 2105', ScalarOutcome.MAXIMIZE),
                
                ScalarOutcome('Damages 2155', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Utility 2155', ScalarOutcome.MAXIMIZE),
                
                ScalarOutcome('Damages 2205', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Utility 2205', ScalarOutcome.MAXIMIZE),
                
                ScalarOutcome('Damages 2305', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Utility 2305', ScalarOutcome.MAXIMIZE),
                ScalarOutcome('Total Aggregated Utility', ScalarOutcome.MAXIMIZE)
               ]


epsilon_list = [0.01]
eps = []

for i in epsilon_list:
    k = np.ones((len(RICE.outcomes))) * i
    eps.append(k)
    
nfe = 200
convergence_metrics = [EpsilonProgress()]

convergence_metrics = [HyperVolume(minimum=[0,-10000,0,-10000,0,-10000,0,-10000,0,-10000,-5000],
                                   maximum=[500,10000,500,10000,500,10000,500,10000,500,10000,20000]),
                       EpsilonProgress()]

constraints = [Constraint("Utility 2305", outcome_names="Utility 2305",
                          function=lambda x:max(0, -x))]

if __name__ == "__main__":
    for i in range(len(epsilon_list)):    
        start = time.time()
        print("used epsilon is: " +  str(epsilon_list[i]))
        print("starting search for optimization 1 - policy generation cycle: " + str(i+1) + "/" + str(len(epsilon_list)), flush=True)
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        with MultiprocessingEvaluator(RICE) as evaluator:
            results, convergence = evaluator.optimize(nfe=nfe,
                                                      searchover='levers',
                                                      epsilons=eps[i],
                                                      constraints = constraints,
                                                      convergence=convergence_metrics)
            
            results.to_csv("results_utilitarian_policy_gen_"+ str(epsilon_list[i]) +".csv")
            convergence.to_csv("con_utilitarian_policy_gen_" + str(epsilon_list[i]) +".csv")
            
        #results.to_csv("wcs.csv")
        #convergence.to_csv("wcs_con.csv")
        end = time.time()
        print('utilitarian policy optimization 1 generation time is ' + str(round((end - start)/60)) + ' minutes', flush=True)

#worst case climate sensitivity and damage function scenario from Shajee
#reference_scenario = Scenario('reference', **{'t2xco2_index': 257, 't2xco2_dist': 2,'fdamage': 2})

#if __name__ == "__main__":
#    for i in range(len(epsilon_list)):    
#        start = time.time()
#        print("used epsilon is: " +  str(epsilon_list[i]))
#       print("starting search for optimization 2 - policy generation cycle: " + str(i+1) + "/" + str(len(epsilon_list)), flush=True)
        
        #only needed on IPython console within Anaconda
#        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        
#        with MultiprocessingEvaluator(RICE) as evaluator:
#            results, convergence = evaluator.optimize(nfe=nfe,
#                                                      searchover='levers',
#                                                      epsilons=eps[i],
#                                                      convergence=convergence_metrics,
#                                                      constraints=constraints,
#                                                      reference=reference_scenario)
#            
#            results.to_csv("results_utilitarian_policy_gen_worst_case_ref1"+ str(epsilon_list[i]) +".csv")
#            convergence.to_csv("con_utilitarian_policy_gen_worst_case_ref1" + str(epsilon_list[i]) +".csv")
#
#        end = time.time()
#        print('utilitarian policy optimization 2 generation time is '+ str(round((end - start)/60)) + ' minutes', flush=True)
        