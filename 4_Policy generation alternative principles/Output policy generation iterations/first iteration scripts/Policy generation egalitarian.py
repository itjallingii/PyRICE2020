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

model = PyRICE(model_specification="EMA",welfare_function="egalitarian")
RICE = Model('RICE', function = model)

RICE.uncertainties =[IntegerParameter('fdamage',0,2),
                     IntegerParameter('scenario_pop_tfp',0,5),
                     IntegerParameter('scenario_sigma',0,5),
                     IntegerParameter('scenario_cback',0,2),
                     IntegerParameter('cback_to_zero',0,1),
                     RealParameter('fosslim', 4000.0, 13649),
                     RealParameter('limmiu',0.8,1.2)] 

RICE.levers = [RealParameter('sr', 0.1, 0.5),
               RealParameter('irstp', 0.001, 0.015),
               IntegerParameter('miu_period', 5, 30),
               IntegerParameter('egalitarian_discounting',0,1)]   #0 = no discouting , 1 = normal discounting,

RICE.outcomes =[ScalarOutcome('Intratemporal utility GINI 2055', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Intratemporal impact GINI 2055', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Intratemporal utility GINI 2105', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Intratemporal impact GINI 2105', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Intratemporal utility GINI 2155', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Intratemporal impact GINI 2155', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Intratemporal utility GINI 2205', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Intratemporal impact GINI 2205', ScalarOutcome.MINIMIZE), 
                
                ScalarOutcome('Intratemporal utility GINI 2305', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Intratemporal impact GINI 2305', ScalarOutcome.MINIMIZE),
                
                ScalarOutcome('Intertemporal utility GINI', ScalarOutcome.MINIMIZE),
                ScalarOutcome('Intertemporal impact GINI', ScalarOutcome.MINIMIZE)
               ]

epsilon_list = [0.001]
eps = []

for i in epsilon_list:
    k = np.ones((len(RICE.outcomes))) * i
    eps.append(k)
    
nfe = 200000

convergence_metrics = [HyperVolume(minimum=[0,0,0,0,0,0,0,0,0,0,0,0],
                                   maximum=[1,1,1,1,1,1,1,1,1,1,1,1]),
                       EpsilonProgress()]

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
                                                      convergence=convergence_metrics)
            
            results.to_csv("results_egalitarian_policy_gen_"+ str(epsilon_list[i]) +".csv")
            convergence.to_csv("con_egalitarian_policy_gen_" + str(epsilon_list[i]) +".csv")
            
        #results.to_csv("wcs.csv")
        #convergence.to_csv("wcs_con.csv")
        end = time.time()
        print('Egalitarian policy optimization 1 generation time is ' + str(round((end - start)/60)) + ' minutes', flush=True)

        