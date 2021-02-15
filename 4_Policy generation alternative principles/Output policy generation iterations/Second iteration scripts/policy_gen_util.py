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
               IntegerParameter('miu_period', 5, 30)]

RICE.outcomes =[ScalarOutcome('Utility 2055', ScalarOutcome.MAXIMIZE),
                ScalarOutcome('Utility 2105', ScalarOutcome.MAXIMIZE),
                ScalarOutcome('Utility 2155', ScalarOutcome.MAXIMIZE),
                ScalarOutcome('Utility 2205', ScalarOutcome.MAXIMIZE),
                ScalarOutcome('Utility 2305', ScalarOutcome.MAXIMIZE),
                ScalarOutcome('Total Aggregated Utility', ScalarOutcome.MAXIMIZE)]

range_of_levers = 5
range_irstp = list(np.arange(0.001, 0.0075, (0.0075-0.001)/range_of_levers))

print(range_irstp)

epsilon_list = [1,1,1,1,1,10]

nfe = 75000

convergence_metrics = [EpsilonProgress()]

constraints = [Constraint('Total Aggregated Utility', outcome_names='Total Aggregated Utility',
                          function=lambda x:max(500, -x))]

if __name__ == "__main__":
    for i in range(0,range_of_levers):    
        reference_scenario = Scenario('reference', **{'irstp' : range_irstp[i]})

        start = time.time()
        print("starting search for reference scenario: " + str(i) + "NFE = " + str(nfe))

        #only needed on IPython console within Anaconda
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

        with MultiprocessingEvaluator(RICE) as evaluator:
            results, convergence = evaluator.optimize(nfe=nfe,
                                                      searchover='levers',
                                                      epsilons=epsilon_list,
                                                      reference = reference_scenario,
                                                      convergence=convergence_metrics,
                                                      constraints = constraints)

            results.to_csv("//root//util_gen//server//output//results_utilitarian_policy_lower_irstp_gen_reference_scenario_" + str(i) + ".csv")
            convergence.to_csv("//root//util_gen//server//output//con_utilitarian_policy_lower_irstp_gen_reference_scenario_" + str(i) + ".csv")

        end = time.time()
        print("utilitarian policy generation time for scenario: " + str(i) + " - " + str(round((end - start)/60)) + ' minutes')

        