from ema_workbench import (Model, CategoricalParameter,ArrayOutcome,
                           ScalarOutcome, TimeSeriesOutcome, IntegerParameter, RealParameter)

from ema_workbench import (Policy, Scenario)

import numpy as np
import pandas as pd

def get_all_model_outcomes_uncertainty_search(optimization_formulation = "utilitarian",horizon = 2305, precision =20):
    timepoints_to_save = np.arange(2005,horizon+precision,precision)
    if optimization_formulation == "utilitarian":
        
            objectives_list_name = ['Intertemporal utility GINI','Intertemporal impact GINI','Total Aggregated Utility','Regions below treshold']
            
            objectives_list_timeseries_name = ['Damages ','Utility ',
                        'Lowest income per capita ','Highest climate impact per capita ',
                        'Distance to treshold ','Population under treshold ',
                        'Intratemporal utility GINI ','Intratemporal impact GINI ',
                        'Atmospheric Temperature ', 'Industrial Emission ', 'Total Output ']

            supplementary_list_timeseries_name = ['CPC ','Population ']
            supplementary_list_quintile_name = ['CPC pre damage ','CPC post damage ']
        
            outcomes = []
            for name in objectives_list_timeseries_name:
                for year in timepoints_to_save:
                    name_year = name + str(year)
                   
                    outcome = ScalarOutcome(name_year, ScalarOutcome.INFO)
                    outcomes.append(outcome)

            for name in objectives_list_name:
                if name == "Regions below treshold":
                    outcome = ArrayOutcome(name)
                    
                else:
                    outcome = ScalarOutcome(name, ScalarOutcome.INFO)
                    outcomes.append(outcome)
                

            for name in supplementary_list_timeseries_name:
                for year in timepoints_to_save:
                    name_year = name + str(year)
                    outcome = ArrayOutcome(name_year)
                    outcomes.append(outcome)

            for name in supplementary_list_quintile_name:
                for year in timepoints_to_save:
                    name_year = name + str(year)
                    outcome = ArrayOutcome(name_year)
                    outcomes.append(outcome)
    return outcomes