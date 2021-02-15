from ema_workbench import (Model, CategoricalParameter,
                           ScalarOutcome, TimeSeriesOutcome, IntegerParameter, RealParameter)

from ema_workbench import (Policy, Scenario)

def get_all_model_outcomes_uncertainty_search(optimization_formulation = "utilitarian"):
    
    if optimization_formulation == "utilitarian":
        outcomes = [
            ScalarOutcome('Intratemporal utility GINI 2055', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2055', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2105', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2105', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2155', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2155', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2205', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2205', ScalarOutcome.INFO), 

            ScalarOutcome('Intratemporal utility GINI 2305', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2305', ScalarOutcome.INFO),

            ScalarOutcome('Intertemporal utility GINI', ScalarOutcome.INFO),
            ScalarOutcome('Intertemporal impact GINI', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2055', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Utility 2055', ScalarOutcome.MAXIMIZE),

            ScalarOutcome('Damages 2105', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Utility 2105', ScalarOutcome.MAXIMIZE),

            ScalarOutcome('Damages 2155', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Utility 2155', ScalarOutcome.MAXIMIZE),

            ScalarOutcome('Damages 2205', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Utility 2205', ScalarOutcome.MAXIMIZE),

            ScalarOutcome('Damages 2305', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Utility 2305', ScalarOutcome.MAXIMIZE),
            ScalarOutcome('Total Aggregated Utility', ScalarOutcome.MAXIMIZE),

            ScalarOutcome('Lowest income per capita 2055', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2055', ScalarOutcome.INFO),   

            ScalarOutcome('Lowest income per capita 2105',ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2105', ScalarOutcome.INFO),  

            ScalarOutcome('Lowest income per capita 2155', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2155', ScalarOutcome.INFO),   

            ScalarOutcome('Lowest income per capita 2205', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2205', ScalarOutcome.INFO),

            ScalarOutcome('Lowest income per capita 2305', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2305', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2055', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2055', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2105', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2105', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2155', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2155', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2205', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2205', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO), 

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO),     

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2055', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2055', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2055', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2105', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2105', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2105', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2205', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2205', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2205', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2305', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2305', ScalarOutcome.INFO),

        ]
    if optimization_formulation == "sufficitarian":
        outcomes = [
            ScalarOutcome('Intratemporal utility GINI 2055', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2055', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2105', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2105', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2155', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2155', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2205', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2205', ScalarOutcome.INFO), 

            ScalarOutcome('Intratemporal utility GINI 2305', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2305', ScalarOutcome.INFO),

            ScalarOutcome('Intertemporal utility GINI', ScalarOutcome.INFO),
            ScalarOutcome('Intertemporal impact GINI', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2055', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2055', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2105', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2105', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2155', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2155', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2205', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2205', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2305', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Aggregated Utility', ScalarOutcome.INFO),

            ScalarOutcome('Lowest income per capita 2055', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2055', ScalarOutcome.INFO),   

            ScalarOutcome('Lowest income per capita 2105',ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2105', ScalarOutcome.INFO),  

            ScalarOutcome('Lowest income per capita 2155', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2155', ScalarOutcome.INFO),   

            ScalarOutcome('Lowest income per capita 2205', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2205', ScalarOutcome.INFO),

            ScalarOutcome('Lowest income per capita 2305', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2305', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2055', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Population under treshold 2055', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Distance to treshold 2105', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Population under treshold 2105', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Distance to treshold 2155', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Population under treshold 2155', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Distance to treshold 2205', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Population under treshold 2205', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.MINIMIZE), 

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.MINIMIZE),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.MINIMIZE),     
            
            ScalarOutcome('Atmospheric Temperature 2055', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2055', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2055', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2105', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2105', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2105', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2205', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2205', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2205', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2305', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2305', ScalarOutcome.INFO),

        ]
    
    if optimization_formulation == "prioritarian":
        outcomes = [
            ScalarOutcome('Intratemporal utility GINI 2055', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2055', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2105', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2105', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2155', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2155', ScalarOutcome.INFO),

            ScalarOutcome('Intratemporal utility GINI 2205', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2205', ScalarOutcome.INFO), 

            ScalarOutcome('Intratemporal utility GINI 2305', ScalarOutcome.INFO),
            ScalarOutcome('Intratemporal impact GINI 2305', ScalarOutcome.INFO),

            ScalarOutcome('Intertemporal utility GINI', ScalarOutcome.INFO),
            ScalarOutcome('Intertemporal impact GINI', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2055', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2055', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2105', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2105', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2155', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2155', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2205', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2205', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2305', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Aggregated Utility', ScalarOutcome.INFO),

            ScalarOutcome('Lowest income per capita 2055', ScalarOutcome.MAXIMIZE),
            ScalarOutcome('Highest climate impact per capita 2055', ScalarOutcome.MINIMIZE),   

            ScalarOutcome('Lowest income per capita 2105',ScalarOutcome.MAXIMIZE),
            ScalarOutcome('Highest climate impact per capita 2105', ScalarOutcome.MINIMIZE),  

            ScalarOutcome('Lowest income per capita 2155', ScalarOutcome.MAXIMIZE),
            ScalarOutcome('Highest climate impact per capita 2155', ScalarOutcome.MINIMIZE),   

            ScalarOutcome('Lowest income per capita 2205', ScalarOutcome.MAXIMIZE),
            ScalarOutcome('Highest climate impact per capita 2205', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Lowest income per capita 2305', ScalarOutcome.MAXIMIZE),
            ScalarOutcome('Highest climate impact per capita 2305', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Distance to treshold 2055', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2055', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2105', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2105', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2155', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2155', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2205', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2205', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO), 

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO),     

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2055', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2055', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2055', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2105', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2105', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2105', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2205', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2205', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2205', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2305', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2305', ScalarOutcome.INFO),

        ]
        
    if optimization_formulation == "egalitarian":
        outcomes = [
            ScalarOutcome('Intratemporal utility GINI 2055', ScalarOutcome.MINIMIZE),
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
            ScalarOutcome('Intertemporal impact GINI', ScalarOutcome.MINIMIZE),

            ScalarOutcome('Damages 2055', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2055', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2105', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2105', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2155', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2155', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2205', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2205', ScalarOutcome.INFO),

            ScalarOutcome('Damages 2305', ScalarOutcome.INFO),
            ScalarOutcome('Utility 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Aggregated Utility', ScalarOutcome.INFO),

            ScalarOutcome('Lowest income per capita 2055', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2055', ScalarOutcome.INFO),   

            ScalarOutcome('Lowest income per capita 2105',ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2105', ScalarOutcome.INFO),  

            ScalarOutcome('Lowest income per capita 2155', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2155', ScalarOutcome.INFO),   

            ScalarOutcome('Lowest income per capita 2205', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2205', ScalarOutcome.INFO),

            ScalarOutcome('Lowest income per capita 2305', ScalarOutcome.INFO),
            ScalarOutcome('Highest climate impact per capita 2305', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2055', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2055', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2105', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2105', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2155', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2155', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2205', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2205', ScalarOutcome.INFO),

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO), 

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO),     

            ScalarOutcome('Distance to treshold 2305', ScalarOutcome.INFO),
            ScalarOutcome('Population under treshold 2305', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2055', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2055', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2055', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2105', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2105', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2105', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2205', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2205', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2205', ScalarOutcome.INFO),

            ScalarOutcome('Atmospheric Temperature 2305', ScalarOutcome.INFO),
            ScalarOutcome('Industrial Emission 2305', ScalarOutcome.INFO),
            ScalarOutcome('Total Output 2305', ScalarOutcome.INFO),

        ]
    return outcomes