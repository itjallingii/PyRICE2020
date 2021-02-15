import numpy as np
import pandas as pd
import time

from ema_workbench.analysis import clusterer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from ema_workbench import save_results, load_results

import os
import sys

cd = os.path.dirname(os.getcwd())
sys.path.append(cd)
print(cd)

from ema_workbench import (perform_experiments, Model, Policy, RealParameter, 
                           IntegerParameter, ScalarOutcome, ema_logging, MultiprocessingEvaluator)

ema_logging.log_to_stderr(ema_logging.INFO)

timepoints_to_save = np.arange(2005,2105+10,10)                          

results_total_short_term = pd.read_csv("experiments_short_term_uncertainty_total.csv")

results_total_short_term['Economic scenario'] = ""
results_total_short_term.loc[results_total_short_term["scenario_pop_gdp"] == 0,'Economic scenario'] = "Nordhaus scenario"
results_total_short_term.loc[results_total_short_term["scenario_pop_gdp"] == 1,'Economic scenario'] = "SSP1"
results_total_short_term.loc[results_total_short_term["scenario_pop_gdp"] == 2,'Economic scenario'] = "SSP2"
results_total_short_term.loc[results_total_short_term["scenario_pop_gdp"] == 3,'Economic scenario'] = "SSP3"
results_total_short_term.loc[results_total_short_term["scenario_pop_gdp"] == 4,'Economic scenario'] = "SSP4"
results_total_short_term.loc[results_total_short_term["scenario_pop_gdp"] == 5,'Economic scenario'] = "SSP5"
results_total_short_term['policy_recoded']=results_total_short_term['principle'].astype(str).str[0:4] + "_"+ results_total_short_term['policy'] 

#objectives_list_timeseries_name =  ['Intratemporal utility GINI ','Intratemporal impact GINI ',
#                                    'Utility ','Damages ','Lowest income per capita ',
#                                    'Highest climate impact per capita ','Industrial Emission ','Total Output ','Atmospheric Temperature ',
#                                    'Population under treshold ','Distance to treshold ' ]

objectives_list_timeseries_name =  ['Damages ','Lowest income per capita ','Highest climate impact per capita ','Industrial Emission ',
                                    'Total Output ','Atmospheric Temperature ','Population under treshold ','Distance to treshold ' ]

timepoints_to_save = np.arange(2005,2105+10,10)                          

policy_list = ["Nord_nordhaus","Prio_policy19","Suff_policy30","Egal_policy36"]

for index in range(0,1):
    policy_selected = policy_list[index]
        
    cluster_data_35k = results_total_short_term[results_total_short_term['policy_recoded'] == policy_selected]
    data = cluster_data_35k.to_numpy() 
    
    #randomly select 15000 timeseries for clustering
    rows = np.array(list(range(15000)))
    
    #set seed so that all clustering is done on the same rows
    np.random.seed(0)
    selected_series = np.random.choice(rows, size=15000, replace=False)
    
    #get input data with select rows
    cluster_data_15k = data[selected_series]
    
    #construct original input dataframe with only rows from random selection (15k)
    input_data_15k = pd.DataFrame(data=cluster_data_15k,columns= cluster_data_35k.columns)
    input_data_15k = input_data_15k.set_index(input_data_15k.columns[0])
    
    #construct experiment dataframe with only scenario and policy columns
    experiments = input_data_15k.iloc[:,0:17]
    
    for objective in objectives_list_timeseries_name:
        outcomes = []
        for year in timepoints_to_save:
                name_year = objective + str(year)
                outcomes.append(name_year)
                
        print(outcomes)

        data = input_data_15k[outcomes].to_numpy(dtype =float)
        #data = data.float()
        print("started clustering for policy: " + policy_selected + "with: " + objective + "for: " + str(len(data)) + " timeseries")
        start = time.time()
        distances = clusterer.calculate_cid(data)
                
        end = time.time()
        print('Cluster time is ' + str(round((end - start))) + ' secondes')
        print("")
        
        #calculate silhouette width
        sil_score_lst = []
        start = time.time()
        for n_clusters in range(2,10):
            clusterers = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage="complete")
            cluster_labels = clusterers.fit_predict(distances)
            silhouette_avg = silhouette_score(distances, cluster_labels, metric="precomputed")
            sil_score_lst.append(silhouette_avg)
            
            print(objective + "_" + policy_selected + ": For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        
        end = time.time()
        print("")        
        print('Silhouette time is ' + str(round((end - start)/60)) + ' minutes')

        with open(objective + "_" + policy_selected + '_cluster_silhouette_width.txt', 'w') as f:
            for s in sil_score_lst:
                f.write(str(s) + "\n")
                
        #do agglomerative clustering on the distances
        start = time.time()
        for j in range(2, 8):
            clusters = clusterer.apply_agglomerative_clustering(distances,n_clusters=j)
            x = experiments.copy()
            x['clusters'] = clusters.astype('object')
            x.to_csv('TSC_15k_' + objective + "_" + policy_selected + '_cluster_' + str(j) + '.csv')
        end = time.time()
        print('Agglomerative clustering time is ' + str(round((end - start) / 60)) + ' minutes')
        print("")
        
        