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

distances = np.load('Intratemporal utility GINI _Egal_policy36TSC_30k_scen_damages_distances.npy')

distances = distances[0:20000,0:20000]

#calculate silhouette width

print("start agglomerative clustering")
sil_score_lst = []
start = time.time()
for n_clusters in range(2,11):
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