import pandas as pd
import json
import os
from pandas.io.json import json_normalize
import ijson

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# reading the JSON data using json.load()
file = 'pubs_test.json'
with open(file) as f:
    d = json.load(f)

df2 = pd.DataFrame.from_dict(d, orient="index")
df2.reset_index(level=0, inplace=True)
auth_name = ['Qian Liu', 'Shuang Li', 'Shuai Wang', 'Meng Liu', 'Qing Wang', 'T. Suzuki', 'Peng Chen', 'M. Liu', 'Peng Zhao', 'N. Li', 'Qin Zhang', 'Min Wu', 'Meng Wang', 'Peng Gao', 'Q. Liu', 'M. Zhang', 'Qiang Liu', 'Nan Li', 'R. Zhang', 'Rui Li']

final_results = {}
for auth in range(9):
    print(df2['index'][auth])
    new = []
    for i in df2.columns[1:]:
        if df2[i][auth] is not None:
            t = {}
            for j in df2[i][auth]['authors']:
                if j['name'] == auth_name[auth]:
                    t['org'] = j['org']
            t['id'] = df2[i][auth]['id']
            new.append(t)
    new_df = pd.DataFrame(new)
    new_df.fillna(value='', inplace=True)
    X = CountVectorizer().fit_transform(new_df['org'])
    
    range_n_cluster = range(15,36)
    best_clusters = 0
    best_silh = 0
    for n_clusters in range_n_cluster:
        cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(X)
        silh = silhouette_score(X, cluster_labels)
        if silh > best_silh:
            best_silh = silh
            best_clusters = n_clusters
    
    new_df['cluster'] = KMeans(n_clusters=best_clusters).fit_predict(X)
    res = new_df.groupby('cluster')['id'].apply(list)
    k = df2['index'][auth]
    final_results[k] = res.tolist()

with open('result.json', 'w') as f:
    json.dump(final_results, f)