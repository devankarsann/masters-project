#!/usr/bin/env python
# coding: utf-8

# # script arguments

# In[11]:


import sys

# argument list:
# - stemmed: true or false
# - representation: BOW, TFIDF, LDA_10, LDA_25, LDA_50, Word2Vec, Glove, Doc2Vec
# - eps: 0.1, 0.25, 0.5
# - min_samples: 5, 10, 20

stemmed = sys.argv[1]
representation = sys.argv[2]
eps = float(sys.argv[3])
min_sample = int(sys.argv[4])

if (stemmed not in ['true', 'false']):
    print('incorrect stemmed argument value: ' + stemmed)
    quit()
    
if (representation not in ['BOW', 'TFIDF', 'LDA_10', 'LDA_25', 'LDA_50', 'Word2Vec', 'Glove', 'Doc2Vec']):
    print('incorrect dbscan representation argument value: ' + representatioon)
    quit()
    
#if (eps not in [0.1, 0.25, 0.5]):
#    print('incorrect dbscan eps argument value: ' + str(eps))
#    quit()
    
#if (min_sample not in [5, 10, 20]):
#    print('incorrect dbscan min_sample argument value: ' + str(min_sample))
#    quit()


# In[4]:


#stemmed = 'true'

#representation = 'BOW' #
#representation = 'TFIDF' # 
#representation = 'LDA_10' # 
#representation = 'LDA_25' # 
#representation = 'LDA_50' # 
#representation = 'Word2Vec' # 
#representation = 'Glove' # 
#representation = 'Doc2Vec' # 

#eps = .02
#min_sample = 2


# In[2]:


#stemmed = 'false'

#representation = 'BOW' # 
#representation = 'TFIDF' # 
#representation = 'LDA_10' # 
#representation = 'LDA_25' # 
#representation = 'LDA_50' # 
#representation = 'Word2Vec' # 
#representation = 'Glove' # 
#representation = 'Doc2Vec' # 

#eps = 1.6
#min_sample = 17


# In[8]:


from clustering_and_metrics_dbscan import bow_model, tfidf_model, lda_10_model, lda_25_model, lda_50_model, w2v_model, gv_model, d2v_model
import numpy as np

import pickle
import time


# ## creating and saving result object

# In[5]:


from WebisDbscanMetricResults import DbscanMetricResults

def calculate_and_save_metrics(dbscan_results, time_1, time_2):
    # interdistance mean
    print('calculating interdistance mean')
    intercluster_distances = dbscan_results.intercluster_distances
    intercluster_values = dbscan_results.intercluster_values
    intercluster_mean = np.average(intercluster_values)
    
    # silhouette coefficient
    print('calculating silhouettte')
    a = np.average(dbscan_results.intra_avg_list)
    b = intercluster_mean
    silhouette_score = (b - a)/np.max([a, b])

   # silhouette coefficient per cluster
    silhouette_cluster_map = {}
    for cluster in range(len(dbscan_results.intra_avg_list)):
        a = dbscan_results.intra_avg_list[cluster]
        # print('================')
        # print(cluster)
        # print(intercluster_values)
        b = intercluster_values[cluster]
        score = (b - a)/np.max([a, b])
        silhouette_cluster_map[cluster] = score 
    
    # creating and saving result object
    print('creating metrics object')
    dbscanMetricResults = DbscanMetricResults(stemmed, representation, eps, min_sample)
    dbscanMetricResults.set_intra_metrics(dbscan_results.intra_avg_list, dbscan_results.intra_avg_map, dbscan_results.intra_variance_list, dbscan_results.intra_variance_map)
    dbscanMetricResults.set_inter_metrics(intercluster_values, intercluster_mean)
    dbscanMetricResults.set_silhouette(silhouette_score, silhouette_cluster_map)
    
    # calculate hits
    dbscan_results.calculate_total_hits()
    dbscan_results.calculate_paraphrase_hits()
    dbscan_results.calculate_plagiarize_hits()
    
    # save hits
    dbscanMetricResults.set_total_hit_metrics(dbscan_results.total_checks, dbscan_results.total_checks_map, dbscan_results.total_hits, dbscan_results.total_hits_map, dbscan_results.total_hit_percent, dbscan_results.total_hit_percent_map)
    dbscanMetricResults.set_paraphrase_hit_metrics(dbscan_results.paraphrase_checks, dbscan_results.paraphrase_checks_map, dbscan_results.paraphrase_hits, dbscan_results.paraphrase_hits_map, dbscan_results.paraphrase_hit_percent, dbscan_results.paraphrase_hit_percent_map)
    dbscanMetricResults.set_plagiarize_hit_metrics(dbscan_results.plagiarize_checks, dbscan_results.plagiarize_checks_map, dbscan_results.plagiarize_hits, dbscan_results.plagiarize_hits_map, dbscan_results.plagiarize_hit_percent, dbscan_results.plagiarize_hit_percent_map)
    
    # set computing time metrics
    time_3 = round(time.time() * 1000)
    dbscanMetricResults.set_compute_time(time_2-time_1, time_3-time_2)
    
    print('saving metrics object')
    dbscanMetricResults.save()


# ### clustering and metric values

# In[7]:


#d2v_dbscan.avg_list
#d2v_dbscan.variance_list
#d2v_dbscan.num_clusters
#d2v_dbscan.intercluster_values
#np.average(d2v_dbscan.intercluster_values)


# ## bag of words

# In[8]:


def new_bow_dbscan():
    bow_dbscan = bow_model(stemmed, eps, min_sample)
    bow_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    bow_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    bow_dbscan.calculate_intracluster_similarity()
    bow_dbscan.calculate_intercluster_dist()
    return bow_dbscan, time_1, time_2


# In[110]:


if representation == 'BOW':
    dbscan_results, time_1, time_2 = new_bow_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# ## tfidf matrix

# In[87]:


def new_tfidf_dbscan():
    tfidf_dbscan = tfidf_model(stemmed, eps, min_sample)
    tfidf_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    tfidf_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    tfidf_dbscan.calculate_intracluster_similarity()
    tfidf_dbscan.calculate_intercluster_dist()
    return tfidf_dbscan, time_1, time_2


# In[113]:


if representation == 'TFIDF':
    dbscan_results, time_1, time_2 = new_tfidf_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# # LDA

# ## lda 10 matrix

# In[9]:


def new_lda_10_dbscan():
    lda_10_dbscan = lda_10_model(stemmed, eps, min_sample)
    lda_10_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    lda_10_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    lda_10_dbscan.calculate_intracluster_similarity()
    lda_10_dbscan.calculate_intercluster_dist()
    return lda_10_dbscan, time_1, time_2


# In[10]:


if representation == 'LDA_10':
    dbscan_results, time_1, time_2 = new_lda_10_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# ### testing webis specific metric

# In[1]:


#dbscan_results.df_merged.head(10)


# In[2]:


#len(dbscan_results.df_merged[dbscan_results.df_merged['paraphrase'] == True])


# In[3]:


#len(dbscan_results.df_merged[dbscan_results.df_merged['paraphrase'] == False])


# In[52]:


#len(dbscan_results.cluster_list)


# In[53]:


#checks = 0
#hits = 0
#hit_percent = 0
#for cluster in range(len(dbscan_results.cluster_list)):
#    cluster_docs = [i[1] for i in dbscan_results.cluster_list[cluster]]
#    original_docs = [i+1 for i in cluster_docs if i % 2 == 0]
#    paraphrase_docs = [i for i in cluster_docs if i % 2 == 1]
#    checks += len(original_docs)
#    hits += len(set(original_docs).intersection(set(paraphrase_docs)))
    #print('cluster_docs: ', str(cluster_docs), '\n')
    #print('original_docs: ', str(original_docs), '\n')
    #print('paraphrase_docs: ', str(paraphrase_docs), '\n')
    #print('checks: ', str(checks), '\n')
    #print('hits: ', str(hits), '\n')
#hit_percent = hits / checks


# In[54]:


#checks, hits, hit_percent


# ## lda 25 matrix

# In[118]:


def new_lda_25_dbscan():
    lda_25_dbscan = lda_25_model(stemmed, eps, min_sample)
    lda_25_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    lda_25_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    lda_25_dbscan.calculate_intracluster_similarity()
    lda_25_dbscan.calculate_intercluster_dist()
    return lda_25_dbscan, time_1, time_2


# In[119]:


if representation == 'LDA_25':
    dbscan_results, time_1, time_2 = new_lda_25_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# ## lda 50 matrix

# In[121]:


def new_lda_50_dbscan():
    lda_50_dbscan = lda_50_model(stemmed, eps, min_sample)
    lda_50_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    lda_50_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    lda_50_dbscan.calculate_intracluster_similarity()
    lda_50_dbscan.calculate_intercluster_dist()
    return lda_50_dbscan, time_1, time_2


# In[122]:


if representation == 'LDA_50':
    dbscan_results, time_1, time_2 = new_lda_50_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# ## word2vec matrix

# In[124]:


def new_w2v_dbscan():
    w2v_dbscan = w2v_model(stemmed, eps, min_sample)
    w2v_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    w2v_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    w2v_dbscan.calculate_intracluster_similarity()
    w2v_dbscan.calculate_intercluster_dist()
    return w2v_dbscan, time_1, time_2


# In[125]:


if representation == 'Word2Vec':
    dbscan_results, time_1, time_2 = new_w2v_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# ## glove matrix

# In[15]:


def new_gv_dbscan():
    gv_dbscan = gv_model(stemmed, eps, min_sample)
    gv_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    gv_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    gv_dbscan.calculate_intracluster_similarity()
    gv_dbscan.calculate_intercluster_dist()
    return gv_dbscan, time_1, time_2


# In[16]:


if representation == 'Glove':
    dbscan_results, time_1, time_2 = new_gv_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# ## doc2vec matrix

# In[12]:


def new_d2v_dbscan():
    d2v_dbscan = d2v_model(stemmed, eps, min_sample)
    d2v_dbscan.load_data()
    time_1 = round(time.time() * 1000)
    d2v_dbscan.run_dbscan()
    time_2 = round(time.time() * 1000)
    d2v_dbscan.calculate_intracluster_similarity()
    d2v_dbscan.calculate_intercluster_dist()
    return d2v_dbscan, time_1, time_2


# In[13]:


if representation == 'Doc2Vec':
    dbscan_results, time_1, time_2 = new_d2v_dbscan()
    calculate_and_save_metrics(dbscan_results, time_1, time_2)


# In[ ]:




