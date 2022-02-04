#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# ## Libraries and Filesystem Setup

# In[2]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy


# ### Import LDA document vectors

# In[3]:


lda_10_matrix = np.load('../../processed_files/lda_10_matrix.npy')
lda_25_matrix = np.load('../../processed_files/lda_25_matrix.npy')
lda_50_matrix = np.load('../../processed_files/lda_50_matrix.npy')
df_merged = pickle.load(open("../../processed_files/df_merged.pickle", "rb"))


# In[5]:


lda_10_matrix.shape


# In[4]:


lda_25_matrix.shape


# In[5]:


lda_50_matrix.shape


# In[6]:


df_merged.head()


# In[7]:


df_merged.shape


# In[8]:


type(lda_10_matrix[0])


# In[9]:


type(lda_10_matrix[0])


# In[10]:


type(lda_10_matrix[0])


# In[11]:


np.average(lda_10_matrix[[0, 1, 2]], axis=0)


# ## DB Scan

# In[12]:


from sklearn.cluster import DBSCAN


# In[29]:


# eps (default 0.5): the maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples (default 5): number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself
# algorithm (default 'auto'): ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
lda_10_dbscan = DBSCAN(eps = 0.1, min_samples = 5).fit(lda_10_matrix)
lda_25_dbscan = DBSCAN(eps = 0.1, min_samples = 5).fit(lda_25_matrix)
lda_50_dbscan = DBSCAN(eps = 0.1, min_samples = 5).fit(lda_50_matrix)


# In[14]:


lda_10_dbscan.labels_


# In[30]:


lda_25_dbscan.labels_


# In[31]:


lda_50_dbscan.labels_


# In[32]:


lda_10_num_clusters = max(lda_10_dbscan.labels_)
lda_25_num_clusters = max(lda_25_dbscan.labels_)
lda_50_num_clusters = max(lda_50_dbscan.labels_)


# In[33]:


lda_10_num_clusters


# In[34]:


lda_25_num_clusters


# In[35]:


lda_50_num_clusters


# In[36]:


lda_10_index_cluster = zip(range(len(lda_10_dbscan.labels_)), lda_10_dbscan.labels_)
lda_10_index_cluster_dict = dict(lda_10_index_cluster)
lda_25_index_cluster = zip(range(len(lda_25_dbscan.labels_)), lda_25_dbscan.labels_)
lda_25_index_cluster_dict = dict(lda_25_index_cluster)
lda_50_index_cluster = zip(range(len(lda_50_dbscan.labels_)), lda_50_dbscan.labels_)
lda_50_index_cluster_dict = dict(lda_50_index_cluster)


# In[37]:


lda_10_index_cluster_dict[456]


# In[38]:


lda_25_index_cluster_dict[456]


# In[39]:


lda_50_index_cluster_dict[456]


# In[40]:


lda_10_cluster_index = list(zip(lda_10_dbscan.labels_, range(len(lda_10_dbscan.labels_))))
lda_25_cluster_index = list(zip(lda_25_dbscan.labels_, range(len(lda_25_dbscan.labels_))))
lda_50_cluster_index = list(zip(lda_50_dbscan.labels_, range(len(lda_50_dbscan.labels_))))


# In[43]:


lda_10_cluster_list = dict()
for i in range(lda_10_num_clusters):
    lda_10_cluster_list[i] = list(filter(lambda row: row[0] == i, lda_10_cluster_index))
lda_25_cluster_list = dict()
for i in range(lda_25_num_clusters):
    lda_25_cluster_list[i] = list(filter(lambda row: row[0] == i, lda_25_cluster_index))
lda_50_cluster_list = dict()
for i in range(lda_50_num_clusters):
    lda_50_cluster_list[i] = list(filter(lambda row: row[0] == i, lda_50_cluster_index))


# In[44]:


for i in range(lda_10_num_clusters):
    print('cluster: ', i, 'size: ', len(lda_10_cluster_list[i]))


# In[45]:


for i in range(lda_25_num_clusters):
    print('cluster: ', i, 'size: ', len(lda_25_cluster_list[i]))


# In[46]:


for i in range(lda_50_num_clusters):
    print('cluster: ', i, 'size: ', len(lda_50_cluster_list[i]))


# In[47]:


#len(cluster_list[0])


# In[48]:


#len(cluster_list[1])


# In[49]:


#len(cluster_list[2])


# In[50]:


#len(cluster_list[num_clusters-1])


# In[51]:


#cluster_list[1]


# In[52]:


#[i[1] for i in cluster_list[0]]


# In[60]:


import statistics

# index is cluster index
def lda_10_intracluster_similarity(index):
    cluster_centroid = np.average(lda_10_matrix[[i[1] for i in lda_10_cluster_list[index]]], axis=0)
    dist_list = []
    cluster = lda_10_cluster_list[index]
    for i in cluster:
        distance = np.linalg.norm(cluster_centroid-lda_10_matrix[i[1]])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)

    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    
    return avg, variance

# index is cluster index
def lda_25_intracluster_similarity(index):
    cluster_centroid = np.average(lda_25_matrix[[i[1] for i in lda_25_cluster_list[index]]], axis=0)
    dist_list = []
    cluster = lda_25_cluster_list[index]
    for i in cluster:
        distance = np.linalg.norm(cluster_centroid-lda_25_matrix[i[1]])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)

    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    
    return avg, variance

# index is cluster index
def lda_50_intracluster_similarity(index):
    cluster_centroid = np.average(lda_50_matrix[[i[1] for i in lda_50_cluster_list[index]]], axis=0)
    dist_list = []
    cluster = lda_50_cluster_list[index]
    for i in cluster:
        distance = np.linalg.norm(cluster_centroid-lda_50_matrix[i[1]])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)

    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    
    return avg, variance


# In[61]:


lda_10_intracluster_similarity(0)


# In[62]:


lda_25_intracluster_similarity(0)


# In[63]:


lda_50_intracluster_similarity(0)


# In[64]:


# index is cluster / cluster id
def generate_raw_content_cluster_10_df(index):
    index_list = lda_10_cluster_list[index]
    index_list = [x[1] for x in index_list]
    cluster_seed = df_merged.loc[index].to_frame().T
    #cluster_df = df_merged.loc[index_list[1:]]
    cluster_df = df_merged.loc[index_list[1:]]
    combined = pd.concat([cluster_seed, cluster_df])
    combined['cluster'] = index
    return combined

def generate_raw_content_cluster_25_df(index):
    index_list = lda_25_cluster_list[index]
    index_list = [x[1] for x in index_list]
    cluster_seed = df_merged.loc[index].to_frame().T
    #cluster_df = df_merged.loc[index_list[1:]]
    cluster_df = df_merged.loc[index_list[1:]]
    combined = pd.concat([cluster_seed, cluster_df])
    combined['cluster'] = index
    return combined

def generate_raw_content_cluster_50_df(index):
    index_list = lda_50_cluster_list[index]
    index_list = [x[1] for x in index_list]
    cluster_seed = df_merged.loc[index].to_frame().T
    #cluster_df = df_merged.loc[index_list[1:]]
    cluster_df = df_merged.loc[index_list[1:]]
    combined = pd.concat([cluster_seed, cluster_df])
    combined['cluster'] = index
    return combined


# In[65]:


generate_raw_content_cluster_10_df(101)


# In[67]:


generate_raw_content_cluster_25_df(10)


# In[68]:


generate_raw_content_cluster_50_df(10)


# In[69]:


lda_10_avg_list = []
lda_10_variance_list = []

for i in range(lda_10_num_clusters):
    avg, variance = lda_10_intracluster_similarity(i)
    lda_10_avg_list.append(avg)
    lda_10_variance_list.append(variance)
    #print()

lda_25_avg_list = []
lda_25_variance_list = []

for i in range(lda_25_num_clusters):
    avg, variance = lda_25_intracluster_similarity(i)
    lda_25_avg_list.append(avg)
    lda_25_variance_list.append(variance)
    #print()
    
lda_50_avg_list = []
lda_50_variance_list = []

for i in range(lda_50_num_clusters):
    avg, variance = lda_50_intracluster_similarity(i)
    lda_50_avg_list.append(avg)
    lda_50_variance_list.append(variance)
    #print()


# In[70]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(lda_10_avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[71]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(lda_25_avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[73]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(lda_50_avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# ## Intercluster Similarity

# In[74]:


def lda_10_intercluster_dist():
    indices = range(lda_10_num_clusters)
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                cluster_centroid_i = np.average(lda_10_matrix[[x[1] for x in lda_10_cluster_list[i]]], axis=0)
                cluster_centroid_j = np.average(lda_10_matrix[[x[1] for x in lda_10_cluster_list[j]]], axis=0)
                distance = np.linalg.norm(cluster_centroid_i - cluster_centroid_j)
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values

def lda_25_intercluster_dist():
    indices = range(lda_25_num_clusters)
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                cluster_centroid_i = np.average(lda_25_matrix[[x[1] for x in lda_25_cluster_list[i]]], axis=0)
                cluster_centroid_j = np.average(lda_25_matrix[[x[1] for x in lda_25_cluster_list[j]]], axis=0)
                distance = np.linalg.norm(cluster_centroid_i - cluster_centroid_j)
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values

def lda_50_intercluster_dist():
    indices = range(lda_50_num_clusters)
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                cluster_centroid_i = np.average(lda_50_matrix[[x[1] for x in lda_50_cluster_list[i]]], axis=0)
                cluster_centroid_j = np.average(lda_50_matrix[[x[1] for x in lda_50_cluster_list[j]]], axis=0)
                distance = np.linalg.norm(cluster_centroid_i - cluster_centroid_j)
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values


# In[75]:


intercluster_distances, values = lda_10_intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[76]:


intercluster_distances, values = lda_25_intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[77]:


intercluster_distances, values = lda_50_intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[ ]:




