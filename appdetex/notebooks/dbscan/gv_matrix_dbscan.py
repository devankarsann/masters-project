#!/usr/bin/env python
# coding: utf-8

# In[3]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# ## Libraries and Filesystem Setup

# In[4]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy


# ### Import LDA document vectors

# In[6]:


gv_matrix = np.load('../../processed_files/gv_matrix.npy')
df_merged = pickle.load(open("../../processed_files/df_merged.pickle", "rb"))


# In[7]:


gv_matrix.shape


# In[8]:


df_merged.head()


# In[9]:


df_merged.shape


# In[10]:


type(gv_matrix[0])


# In[12]:


np.average(gv_matrix[[0, 1, 2]], axis=0)


# ## DB Scan

# In[13]:


from sklearn.cluster import DBSCAN


# In[14]:


# eps (default 0.5): the maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples (default 5): number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself
# algorithm (default 'auto'): ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
dbscan = DBSCAN(eps = 0.1, min_samples = 5).fit(gv_matrix)


# In[15]:


dbscan.labels_


# In[16]:


num_clusters = max(dbscan.labels_)
num_clusters


# In[17]:


dbscan


# In[18]:


index_cluster = zip(range(len(dbscan.labels_)), dbscan.labels_)
index_cluster_dict = dict(index_cluster)
index_cluster_dict[456]


# In[19]:


cluster_index = list(zip(dbscan.labels_, range(len(dbscan.labels_))))


# In[20]:


cluster_list = dict()
for i in range(num_clusters):
    cluster_list[i] = list(filter(lambda row: row[0] == i, cluster_index))


# In[21]:


for i in range(num_clusters):
    print('cluster: ', i, 'size: ', len(cluster_list[i]))


# In[22]:


len(cluster_list[0])


# In[23]:


len(cluster_list[1])


# In[24]:


len(cluster_list[2])


# In[25]:


len(cluster_list[num_clusters-1])


# In[26]:


cluster_list[1]


# In[28]:


[i[1] for i in cluster_list[0]]


# In[32]:


import statistics

# index is cluster index
def intracluster_similarity(index):
    cluster_centroid = np.average(gv_matrix[[i[1] for i in cluster_list[index]]], axis=0)
    dist_list = []
    cluster = cluster_list[index]
    for i in cluster:
        distance = np.linalg.norm(cluster_centroid-gv_matrix[i[1]])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)

    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    
    return avg, variance


# In[33]:


intracluster_similarity(0)


# In[34]:


intracluster_similarity(1)


# In[35]:


# index is cluster / cluster id
def generate_raw_content_cluster_df(index):
    index_list = cluster_list[index]
    index_list = [x[1] for x in index_list]
    cluster_seed = df_merged.loc[index].to_frame().T
    #cluster_df = df_merged.loc[index_list[1:]]
    cluster_df = df_merged.loc[index_list[1:]]
    combined = pd.concat([cluster_seed, cluster_df])
    combined['cluster'] = index
    return combined


# In[37]:


generate_raw_content_cluster_df(30)


# In[38]:


avg_list = []
variance_list = []

for i in range(num_clusters):
    avg, variance = intracluster_similarity(i)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[39]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# ## Intercluster Similarity

# In[40]:


def intercluster_dist():
    indices = range(num_clusters)
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                cluster_centroid_i = np.average(gv_matrix[[x[1] for x in cluster_list[i]]], axis=0)
                cluster_centroid_j = np.average(gv_matrix[[x[1] for x in cluster_list[j]]], axis=0)
                distance = np.linalg.norm(cluster_centroid_i - cluster_centroid_j)
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values


# In[41]:


intercluster_distances, values = intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[ ]:




