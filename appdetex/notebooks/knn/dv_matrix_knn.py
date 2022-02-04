#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# ## Libraries and Filesystem Setup

# In[2]:


import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse


# ### Import Doc2Vec document vectors

# In[3]:


dv_matrix = np.load(open("../processed_files/dv_matrix.npy", "rb"), allow_pickle=True)
df_merged = pickle.load(open("../processed_files/df_merged.pickle", "rb"))


# In[4]:


print(dv_matrix.shape)
print(dv_matrix[0].shape)


# In[5]:


#dv_mat = np.load('../processed_files/dv_matrix.npy', allow_pickle = True)


# In[6]:


dv_matrix[0]


# In[7]:


#dv_mat = np.matrix(dv_mat.tolist())
#dv_mat


# In[8]:


#for i in range(len(dv_mat)):
#    try:
#        #print(type(dv_matrix[i])
#        len(dv_matrix[i])
#    except:
#        dv_mat[i] = np.asmatrix(np.zeros(300))


# ## doc2vec

# In[9]:


from sklearn.preprocessing import normalize
dv_mat = normalize(dv_matrix, norm='l2', axis=1)


# In[10]:


get_ipython().run_cell_magic('time', '', "nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(dv_mat)\ndistances, indices = nbrs.kneighbors(dv_mat)")


# In[75]:


print(distances.shape)
print(indices.shape)


# In[76]:


distances[0]


# In[77]:


indices[0]


# In[126]:


import statistics

def intracluster_similarity(index):
    cluster_centroid = dv_mat[index]
    dist_list = []
    for i in indices[index][1:]:
        distance = np.linalg.norm(cluster_centroid-dv_mat[i])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)

    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    
    return avg, variance


# In[127]:


intracluster_similarity(0)


# In[80]:


def generate_raw_content_cluster_df(index):
    index_list = list(indices[index])
    distance_list = list(distances[index])
    cluster_seed = df_merged.loc[index].to_frame().T
    cluster_seed['DISTANCE'] = 0
    cluster_df = df_merged.loc[index_list[1:]]
    cluster_df['DISTANCE'] = distance_list[1:]
    combined = pd.concat([cluster_seed, cluster_df.sort_values(by='DISTANCE', ascending=True)])
    return combined.style.set_properties(subset=['RAW_CONTENT'], **{'width-min': '100px'})
    #return combined


# In[83]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic e
generate_raw_content_cluster_df(393)


# In[145]:


avg_list = []
variance_list = []

for i in range(dv_mat.shape[0]):
    avg, variance = intracluster_similarity(i)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[146]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[129]:


# neighbors = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[147]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[130]:


# neighbors = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[18]:


print(len(avg_list), len(variance_list))


# ## Intercluster Similarity

# In[134]:


def calculate_centroid(index):
    cluster_centroid = dv_mat[index]
    for i in indices[index][1:]:
        cluster_centroid = np.add(cluster_centroid, dv_mat[i])
    return cluster_centroid/len(indices[index][1:])


# In[148]:


index_centroid = dict()
for i in range(dv_mat.shape[0]):
    index_centroid[i] = calculate_centroid(i)


# In[136]:


index_centroid[0]


# In[24]:


#centroid_centroid_distance = {}
#for i in range(dv_mat.shape[0]):
#    centroid_centroid_distance[i] = {}
#    for j in range(dv_mat.shape[0]):
#        if i < j:
#            centroid_centroid_distance[i][j] = np.linalg.norm(index_centroid[i]-index_centroid[j])


# In[25]:


#centroid_centroid_distance


# In[139]:


from random import sample
def sample_mean_intercluster_dist(sample_size):
    indices_sample = sample(range(dv_mat.shape[0]),sample_size)
    centroid_centroid_distance = {}
    for i in indices_sample:
        for j in indices_sample:
            if i < j:
                key = str(i) + "::" + str(j)
                centroid_centroid_distance[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
    return np.array(list(centroid_centroid_distance.values())).mean()


# # neighbors = 10

# ## 10 documents sample

# In[149]:


sample_10_data = [sample_mean_intercluster_dist(10) for i in range(5)]
print(sample_10_data)
print()
print(np.average(sample_10_data))


# ## 100 documents sample

# In[150]:


sample_100_data = [sample_mean_intercluster_dist(100) for i in range(5)]
print(sample_100_data)
print()
print(np.average(sample_100_data))


# ## 500 documents sample

# In[151]:


sample_500_data = [sample_mean_intercluster_dist(500) for i in range(5)]
print(sample_500_data)
print()
print(np.average(sample_500_data))


# # neighbors = 20

# ## 10 documents sample

# In[140]:


sample_10_data = [sample_mean_intercluster_dist(10) for i in range(5)]
print(sample_10_data)
print()
print(np.average(sample_10_data))


# ## 100 documents sample

# In[141]:


sample_100_data = [sample_mean_intercluster_dist(100) for i in range(5)]
print(sample_100_data)
print()
print(np.average(sample_100_data))


# ## 500 documents sample

# In[142]:


sample_500_data = [sample_mean_intercluster_dist(500) for i in range(5)]
print(sample_500_data)
print()
print(np.average(sample_500_data))


# In[ ]:




