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
import scipy
from sklearn.preprocessing import normalize


# ### Import Glove document vectors

# In[3]:


#gv_matrix = scipy.sparse.load_npz('../processed_files/gv_matrix.npz')
df_merged = pickle.load(open("../processed_files/df_merged.pickle", "rb"))


# In[4]:


df_merged.head()


# In[5]:


df_merged.shape


# In[6]:


gv_matrix = np.load("../processed_files/gv_matrix.npy", allow_pickle = True)


# In[7]:


gv_matrix.shape


# In[8]:


from sklearn.preprocessing import normalize
gv_matrix = normalize(gv_matrix, norm = 'l2', axis = 1)


# ## gv_matrix

# In[9]:


get_ipython().run_cell_magic('time', '', "nbrs = NearestNeighbors(n_neighbors = 10, algorithm = 'ball_tree').fit(gv_matrix)\ndistances, indices = nbrs.kneighbors(gv_matrix)")


# In[77]:


indices[0]


# In[76]:


distances[0]


# In[91]:


import statistics

def intracluster_similarity(index):
    cluster_centroid = gv_mat[index]
    dist_list = []
    for i in indices[index][1:]:
        distance = np.linalg.norm(cluster_centroid-gv_mat[i])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)
    
    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    #print('variance:', variance)
    
    return avg, variance


# In[92]:


intracluster_similarity(6)


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


# In[81]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic e
generate_raw_content_cluster_df(51)


# In[110]:


avg_list = []
variance_list = []

for i in range(gv_mat.shape[0]):
    avg, variance = intracluster_similarity(i)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[111]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[94]:


# neighbors = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[113]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[95]:


# neighbors = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[96]:


print(len(avg_list), len(variance_list))


# ## Intercluster Similarity

# In[97]:


def calculate_centroid(index):
    cluster_centroid = gv_mat[index]
    for i in indices[index][1:]:
        cluster_centroid = np.add(cluster_centroid, gv_mat[i])
    return cluster_centroid/len(indices[index][1:])


# In[114]:


index_centroid = dict()
for i in range(gv_mat.shape[0]):
    index_centroid[i] = calculate_centroid(i)


# In[99]:


index_centroid[0]


# In[24]:


#centroid_centroid_distance = {}
#for i in range(gv_matrix.shape[0]):
#    centroid_centroid_distance[i] = {}
#    for j in range(gv_matrix.shape[0]):
#        if i < j:
#            centroid_centroid_distance[i][j] = np.linalg.norm(index_centroid[i]-index_centroid[j])


# In[25]:


#centroid_centroid_distance


# In[102]:


from random import sample
def sample_mean_intercluster_dist(sample_size):
    indices_sample = sample(range(gv_matrix.shape[0]),sample_size)
    centroid_centroid_distance = {}
    for i in indices_sample:
        for j in indices_sample:
            if i < j:
                key = str(i) + "::" + str(j)
                centroid_centroid_distance[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
    return np.array(list(centroid_centroid_distance.values())).mean()


# # neighbors = 10

# ## 10 documents sample

# In[115]:


sample_10_data = [sample_mean_intercluster_dist(10) for i in range(5)]
print(sample_10_data)
print()
print(np.average(sample_10_data))


# ## 100 documents sample

# In[116]:


sample_100_data = [sample_mean_intercluster_dist(100) for i in range(5)]
print(sample_100_data)
print()
print(np.average(sample_100_data))


# ## 500 documents sample

# In[117]:


sample_500_data = [sample_mean_intercluster_dist(500) for i in range(5)]
print(sample_500_data)
print()
print(np.average(sample_500_data))


# # neighbors = 20

# ## 10 documents sample

# In[106]:


sample_10_data = [sample_mean_intercluster_dist(10) for i in range(5)]
print(sample_10_data)
print()
print(np.average(sample_10_data))


# ## 100 documents sample

# In[107]:


sample_100_data = [sample_mean_intercluster_dist(100) for i in range(5)]
print(sample_100_data)
print()
print(np.average(sample_100_data))


# ## 500 documents sample

# In[108]:


sample_500_data = [sample_mean_intercluster_dist(500) for i in range(5)]
print(sample_500_data)
print()
print(np.average(sample_500_data))


# In[ ]:




