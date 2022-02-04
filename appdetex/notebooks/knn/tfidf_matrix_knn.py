#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
from sklearn.preprocessing import normalize


# ## Libraries and Filesystem Setup

# In[2]:


import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy


# ### Import LDA document vectors

# In[4]:


#tfidf_matrix = np.load(open("../processed_files/tfidf_matrix.npy"))
tfidf_matrix = scipy.sparse.load_npz('../../processed_files/tfidf_matrix.npz')
df_merged = pickle.load(open("../../processed_files/df_merged.pickle", "rb"))


# In[5]:


df_merged.head()


# In[6]:


df_merged.shape


# In[7]:


print(tfidf_matrix.shape)
print(tfidf_matrix[0].shape)


# ## tfidf

# In[8]:


#from sklearn.preprocessing import normalize
#tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)


# In[10]:


tfidf_matrix.shape


# In[11]:


tfidf_matrix[0].shape


# In[9]:


new_tfidf_matrix = tfidf_matrix / np.linalg.norm(tfidf_matrix)


# In[26]:


new_tfidf_matrix = tfidf_matrix.todense() / np.linalg.norm(tfidf_matrix.todense())


# In[12]:


get_ipython().run_cell_magic('time', '', "nbrs = NearestNeighbors(n_neighbors = 10, algorithm = 'ball_tree').fit(tfidf_matrix)\ndistances, indices = nbrs.kneighbors(tfidf_matrix)")


# In[61]:


print(distances.shape)
print(indices.shape)


# In[62]:


distances[0]


# In[63]:


indices[0]


# In[76]:


import statistics

def intracluster_similarity(index):
    cluster_centroid = tfidf_matrix[index]
    dist_list = []
    for i in indices[index][1:]:
        distance = np.linalg.norm(cluster_centroid.toarray()-tfidf_matrix[i].toarray())
        dist_list.append(distance)
        #print(index, ',', i, '=', distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)
    
    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    #print('variance:', variance)
    
    return avg, variance


# In[77]:


intracluster_similarity(1234)


# In[55]:


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


# In[66]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic e
generate_raw_content_cluster_df(64)


# In[101]:


avg_list = []
variance_list = []

for i in range(tfidf_matrix.shape[0]):
    avg, variance = intracluster_similarity(i)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[102]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[79]:


# neighbors = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[103]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[80]:


# neighbors = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[81]:


print(len(avg_list), len(variance_list))


# ## Intercluster Similarity

# In[82]:


def calculate_centroid(index):
    cluster_centroid = tfidf_matrix[index]
    for i in indices[index][1:]:
        cluster_centroid = np.add(cluster_centroid, tfidf_matrix[i])
    return cluster_centroid/len(indices[index][1:])


# In[104]:


index_centroid = dict()
for i in range(tfidf_matrix.shape[0]):
    index_centroid[i] = calculate_centroid(i)


# In[86]:


index_centroid[0]


# In[24]:


#centroid_centroid_distance = {}
#for i in range(tfidf_matrix.shape[0]):
#    centroid_centroid_distance[i] = {}
#    for j in range(tfidf_matrix.shape[0]):
#        if i < j:
#            centroid_centroid_distance[i][j] = np.linalg.norm(index_centroid[i]-index_centroid[j])


# In[25]:


#centroid_centroid_distance


# In[92]:


from random import sample
def sample_mean_intercluster_dist(sample_size):
    indices_sample = sample(range(tfidf_matrix.shape[0]),sample_size)
    centroid_centroid_distance = {}
    for i in indices_sample:
        for j in indices_sample:
            if i < j:
                key = str(i) + "::" + str(j)
                centroid_centroid_distance[key] = np.linalg.norm(index_centroid[i].toarray()-index_centroid[j].toarray())
    return np.array(list(centroid_centroid_distance.values())).mean()


# # neighbors = 10

# ## 10 documents sample

# In[105]:


sample_10_data = [sample_mean_intercluster_dist(10) for i in range(5)]
print(sample_10_data)
print()
print(np.average(sample_10_data))


# ## 100 documents sample

# In[106]:


sample_100_data = [sample_mean_intercluster_dist(100) for i in range(5)]
print(sample_100_data)
print()
print(np.average(sample_100_data))


# ## 500 documents sample

# In[107]:


sample_500_data = [sample_mean_intercluster_dist(500) for i in range(5)]
print(sample_500_data)
print()
print(np.average(sample_500_data))


# # neighbors = 20

# ## 10 documents sample

# In[96]:


sample_10_data = [sample_mean_intercluster_dist(10) for i in range(5)]
print(sample_10_data)
print()
print(np.average(sample_10_data))


# ## 100 documents sample

# In[97]:


sample_100_data = [sample_mean_intercluster_dist(100) for i in range(5)]
print(sample_100_data)
print()
print(np.average(sample_100_data))


# ## 500 documents sample

# In[98]:


sample_500_data = [sample_mean_intercluster_dist(500) for i in range(5)]
print(sample_500_data)
print()
print(np.average(sample_500_data))


# In[ ]:




