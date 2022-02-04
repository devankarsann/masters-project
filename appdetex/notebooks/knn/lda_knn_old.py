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


# ### Import LDA document vectors

# In[26]:


LDA = pickle.load(open("../processing_files/LDA_mat.pickle", "rb"))
df_merged = pickle.load(open("../processing_files/df_merged.pickle", "rb"))


# In[27]:


df_merged.head()


# In[28]:


df_merged.shape


# In[29]:


LDA.shape


# ## knn

# In[30]:


get_ipython().run_cell_magic('time', '', "nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(LDA)\ndistances, indices = nbrs.kneighbors(LDA)")


# In[31]:


print(distances.shape)
print(indices.shape)


# In[32]:


distances[0]


# In[33]:


indices[0]


# In[34]:


import statistics

def intracluster_similarity(index):
    cluster_centroid = LDA[index]
    sum_dist = 0
    dist_list = []
    for i in indices[index][1:]:
        distance = np.linalg.norm(cluster_centroid-LDA[i])
        sum_dist += distance
        dist_list.append(distance)
        #print(index, ',', i, '=', distance)
        
    avg = sum_dist/(len(indices[index])-1)
    #print('average:', avg)
    
    variance = statistics.variance(dist_list)
    #print('variance:', variance)
    
    return avg, variance


# In[35]:


intracluster_similarity(0)


# In[36]:


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


# In[39]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic e
generate_raw_content_cluster_df(2)


# In[15]:


avg_list = []
variance_list = []

for i in range(LDA.shape[0]):
    avg, variance = intracluster_similarity(i)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[16]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[17]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[18]:


len(avg_list)


# In[19]:


len(variance_list)


# ## Intercluster Similarity

# In[20]:


#def intercluster_similarity(index):


# In[21]:


def calculate_centroid(index):
    cluster_centroid = LDA[index]
    for i in indices[index][1:]:
        cluster_centroid = np.add(cluster_centroid, LDA[i])
    return cluster_centroid/len(indices[index][1:])


# In[22]:


index_centroid = dict()
for i in range(LDA.shape[0]):
    index_centroid[i] = calculate_centroid(i)


# In[23]:


index_centroid


# In[24]:


#centroid_centroid_distance = {}
#for i in range(LDA.shape[0]):
#    centroid_centroid_distance[i] = {}
#    for j in range(LDA.shape[0]):
#        if i < j:
#            centroid_centroid_distance[i][j] = np.linalg.norm(index_centroid[i]-index_centroid[j])


# In[25]:


#centroid_centroid_distance


# In[ ]:




