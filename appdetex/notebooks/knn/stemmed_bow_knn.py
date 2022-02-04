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
from random import sample


# ### Import BOW document vectors

# In[3]:


#BOW = pickle.load(open("../processing_files/bow.pickle", "rb"))
#BOW = np.load("../../processed_files/bow/bow_matrix.npy")
#BOW = scipy.sparse.load_npz('../../processed_files/bow/bow_matrix.npz')
stemmed_BOW = scipy.sparse.load_npz('../../processed_files/bow/stemmed_bow_matrix.npz')


# In[5]:


#stemmed_BOW = np.load("../../processed_files/bow/bow_stemmed_matrix.npy")


# In[6]:


df_merged = pickle.load(open("../../processed_files/df_merged.pickle", "rb"))


# In[7]:


df_merged.head()


# In[8]:


df_merged.shape


# In[9]:


BOW.shape


# In[ ]:


#stemmed_BOW.shape


# ## knn

# In[ ]:


get_ipython().run_cell_magic('time', '', "nbrs50 = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(BOW)\ndistances50, indices50 = nbrs50.kneighbors(BOW)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "stemmed_nbrs50 = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(stemmed_BOW)\nstemmed_distances50, stemmed_indices50 = stemmed_nbrs50.kneighbors(stemmed_BOW)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "nbrs10 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(BOW)\ndistances10, indices10 = nbrs10.kneighbors(BOW)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "stemmed_nbrs10 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(stemmed_BOW)\nstemmed_distances10, stemmed_indices10 = stemmed_nbrs10.kneighbors(stemmed_BOW)")


# In[9]:


print(distances50.shape)
print(indices50.shape)


# In[10]:


print(distances10.shape)
print(distances10.shape)


# In[11]:


distances50[0]


# In[12]:


indices50[0]


# In[13]:


import statistics

def intracluster_similarity(index, indices):
    cluster_centroid = BOW[index].toarray()
    sum_dist = 0
    dist_list = []
    for i in indices[index][1:]:
        distance = np.linalg.norm(cluster_centroid-BOW[i].toarray())
        sum_dist += distance
        dist_list.append(distance)
        #print(index, ',', i, '=', distance)
        
    avg = sum_dist/(len(indices[index])-1)
    #print('average:', avg)
    
    variance = statistics.variance(dist_list)
    #print('variance:', variance)
    
    return avg, variance


# In[14]:


intracluster_similarity(0, indices50)


# In[18]:


def generate_raw_content_cluster_df(index, indices, distances):
    index_list = list(indices[index])
    distance_list = list(distances[index])
    cluster_seed = df_merged.loc[index].to_frame().T
    cluster_seed['DISTANCE'] = 0
    cluster_df = df_merged.loc[index_list[1:]]
    cluster_df['DISTANCE'] = distance_list[1:]
    combined = pd.concat([cluster_seed, cluster_df.sort_values(by='DISTANCE', ascending=True)])
    return combined.style.set_properties(subset=['RAW_CONTENT'], **{'width-min': '100px'})


# In[19]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic pages
generate_raw_content_cluster_df(2, indices50, distances50)


# In[20]:


avg_list = []
variance_list = []

for i in range(BOW.shape[0]):
    avg, variance = intracluster_similarity(i, indices50)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[21]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[22]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(variance_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[23]:


len(avg_list)


# In[24]:


len(variance_list)


# ## Intercluster Similarity

# In[25]:


#def intercluster_similarity(index):


# In[30]:


def calculate_centroid(index):
    cluster_centroid = BOW[index].toarray()
    for i in indices50[index][1:]:
        cluster_centroid = np.add(cluster_centroid, BOW[i].toarray())
    return cluster_centroid/len(indices50[index][1:])


# In[31]:


index_centroid = dict()
for i in range(BOW.shape[0]):
    index_centroid[i] = calculate_centroid(i)


# In[32]:


#centroid_centroid_distance = {}
#for i in range(BOW.shape[0]):
#    for j in range(BOW.shape[0]):
#        if i < j:
#            key = str(i) + "::" + str(j)
#            centroid_centroid_distance[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
#centroid_centroid_distance


# ## 10 documents sample

# In[33]:


indices10sample = sample(range(BOW.shape[0]),10)


# In[34]:


centroid_centroid_distance_10 = {}
for i in indices10sample:
    for j in indices10sample:
        if i < j:
            key = str(i) + "::" + str(j)
            centroid_centroid_distance_10[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
centroid_centroid_distance_10


# In[35]:


# 9, 6, 21, 10
np.array(list(centroid_centroid_distance_10.values())).mean()


# ## 100 documents samples

# In[36]:


indices100sample = sample(range(BOW.shape[0]),100)


# In[37]:


centroid_centroid_distance_100 = {}
for i in indices100sample:
    for j in indices100sample:
        if i < j:
            key = str(i) + "::" + str(j)
            centroid_centroid_distance_100[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
centroid_centroid_distance_100


# In[38]:


# 12, 46, 15, 17, 20
np.array(list(centroid_centroid_distance_100.values())).mean()


# ## 200 documents sample

# In[39]:


indices200sample = sample(range(BOW.shape[0]),200)


# In[40]:


centroid_centroid_distance_200 = {}
for i in indices200sample:
    for j in indices200sample:
        if i < j:
            key = str(i) + "::" + str(j)
            centroid_centroid_distance_200[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
centroid_centroid_distance_200


# In[41]:


# 23, 22, 25, 13, 19, 20
np.array(list(centroid_centroid_distance_200.values())).mean()


# ## 300 documents sample

# In[42]:


indices300sample = sample(range(BOW.shape[0]),300)


# In[43]:


centroid_centroid_distance_300 = {}
for i in indices300sample:
    for j in indices300sample:
        if i < j:
            key = str(i) + "::" + str(j)
            centroid_centroid_distance_300[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
centroid_centroid_distance_300


# In[44]:


# 17, 19, 24, 17
np.array(list(centroid_centroid_distance_300.values())).mean()


# ## 400 documents sample

# In[45]:


indices400sample = sample(range(BOW.shape[0]),400)


# In[46]:


centroid_centroid_distance_400 = {}
for i in indices400sample:
    for j in indices400sample:
        if i < j:
            key = str(i) + "::" + str(j)
            centroid_centroid_distance_400[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
centroid_centroid_distance_400


# In[47]:


# 22, 16, 17, 17
np.array(list(centroid_centroid_distance_400.values())).mean()


# ## 500 documents sample

# In[48]:


indices500sample = sample(range(BOW.shape[0]),500)


# In[49]:


centroid_centroid_distance_500 = {}
for i in indices500sample:
    for j in indices500sample:
        if i < j:
            key = str(i) + "::" + str(j)
            centroid_centroid_distance_500[key] = np.linalg.norm(index_centroid[i]-index_centroid[j])
centroid_centroid_distance_500


# In[50]:


# 15, 19, 17, 22, 18, 17
np.array(list(centroid_centroid_distance_500.values())).mean()


# In[ ]:




