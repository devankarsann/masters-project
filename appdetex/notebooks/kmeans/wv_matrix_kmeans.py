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


# ### Import Word2Vec document vectors

# In[3]:


wv_matrix = np.load("../../processed_files/w2v_matrix.npy", allow_pickle = True)
df_merged = pickle.load(open("../../processed_files/df_merged.pickle", "rb"))


# In[4]:


df_merged.head()


# In[5]:


df_merged.shape


# In[6]:


wv_matrix.shape


# In[7]:


#wv_matrix


# ## word2vec

# In[8]:


from sklearn.preprocessing import normalize
wv_matrix = normalize(wv_matrix, norm='l2', axis=1)


# In[9]:


num_clusters = 20
algorithm = 'full'


# In[10]:


get_ipython().run_cell_magic('time', '', 'from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=num_clusters, random_state=0, algorithm=algorithm).fit(wv_matrix)')


# In[11]:


kmeans.labels_


# In[12]:


index_cluster = zip(range(len(kmeans.labels_)), kmeans.labels_)
index_cluster_dict = dict(index_cluster)
index_cluster_dict[456]


# In[13]:


cluster_index = list(zip(kmeans.labels_, range(len(kmeans.labels_))))


# In[14]:


cluster_list = dict()
for i in range(num_clusters):
    cluster_list[i] = list(filter(lambda row: row[0] == i, cluster_index))


# In[15]:


cluster_list[0]


# In[16]:


print(kmeans.labels_.shape)
print(kmeans.cluster_centers_.shape)


# In[17]:


import statistics

# index is cluster index
def intracluster_similarity(index):
    cluster_centroid = kmeans.cluster_centers_[index]
    dist_list = []
    cluster = cluster_list[index]
    for i in cluster:
        distance = np.linalg.norm(cluster_centroid-wv_matrix[i[1]])
        dist_list.append(distance)
        
    #avg = sum_dist/(len(indices[index])-1)
    avg = np.average(dist_list)

    #variance = statistics.variance(dist_list)
    variance = np.var(dist_list)
    
    return avg, variance


# In[18]:


intracluster_similarity(0)


# In[19]:


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


# In[20]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic e
generate_raw_content_cluster_df(1)


# In[21]:


avg_list = []
variance_list = []

for i in range(num_clusters):
    avg, variance = intracluster_similarity(i)
    avg_list.append(avg)
    variance_list.append(variance)
    #print()


# In[22]:


# number of clusters = 20
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.hist(avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[23]:


print(len(avg_list), len(variance_list))


# ## Intercluster Similarity

# In[24]:


def intercluster_dist():
    indices = range(len(kmeans.cluster_centers_))
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                distance = np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j])
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values


# ## number of clusters = 20

# In[25]:


intercluster_distances, values = intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[ ]:




