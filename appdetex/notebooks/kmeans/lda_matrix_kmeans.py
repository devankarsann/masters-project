#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[7]:


lda_10_matrix = np.load('../../processed_files/lda_10_matrix.npy')
lda_25_matrix = np.load('../../processed_files/lda_25_matrix.npy')
lda_50_matrix = np.load('../../processed_files/lda_50_matrix.npy')
df_merged = pickle.load(open("../../processed_files/df_merged.pickle", "rb"))


# In[9]:


lda_10_matrix.shape


# In[10]:


lda_25_matrix.shape


# In[11]:


lda_50_matrix.shape


# In[12]:


df_merged.head()


# In[13]:


df_merged.shape


# In[15]:


type(lda_10_matrix[0])


# In[16]:


type(lda_25_matrix[0])


# In[17]:


type(lda_50_matrix[0])


# ## lda

# In[18]:


from sklearn.preprocessing import normalize


# In[19]:


num_clusters = 20
algorithm = 'full'


# In[21]:


from sklearn.cluster import KMeans
lda_10_kmeans = KMeans(n_clusters=num_clusters, random_state=0, algorithm=algorithm).fit(lda_10_matrix)
lda_25_kmeans = KMeans(n_clusters=num_clusters, random_state=0, algorithm=algorithm).fit(lda_25_matrix)
lda_50_kmeans = KMeans(n_clusters=num_clusters, random_state=0, algorithm=algorithm).fit(lda_50_matrix)


# In[10]:


lda_10_kmeans.labels_


# In[22]:


lda_25_kmeans.labels_


# In[23]:


lda_50_kmeans.labels_


# In[24]:


lda_10_index_cluster = zip(range(len(lda_10_kmeans.labels_)), lda_10_kmeans.labels_)
lda_10_index_cluster_dict = dict(lda_10_index_cluster)


# In[25]:


lda_25_index_cluster = zip(range(len(lda_25_kmeans.labels_)), lda_25_kmeans.labels_)
lda_25_index_cluster_dict = dict(lda_25_index_cluster)


# In[26]:


lda_50_index_cluster = zip(range(len(lda_50_kmeans.labels_)), lda_50_kmeans.labels_)
lda_50_index_cluster_dict = dict(lda_50_index_cluster)
lda_50_index_cluster_dict[456]


# In[27]:


lda_10_cluster_index = list(zip(lda_10_kmeans.labels_, range(len(lda_10_kmeans.labels_))))
lda_25_cluster_index = list(zip(lda_25_kmeans.labels_, range(len(lda_25_kmeans.labels_))))
lda_50_cluster_index = list(zip(lda_50_kmeans.labels_, range(len(lda_50_kmeans.labels_))))


# In[29]:


lda_10_cluster_list = dict()
for i in range(num_clusters):
    lda_10_cluster_list[i] = list(filter(lambda row: row[0] == i, lda_10_cluster_index))
lda_25_cluster_list = dict()
for i in range(num_clusters):
    lda_25_cluster_list[i] = list(filter(lambda row: row[0] == i, lda_25_cluster_index))
lda_50_cluster_list = dict()
for i in range(num_clusters):
    lda_50_cluster_list[i] = list(filter(lambda row: row[0] == i, lda_50_cluster_index))


# In[31]:


lda_10_cluster_list[0]


# In[32]:


print(lda_10_kmeans.labels_.shape)
print(lda_10_kmeans.cluster_centers_.shape)


# In[33]:


import statistics

# index is cluster index
def lda_10_intracluster_similarity(index):
    cluster_centroid = lda_10_kmeans.cluster_centers_[index]
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


# In[34]:


# index is cluster index
def lda_25_intracluster_similarity(index):
    cluster_centroid = lda_25_kmeans.cluster_centers_[index]
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


# In[35]:


# index is cluster index
def lda_50_intracluster_similarity(index):
    cluster_centroid = lda_50_kmeans.cluster_centers_[index]
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


# In[36]:


lda_10_intracluster_similarity(19)


# In[37]:


lda_25_intracluster_similarity(19)


# In[38]:


lda_50_intracluster_similarity(19)


# In[39]:


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


# In[40]:


# non-unique index errors
# 14 is a good example
# 21 can we stop output after distance goes from 3 to 10.583?
# 22 is lots of similar basic e
generate_raw_content_cluster_10_df(8)


# In[41]:


generate_raw_content_cluster_25_df(8)


# In[42]:


generate_raw_content_cluster_50_df(8)


# In[43]:


lda_10_avg_list = []
lda_10_variance_list = []

for i in range(num_clusters):
    avg, variance = lda_10_intracluster_similarity(i)
    lda_10_avg_list.append(avg)
    lda_10_variance_list.append(variance)


# In[44]:


lda_25_avg_list = []
lda_25_variance_list = []

for i in range(num_clusters):
    avg, variance = lda_25_intracluster_similarity(i)
    lda_25_avg_list.append(avg)
    lda_25_variance_list.append(variance)


# In[45]:


lda_50_avg_list = []
lda_50_variance_list = []

for i in range(num_clusters):
    avg, variance = lda_50_intracluster_similarity(i)
    lda_50_avg_list.append(avg)
    lda_50_variance_list.append(variance)


# In[46]:


bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(lda_10_avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[47]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(lda_25_avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# In[48]:


# neighbors = 10
bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.hist(lda_50_avg_list, bins=bins, edgecolor="k")
plt.xticks(bins)


# ## Intercluster Similarity

# In[49]:


def lda_10_intercluster_dist():
    indices = range(len(lda_10_kmeans.cluster_centers_))
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                distance = np.linalg.norm(lda_10_kmeans.cluster_centers_[i] - lda_10_kmeans.cluster_centers_[j])
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values

def lda_25_intercluster_dist():
    indices = range(len(lda_25_kmeans.cluster_centers_))
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                distance = np.linalg.norm(lda_25_kmeans.cluster_centers_[i] - lda_25_kmeans.cluster_centers_[j])
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values

def lda_50_intercluster_dist():
    indices = range(len(lda_50_kmeans.cluster_centers_))
    centroid_centroid_distance = {}
    values = []
    for i in indices:
        for j in indices:
            if i < j:
                key = str(i) + "::" + str(j)
                distance = np.linalg.norm(lda_50_kmeans.cluster_centers_[i] - lda_50_kmeans.cluster_centers_[j])
                centroid_centroid_distance[key] = distance
                values.append(distance)
    return centroid_centroid_distance, values


# ## number of clusters = 20

# In[50]:


intercluster_distances, values = lda_10_intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[51]:


intercluster_distances, values = lda_25_intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[52]:


intercluster_distances, values = lda_50_intercluster_dist()
#print('distances', values)
print('average', np.average(values))


# In[ ]:




