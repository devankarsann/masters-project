#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


from clustering_and_metrics_knn import stemmed_bow_model, stemmed_tfidf_model, stemmed_lda_10_model, stemmed_lda_25_model, stemmed_lda_50_model, stemmed_w2v_model, stemmed_gv_model, stemmed_d2v_model
import numpy as np
import pickle


# ### minimize computation while working on new functionality such as metrics
# ### enable skipping initial processing steps and loading object from disk 

# In[3]:


refresh = True


# ## bag of words

# In[4]:


def new_stemmed_bow_knn():
    stemmed_bow_knn = stemmed_bow_model('ball_tree')
    stemmed_bow_knn.load_data()
    stemmed_bow_knn.run_knn()
    stemmed_bow_knn.calculate_intra_average_and_variance()
    stemmed_bow_knn.generate_index_centroid_map()
    
    #print('saving bow_stemmed_knn')
    #with open('../../processed_files/bow_stemmed_knn.pickle', 'wb') as file:
    #    pickle.dump(bow_stemmed_knn, file)
    
    return stemmed_bow_knn


# In[5]:


stemmed_bow_knn = new_stemmed_bow_knn()


# In[7]:


stemmed_bow_knn_inter_dist_sample = [stemmed_bow_knn.sample_mean_intercluster_dist(500) for i in range(5)]
#bow_knn_inter_dist_sample = [0.9016629769778474, 0.9006886090554161, 0.8923767494843687, 0.9026806522086671, 0.8988289848130686]
stemmed_bow_knn_inter_dist_sample


# In[8]:


stemmed_bow_knn_inter_dist_mean = np.average(stemmed_bow_knn_inter_dist_sample)
#bow_knn_inter_dist_mean = 0.8992475945078736
stemmed_bow_knn_inter_dist_mean


# ## tfidf matrix

# In[4]:


def new_tfidf_knn():
    tfidf_knn = tfidf_model('ball_tree')
    tfidf_knn.load_data()
    tfidf_knn.run_knn()
    tfidf_knn.calculate_intra_average_and_variance()
    tfidf_knn.generate_index_centroid_map()
    
    # cannot serialize more than 4 Gib
    #print('saving tfidf_knn')
    #with open('../../processed_files/tfidf_knn.pickle', 'wb') as file:
    #    pickle.dump(tfidf_knn, file)
    
    return tfidf_knn


# In[ ]:


tfidf_knn = new_tfidf_knn()


# In[10]:


tfidf_knn_inter_dist_sample = [tfidf_knn.sample_mean_intercluster_dist(500) for i in range(5)]
#tfidf_knn_inter_dist_sample = [0.9016629769778474, 0.9006886090554161, 0.8923767494843687, 0.9026806522086671, 0.8988289848130686]
tfidf_knn_inter_dist_sample


# In[11]:


tfidf_knn_inter_dist_mean = np.average(tfidf_knn_inter_dist_sample)
#tfidf_knn_inter_dist_mean = 0.8992475945078736
tfidf_knn_inter_dist_mean


# # LDA

# In[ ]:


#def new_lda_knn(matrix_name):
#    lda_knn = lda_10_model(matrix_name, 'ball_tree')
#    lda_knn.load_data()
#    lda_knn.run_knn()
#    lda_knn.calculate_intra_average_and_variance()
#    lda_knn.generate_index_centroid_map()
#    
#    print('saving ', matrix_name)
#    with open('processed_files/' + matrix_name + '.pickle', 'wb') as file:
#        pickle.dump(lda_knn, file)
#    
#    return lda_knn


# In[ ]:


#row_sums = lda_10_knn.matrix.sum(axis = 1)
#copy_matrix = lda_10_knn.matrix.copy()
#asdf = np.divide(copy_matrix, row_sums.reshape(len(row_sums), 1))


# In[ ]:


#row_sums = lda_10_knn.matrix.sum(axis = 1)


# In[ ]:


#row_sums[7]


# In[ ]:


#copy_matrix = lda_10_knn.matrix.copy()


# In[ ]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(copy_matrix)
#new_lda_10_knn_matrix = scaler.transform(copy_matrix)
#new_lda_10_knn_matrix = scaler.fit_transform(copy_matrix)


# In[ ]:


#X_std = (lda_10_knn.matrix - lda_10_knn.matrix.min(axis=0)) / (lda_10_knn.matrix.max(axis=0) - lda_10_knn.matrix.min(axis=0))
#X_scaled = X_std * (1 - 0) + 0


# In[ ]:


#X_std.shape


# In[ ]:


#new_row_sums = new_lda_10_knn_matrix.sum(axis = 1)


# In[ ]:


#new_row_sums[34]


# In[ ]:





# ## lda 10 matrix

# In[12]:


def new_lda_10_knn():
    lda_10_knn = lda_10_model('ball_tree')
    lda_10_knn.load_data()
    lda_10_knn.run_knn()
    lda_10_knn.calculate_intra_average_and_variance()
    lda_10_knn.generate_index_centroid_map()
    
    print('saving lda_10_knn')
    with open('../../processed_files/lda_10_knn.pickle', 'wb') as file:
        pickle.dump(lda_10_knn, file)
    
    return lda_10_knn


# In[13]:


lda_10_knn = None
if refresh == False:
    try:
        print('read lda_10_knn object from memory')
        lda_10_knn = pickle.load(open("../../processed_files/lda_10_knn.pickle", "rb"))
    except:
        print('error reading lda_10_knn from memory, running processing steps again')
        lda_10_knn = new_lda_10_knn()
else:
    # run processing steps again
    print('refresh == true, running lda_10_knn processing steps')
    lda_10_knn = new_lda_10_knn()


# In[14]:


#lda_10_knn.index_centroid.shape


# In[15]:


lda_10_knn_inter_dist_sample = [lda_10_knn.sample_mean_intercluster_dist(500) for i in range(5)]
lda_10_knn_inter_dist_sample


# In[16]:


lda_10_knn_inter_dist_mean = np.average(lda_10_knn_inter_dist_sample)
lda_10_knn_inter_dist_mean


# ## lda 25 matrix

# In[17]:


#refresh = False


# In[20]:


def new_lda_25_knn():
    lda_25_knn = lda_25_model('ball_tree')
    lda_25_knn.load_data()
    lda_25_knn.run_knn()
    lda_25_knn.calculate_intra_average_and_variance()
    lda_25_knn.generate_index_centroid_map()
    
    print('saving lda_25_knn')
    with open('../../processed_files/lda_25_knn.pickle', 'wb') as file:
        pickle.dump(lda_25_knn, file)
    
    return lda_25_knn


# In[21]:


lda_25_knn = None
if refresh == False:
    try:
        print('read lda_25_knn object from memory')
        lda_25_knn = pickle.load(open("../../processed_files/lda_25_knn.pickle", "rb"))
    except:
        print('error reading lda_knn from memory, running processing steps again')
        lda_25_knn = new_lda_25_knn()
else:
    # run processing steps again
    print('refresh == true, running lda_25_knn processing steps')
    lda_25_knn = new_lda_25_knn()


# In[22]:


lda_25_knn_inter_dist_sample = [lda_25_knn.sample_mean_intercluster_dist(500) for i in range(5)]
lda_25_knn_inter_dist_sample


# In[23]:


lda_25_knn_inter_dist_mean = np.average(lda_25_knn_inter_dist_sample)
lda_25_knn_inter_dist_mean


# ## lda 50 matrix

# In[26]:


def new_lda_50_knn():
    lda_50_knn = lda_50_model('ball_tree')
    lda_50_knn.load_data()
    lda_50_knn.run_knn()
    lda_50_knn.calculate_intra_average_and_variance()
    lda_50_knn.generate_index_centroid_map()
    
    print('saving lda_50_knn')
    with open('../../processed_files/lda_50_knn.pickle', 'wb') as file:
        pickle.dump(lda_50_knn, file)
    
    return lda_50_knn


# In[27]:


lda_50_knn = None
if refresh == False:
    try:
        print('read lda_50_knn object from memory')
        lda_50_knn = pickle.load(open("../../processed_files/lda_50_knn.pickle", "rb"))
    except:
        print('error reading lda_50_knn from memory, running processing steps again')
        lda_50_knn = new_lda_knn()
else:
    # run processing steps again
    print('refresh == true, running lda_50_knn processing steps')
    lda_50_knn = new_lda_50_knn()


# In[28]:


lda_50_knn_inter_dist_sample = [lda_50_knn.sample_mean_intercluster_dist(500) for i in range(5)]
lda_50_knn_inter_dist_sample


# In[29]:


lda_50_knn_inter_dist_mean = np.average(lda_50_knn_inter_dist_sample)
lda_50_knn_inter_dist_mean


# In[ ]:


#from jqmcvi import base
#cluster_list = [clus0.values, clus1.values, clus2.values]
#base.dunn(cluster_list)


# ## word2vec matrix

# In[32]:


def new_w2v_knn():
    w2v_knn = w2v_model('ball_tree')
    w2v_knn.load_data()
    w2v_knn.run_knn()
    w2v_knn.calculate_intra_average_and_variance()
    w2v_knn.generate_index_centroid_map()
    
    print('saving w2v_knn')
    with open('../../processed_files/w2v_knn.pickle', 'wb') as file:
        pickle.dump(w2v_knn, file)
    
    return w2v_knn


# In[33]:


w2v_knn = None
if refresh == False:
    try:
        print('read w2v_knn object from memory')
        w2v_knn = pickle.load(open("../../processed_files/w2v_knn.pickle", "rb"))
    except:
        print('error reading w2v_knn from memory, running processing steps again')
        w2v_knn = new_w2v_knn()
else:
    # run processing steps again
    print('refresh == true, running w2v_knn processing steps')
    w2v_knn = new_w2v_knn() 


# In[34]:


#w2v_knn.index_centroid[1]
#w2v_knn.sample_mean_intercluster_dist(10)


# In[35]:


w2v_knn_inter_dist_sample = [w2v_knn.sample_mean_intercluster_dist(500) for i in range(5)]
w2v_knn_inter_dist_sample


# In[36]:


#w2v_knn.index_centroid7500


# In[37]:


w2v_knn_inter_dist_mean = np.average(w2v_knn_inter_dist_sample)
w2v_knn_inter_dist_mean


# ## glove matrix

# In[38]:


def new_gv_knn():
    gv_knn = gv_model('ball_tree')
    gv_knn.load_data()
    gv_knn.run_knn()
    gv_knn.calculate_intra_average_and_variance()
    gv_knn.generate_index_centroid_map()
    
    print('saving gv_knn')
    with open('../../processed_files/gv_knn.pickle', 'wb') as file:
        pickle.dump(gv_knn, file)
    
    return gv_knn


# In[39]:


gv_knn = None
if refresh == False:
    try:
        print('read gv_knn object from memory')
        gv_knn = pickle.load(open("../../processed_files/gv_knn.pickle", "rb"))
    except:
        print('error reading gv_knn from memory, running processing steps again')
        gv_knn = new_gv_knn()
else:
    # run processing steps again
    print('refresh == true, running gv_knn processing steps')
    gv_knn = new_gv_knn()


# In[40]:


gv_knn_inter_dist_sample = [gv_knn.sample_mean_intercluster_dist(500) for i in range(5)]
gv_knn_inter_dist_sample


# In[41]:


gv_knn_inter_dist_mean = np.average(gv_knn_inter_dist_sample)
gv_knn_inter_dist_mean


# ## doc2vec matrix

# In[42]:


def new_d2v_knn():
    d2v_knn = d2v_model('ball_tree')
    d2v_knn.load_data()
    d2v_knn.run_knn()
    d2v_knn.calculate_intra_average_and_variance()
    d2v_knn.generate_index_centroid_map()
    
    print('saving d2v_knn')
    with open('../../processed_files/d2v_knn.pickle', 'wb') as file:
        pickle.dump(d2v_knn, file)
    
    return d2v_knn


# In[43]:


d2v_knn = None
if refresh == False:
    try:
        print('read d2v_knn object from memory')
        d2v_knn = pickle.load(open("../../processed_files/d2v_knn.pickle", "rb"))
    except:
        print('error reading d2v_knn from memory, running processing steps again')
        d2v_knn = new_d2v_knn()
else:
    # run processing steps again
    print('refresh == true, running d2v_knn processing steps')
    d2v_knn = new_d2v_knn()


# In[44]:


d2v_knn_inter_dist_sample = [d2v_knn.sample_mean_intercluster_dist(500) for i in range(5)]
d2v_knn_inter_dist_sample


# In[45]:


d2v_knn_inter_dist_mean = np.average(d2v_knn_inter_dist_sample)
d2v_knn_inter_dist_mean


# ## Checking normlizing

# In[46]:


num_rows_tfidf_knn = len(tfidf_knn.matrix)
num_rows_tfidf_knn


# In[47]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_tfidf_knn):
    if np.sum(tfidf_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('tfidf_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# In[48]:


num_rows_lda_10_knn = len(lda_10_knn.matrix)
num_rows_lda_10_knn


# In[49]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_lda_10_knn):
    if np.sum(lda_10_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('lda_10_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# In[50]:


num_rows_lda_25_knn = len(lda_25_knn.matrix)
num_rows_lda_25_knn


# In[51]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_lda_25_knn):
    if np.sum(lda_25_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('lda_25_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# In[52]:


num_rows_lda_50_knn = len(lda_50_knn.matrix)
num_rows_lda_50_knn


# In[53]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_lda_50_knn):
    if np.sum(lda_50_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('lda_50_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# In[54]:


num_rows_w2v_knn = len(w2v_knn.matrix)
num_rows_w2v_knn


# In[55]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_w2v_knn):
    if np.sum(w2v_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('w2v_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# In[56]:


num_rows_glove_knn = len(gv_knn.matrix)
num_rows_glove_knn


# In[57]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_glove_knn):
    if np.sum(gv_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('glove_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# In[58]:


num_rows_d2v_knn = len(d2v_knn.matrix)
num_rows_d2v_knn


# In[59]:


more_than_one = 0
less_than_one = 0
for i in range(num_rows_d2v_knn):
    if np.sum(d2v_knn.matrix[i]) > 1:
        more_than_one += 1
    else:
        less_than_one += 1
print('d2v_knn --> more_than_one: ', more_than_one, ', less_than_one: ', less_than_one)


# ## metric analysis

# In[61]:


print(bow_stemmed_knn.intra_avg_list[0])
print(tfidf_knn.intra_avg_list[0])
print(lda_10_knn.intra_avg_list[0])
print(lda_25_knn.intra_avg_list[0])
print(lda_50_knn.intra_avg_list[0])
print(w2v_knn.intra_avg_list[0])
print(gv_knn.intra_avg_list[0])
print(d2v_knn.intra_avg_list[0])


# ### average intracluster distances

# In[62]:


import matplotlib.pyplot as plt

data = [bow_stemmed_knn.intra_avg_list, tfidf_knn.intra_avg_list, lda_10_knn.intra_avg_list, lda_25_knn.intra_avg_list, lda_50_knn.intra_avg_list, w2v_knn.intra_avg_list, gv_knn.intra_avg_list, d2v_knn.intra_avg_list]  
#data = [lda_knn.intra_avg_list, w2v_knn.intra_avg_list, gv_knn.intra_avg_list, d2v_knn.intra_avg_list]  
fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(['bow_stemmed', 'tfidf', 'lda_10', 'lda_25', 'lda_50', 'word2vec', 'glove', 'doc2vec'])
plt.show()


# In[ ]:


plt.savefig('intracluster.png')


# ### average intercluster distances (samples)

# In[63]:


objects = ('bow_stemmed', 'tfidf', 'lda_10', 'lda_25', 'lda_50', 'w2v', 'glove', 'd2v')
#objects = ('lda', 'w2v', 'glove', 'd2v')
y_pos = np.arange(len(objects))
performance = [bow_stemmed_knn_inter_dist_mean, tfidf_knn_inter_dist_mean, lda_10_knn_inter_dist_mean, lda_25_knn_inter_dist_mean, lda_50_knn_inter_dist_mean, w2v_knn_inter_dist_mean, gv_knn_inter_dist_mean, d2v_knn_inter_dist_mean]
#performance = [lda_knn_inter_dist_mean, w2v_knn_inter_dist_mean, gv_knn_inter_dist_mean, d2v_knn_inter_dist_mean]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('sampled average (500 cluster, 5 iterations)')
plt.title('Intercluster Distances')

plt.show()


# In[ ]:


plt.savefig('intercluster.png')


# ### silhouetter coefficient

# In[64]:


intra_avg_mean_list = [np.mean(avg_list) for avg_list in [bow_stemmed_knn.intra_avg_list, tfidf_knn.intra_avg_list, lda_10_knn.intra_avg_list, lda_25_knn.intra_avg_list, lda_50_knn.intra_avg_list, w2v_knn.intra_avg_list, gv_knn.intra_avg_list, d2v_knn.intra_avg_list]]
inter_dist_mean_list = [bow_stemmed_knn_inter_dist_mean, tfidf_knn_inter_dist_mean, lda_10_knn_inter_dist_mean, lda_25_knn_inter_dist_mean, lda_50_knn_inter_dist_mean, w2v_knn_inter_dist_mean, gv_knn_inter_dist_mean, d2v_knn_inter_dist_mean]


# In[65]:


intra_inter_zipped = zip(intra_avg_mean_list, inter_dist_mean_list)
silhouette_scores = []
for a, b in intra_inter_zipped:
    score = (b - a)/np.max([a, b])
    silhouette_scores.append(score)


# In[67]:


#objects = ('tfidf', 'lda', 'w2v', 'glove', 'd2v')
objects = ('bow_stemmed', 'tfidf', 'lda_10', 'lda_25', 'lda_50', 'w2v', 'glove', 'd2v')
y_pos = np.arange(len(objects))
performance = [bow_stemmed_knn_inter_dist_mean, tfidf_knn_inter_dist_mean, lda_10_knn_inter_dist_mean, lda_25_knn_inter_dist_mean, lda_50_knn_inter_dist_mean, w2v_knn_inter_dist_mean, gv_knn_inter_dist_mean, d2v_knn_inter_dist_mean]
#performance = [lda_knn_inter_dist_mean, w2v_knn_inter_dist_mean, gv_knn_inter_dist_mean, d2v_knn_inter_dist_mean]

plt.bar(y_pos, silhouette_scores, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('sampled average (500 cluster, 5 iterations)')
plt.title('Silhouettes Coefficients')

plt.show()


# In[ ]:


plt.savefig('silhouette.png')


# In[ ]:




