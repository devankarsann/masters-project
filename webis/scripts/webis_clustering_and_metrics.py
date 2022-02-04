#!/usr/bin/env python
# coding: utf-8

# Libraries and Filesystem Setup

import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import statistics
from random import sample
from sklearn.preprocessing import normalize
from scipy import spatial
import math
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

class knn_clustering_and_metrics:
    matrix_name = None
    webis_df_processed = None
    matrix = None
    intra_avg_list = None
    intra_variance_list = None
    index_centroid = None
    distances = None
    indices = None
    algorithm = None
    num_neighbors = None
    similarities_sparse = None
    
    # algorithm parameter choices include: ball_tree, kd_tree, brute, and auto
    def __init__(self, matrix_name, algorithm, num_neighbors = 10):
        self.matrix_name = matrix_name
        if algorithm not in ['ball_tree', 'kd_tree', 'brute', 'auto']:
            raise (algorithm + ' is not a valid option for algorithm')
        self.algorithm = algorithm
        self.num_neighbors = num_neighbors  
            
    def load_data(self):
        
        # file extension
        
        file_ext = ''
        if self.matrix_name == 'webis_tfidf_matrix':
            file_ext = '.npz'
        else:
            file_ext = '.npy'
        print('using file extension ' + file_ext + ' for ' + self.matrix_name)
        
        # loading file
        
        matrix_file = "processed_files/" + self.matrix_name + file_ext
        print('loading ' + matrix_file)
        matrix = None
        if self.matrix_name == 'webis_tfidf_matrix':
            matrix = scipy.sparse.load_npz(matrix_file)
            sum_row = matrix.sum(axis=1)
            norm_matrix = matrix / sum_row.reshape(len(sum_row),1)
            self.matrix = norm_matrix
            
            
        else:
            matrix = np.load(matrix_file, allow_pickle = True)
            sum_row = matrix.sum(axis=1)+1.0e-6
        #sum_row = matrix.sum(axis=1).reshape(len(matrix),1)
            norm_matrix = matrix / sum_row.reshape(len(sum_row),1)
            self.matrix = norm_matrix
        
        # normalize if needed
        
        # 'lda_matrix' doesn't need to be sparse
        # self.matrix = matrix / np.linalg.norm(matrix)
#       self.matrix = normalize(matrix, norm='l2', axis=1)
        #if self.matrix_name != 'tfidf_matrix':
        #    self.matrix = normalize(matrix, norm = 'l2', axis = 1)
            # print('normalizing with: matrix / np.linalg.norm(matrix)')
            # dense_matrix = matrix.todense()
            # self.matrix = matrix / np.linalg.norm(matrix)
            # print('normalized ' + self.matrix_name)
            #this.similarities_sparse = cosine_similarity(self.matrix,dense_output=False)
        #else:
            # print('did not normalize ' + self.matrix_name)
            # print('not normalizing for tfidf')
        #    self.matrix = matrix
            # print('normalizing with sklearn')
            # self.matrix = normalize(matrix, norm = 'l2', axis = 1)
        
#       row_sums = matrix.sum(axis = 1)
        #copy_matrix = lda_10_knn.matrix.copy()
#       self.matrix = np.divide(matrix, row_sums.reshape(len(row_sums), 1))
        
        #temp_matrix = normalize(matrix, norm='l2', axis=1)
        #self.matrix = temp_matrix
        #sum_row = temp_matrix.sum(axis=1)
        #sum_row = matrix.sum(axis=1).reshape(len(matrix),1)
        #norm_matrix = temp_matrix / sum_row.reshape(len(sum_row),1)
        #self.matrix = norm_matrix
    
        #sself.similarities_sparse = cosine_similarity(self.matrix,dense_output=False)

        # loading raw data
        raw_data_file = 'processed_files/webis_df_processed.pickle'
        print('loading ' + raw_data_file)
        self.webis_df_processed = pickle.load(open(raw_data_file, "rb"))

    def run_knn(self):
        print('finding nearest ' + str(self.num_neighbors) + ' neighbors with ' + self.algorithm + ' algorithm')
        nbrs = NearestNeighbors(self.num_neighbors, self.algorithm).fit(self.matrix)
        self.distances, self.indices = nbrs.kneighbors(self.matrix)

    #def intracluster_similarity(self, index):
    #    cluster_centroid = self.matrix[index]
    #    dist_list = []
    #    for i in self.indices[index][1:]:
    #        cos_sim = None
    #        dot = None
    #        mult = None
    #        if self.matrix_name == 'tfidf_matrix':
                # cos_sim = np.linalg.norm(cluster_centroid - self.matrix[i].todense())
                # similarity = 1 - cos_sim
                #distance = spatial.distance.cosine(cluster_centroid.todense(), self.matrix[i].todense())
                #dot = np.dot(cluster_centroid.todense(), self.matrix[i].todense())
                #mult = (np.linalg.norm(cluster_centroid.todense())*np.linalg.norm(self.matrix[i].todense()))
                #cos_sim = dot / mult
                
     #           cos_sim = sklearn.metrics.pairwise.cosine_similarity([cluster_centroid], [self.matrix[i]])
     #       else:
                # cos_sim = np.linalg.norm(cluster_centroid - self.matrix[i])
                #cos_sim = spatial.distance.cosine(cluster_centroid, self.matrix[i])
                #dot = np.dot(cluster_centroid, self.matrix[i])
                #mult = (np.linalg.norm(cluster_centroid)*np.linalg.norm(self.matrix[i]))
                #cos_sim = dot / mult
                
     #           cos_sim = sklearn.metrics.pairwise.cosine_similarity([cluster_centroid], [self.matrix[i]])
     #       dist_list.append(cos_sim[0])
     #   avg = np.average(dist_list)
     #   variance = np.var(dist_list)
     #   return avg, variance

    def generate_raw_content_cluster_df(self, index):
        index_list = list(self, self.indices[index])
        distance_list = list(self.distances[index])
        cluster_seed = webis_df_processed.loc[index].to_frame().T
        cluster_seed['DISTANCE'] = 0
        cluster_df = webis_df_processed.loc[index_list[1:]]
        cluster_df['DISTANCE'] = distance_list[1:]
        combined = pd.concat([cluster_seed, cluster_df.sort_values(by='DISTANCE', ascending=True)])
        return combined.style.set_properties(subset=['RAW_CONTENT'], **{'width-min': '100px'})

    def calculate_intra_average_and_variance(self):
        print('calculating metrics for intracluster average and variance')
        self.intra_avg_list = np.mean(self.distances, axis=1)
        self.intra_variance_list = np.var(self.distances, axis=1)
        #self.intra_avg_list = []
        #self.intra_variance_list = []
        #for i in range(self.matrix.shape[0]):
         #   avg, variance = self.intracluster_similarity(i)
         #   self.intra_avg_list.append(avg)
         #   self.intra_variance_list.append(variance)

    # Intercluster Similarity

    def calculate_centroid(self, index):
        rows = self.matrix[index,:]
        row_mean = np.array(rows.mean(axis=0))
        #print(row_mean)
        if isinstance(row_mean[0], np.ndarray):
            return row_mean[0]
        else:
            #print(type(row_mean[0]))
            return row_mean
        
        #cluster_centroid = self.matrix[index]
        #for i in self.indices[index][1:]:
        #    cluster_centroid = np.add(cluster_centroid, self.matrix[i])
        #return cluster_centroid/len(self.indices[index])

    def generate_index_centroid_map(self):
        dist_mat = []
        for k in self.indices: 
            temp = self.calculate_centroid(k)
            #print(temp)
            dist_mat.append(temp)
        self.index_centroid = np.array(dist_mat)
            
        #self.index_centroid = {}
        #for i in range(self.matrix.shape[0]):
         #   self.index_centroid[i] = self.calculate_centroid(i)

    # centroid_centroid_distance

    def sample_mean_intercluster_dist(self, sample_size):
        
        indices_sample = sample(range(self.matrix.shape[0]),sample_size)
        centroid_centroid_distance = {}
        for i in indices_sample:
            for j in indices_sample:
                if i < j:
                    key = str(i) + "::" + str(j)
                    cos_sim = None
                    #dot = None
                    #mult = None
                    #if self.matrix_name == 'tfidf_matrix':
                        #distance = np.linalg.norm(self.index_centroid[i].todense() - self.index_centroid[j].todense())
                        #distance = spatial.distance.cosine(self.index_centroid[i].todense(), self.index_centroid[j].todense())
                        #dot = np.dot(self.index_centroid[i].todense(),self.index_centroid[j].todense())
                        #mult = (np.linalg.norm(self.index_centroid[i].todense())*np.linalg.norm(self.index_centroid[j].todense()))
                        #cos_sim = dot/mult
                        
                        #cos_sim = 1 - sklearn.metrics.pairwise.cosine_similarity([self.index_centroid[i]], [self.index_centroid[j]])
                    #else:
                        #distance = np.linalg.norm(self.index_centroid[i] - self.index_centroid[j])
                        #distance = spatial.distance.cosine(self.index_centroid[i], self.index_centroid[j])    
                        #dot = np.dot(self.index_centroid[i],self.index_centroid[j])
                        #mult = (np.linalg.norm(self.index_centroid[i])*np.linalg.norm(self.index_centroid[j]))
                        #cos_sim = dot / mult
                        
                        #cos_sim = 1 - sklearn.metrics.pairwise.cosine_similarity([self.index_centroid[i]], [self.index_centroid[j]])
                    #print('i: ', i, ' --> ', type(self.index_centroid[i]))
                    #print('j: ', j, ' --> ', type(self.index_centroid[j]))
                    
                    #try:
                    cos_sim = 1 - sklearn.metrics.pairwise.cosine_similarity([self.index_centroid[i]], [self.index_centroid[j]])
                    #except: 
                    #    print('PROBLEM PROBLEM PROBLEM')
                    #    print(type(self.index_centroid[i]), type(self.index_centroid[j]))
                    #    break
                    
                    #if math.isnan(cos_sim[0]):
                    #    print('detected nan cos_sim')
                    #    print(dot)
                    #    print(mult)
                    #    print('i: ', i, ' , j: ', j)
                    centroid_centroid_distance[key] = cos_sim

        val = np.array(list(centroid_centroid_distance.values())).mean()
        if math.isnan(val):
            print('detected nan val')
            print(np.array(list(centroid_centroid_distance.values())))
        return val
        

class tfidf_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_tfidf_matrix', algorithm)
    
class lda_10_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_lda_10_matrix', algorithm)
        
class lda_25_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_lda_25_matrix', algorithm)

class lda_50_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_lda_50_matrix', algorithm)
        
class w2v_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_w2v_matrix', algorithm)
        
class gv_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_gv_matrix', algorithm)
        
class d2v_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('webis_d2v_matrix', algorithm)
    