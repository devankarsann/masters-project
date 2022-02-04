#!/usr/bin/env python
# coding: utf-8

# Libraries and Filesystem Setup

import pandas as pd
import pickle
#from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
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

class kmeans_clustering_and_metrics:
    matrix_name = None
    df_merged = None
    matrix = None
    intra_avg_list = None
    intra_variance_list = None
    index_centroid = None
    labels = None
    cluster_centers = None
    algorithm = None
    num_clusters = None
    similarities_sparse = None
    
    # algorithm parameter choices include: ball_tree, kd_tree, brute, and auto
    def __init__(self, matrix_name, algorithm, num_clusters):
        self.matrix_name = matrix_name
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        if algorithm not in ['auto', 'full', 'elkan']:
            raise (algorithm + ' is not a valid option for algorithm')
        self.algorithm = algorithm
        self.num_clusters = num_clusters
            
    def load_data(self):
        
        # file extension
        
        file_ext = ''
        if self.matrix_name == 'tfidf_matrix':
            file_ext = '.npz'
        else:
            file_ext = '.npy'
        print('using file extension ' + file_ext + ' for ' + self.matrix_name)
        
        # loading file
        
        matrix_file = "processed_files/" + self.matrix_name + file_ext
        print('loading ' + matrix_file)
        matrix = None
        if self.matrix_name == 'tfidf_matrix':
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
        raw_data_file = 'processed_files/df_merged.pickle'
        print('loading ' + raw_data_file)
        self.df_merged = pickle.load(open(raw_data_file, "rb"))

    def run_kmeans(self):
        print('clustering documents into ' + str(self.num_clusters) + ' clusters with ' + self.algorithm + ' algorithm')
        nbrs = NearestNeighbors(self.num_neighbors, self.algorithm).fit(self.matrix)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(lda_matrix)        
        self.labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

    def generate_raw_content_cluster_df(self, index):
        index_list = list(self, self.indices[index])
        distance_list = list(self.distances[index])
        cluster_seed = df_merged.loc[index].to_frame().T
        cluster_seed['DISTANCE'] = 0
        cluster_df = df_merged.loc[index_list[1:]]
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
                    cos_sim = 1 - sklearn.metrics.pairwise.cosine_similarity([self.index_centroid[i]], [self.index_centroid[j]])
                    centroid_centroid_distance[key] = cos_sim

        val = np.array(list(centroid_centroid_distance.values())).mean()
        if math.isnan(val):
            print('detected nan val')
            print(np.array(list(centroid_centroid_distance.values())))
        return val
        

class tfidf_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('tfidf_matrix', algorithm)
    
class lda_10_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('lda_10_matrix', algorithm)
        
class lda_25_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('lda_25_matrix', algorithm)

class lda_50_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('lda_50_matrix', algorithm)
        
class w2v_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('w2v_matrix', algorithm)
        
class gv_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('gv_matrix', algorithm)
        
class d2v_model(knn_clustering_and_metrics):
    def __init__(self, algorithm):
        super().__init__('d2v_matrix', algorithm)
    