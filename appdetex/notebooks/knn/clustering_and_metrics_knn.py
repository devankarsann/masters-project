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
    df_merged = None
    matrix = None
    intra_avg_list = None
    intra_variance_list = None
    index_centroid = None
    distances = None
    indices = None
    algorithm = None
    num_neighbors = None
    similarities_sparse = None  
            
    def __init__(self, stemmed, representation, algorithm, num_neighbors):
        self.stemmed = stemmed
        self.representation = representation
        self.matrix_name = representation + '/'
        if stemmed == 'true':
            self.matrix_name += 'stemmed_' 
        self.matrix_name += representation + '_matrix'
        if algorithm not in ['ball_tree', 'kd_tree', 'brute', 'auto']:
            raise (algorithm + ' is not a valid option for algorithm argument')
        self.algorithm = algorithm
        self.num_neighbors = num_neighbors  
            
    def load_data(self):
        
        # file extension
        
        file_ext = ''
        if self.matrix_name in ['tfidf/tfidf_matrix', 'tfidf/stemmed_tfidf_matrix', 'bow/bow_matrix', 'bow/stemmed_bow_matrix']:
            file_ext = '.npz'
        else:
            file_ext = '.npy'
        print('using file extension ' + file_ext + ' for ' + self.matrix_name)
        
        # loading file
        
        matrix_file = "../../processed_files/" + self.matrix_name + file_ext
        print('loading ' + matrix_file)
        matrix = None
        if self.matrix_name in ['tfidf/tfidf_matrix', 'tfidf/stemmed_tfidf_matrix']:
            matrix = scipy.sparse.load_npz(matrix_file)
            sum_row = matrix.sum(axis=1)
            norm_matrix = matrix / sum_row.reshape(len(sum_row),1)
            self.matrix = norm_matrix
        elif self.matrix_name in ['bow/bow_matrix', 'bow/stemmed_bow_matrix']:
            self.matrix = scipy.sparse.load_npz(matrix_file)
        else:
            matrix = np.load(matrix_file, allow_pickle = True)
            sum_row = matrix.sum(axis=1)+1.0e-6
            norm_matrix = matrix / sum_row.reshape(len(sum_row),1)
            self.matrix = norm_matrix
        #else:
        #    self.matrix = np.load(matrix_file, allow_pickle = True)
            
        # loading raw data
        raw_data_file = '../../processed_files/df_merged.pickle'
        print('loading ' + raw_data_file)
        self.df_merged = pickle.load(open(raw_data_file, "rb"))

    def run_knn(self):
        print('finding nearest ' + str(self.num_neighbors) + ' neighbors with ' + self.algorithm + ' algorithm')
        # setting an array element with a sequence
        # should I call to list on the bow data before saving it?
        print('type of metrix: ' + str(type(self.matrix)))
        nbrs = NearestNeighbors(self.num_neighbors, self.algorithm).fit(self.matrix)
        self.distances, self.indices = nbrs.kneighbors(self.matrix)

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

    def calculate_centroid(self, index):
        rows = self.matrix[index,:]
        row_mean = np.array(rows.mean(axis=0))
        if isinstance(row_mean[0], np.ndarray):
            return row_mean[0]
        else:
            return row_mean

    def generate_index_centroid_map(self):
        dist_mat = []
        for k in self.indices: 
            temp = self.calculate_centroid(k)
            dist_mat.append(temp)
        self.index_centroid = np.array(dist_mat)

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
        
class bow_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'bow', algorithm, num_neighbors)
        
class tfidf_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'tfidf', algorithm, num_neighbors)
    
class lda_10_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'lda_10', algorithm, num_neighbors)
        
class lda_25_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'lda_25', algorithm, num_neighbors)

class lda_50_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'lda_50', algorithm, num_neighbors)
        
class w2v_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'w2v', algorithm, num_neighbors)
        
class gv_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'glove', algorithm, num_neighbors)
        
class d2v_model(knn_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_neighbors):
        super().__init__(stemmed, 'd2v', algorithm, num_neighbors)
