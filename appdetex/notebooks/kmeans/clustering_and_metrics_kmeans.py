#!/usr/bin/env python
# coding: utf-8

# Libraries and Filesystem Setup

import pandas as pd
import pickle
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
from sklearn.cluster import KMeans
import statistics

class kmeans_clustering_and_metrics:
    matrix_name = None
    df_merged = None
    matrix = None
    intra_avg_list = None
    intra_variance_list = None
    index_centroid = None
    algorithm = None
    num_clusters = None
    similarities_sparse = None
    kmeans = None
    index_cluster = None
    index_cluster_dict = None
    cluster_index = None
    cluster_list = None
    cluster_centroid = None
    avg_list = []
    variance_list = []
        
    def __init__(self, stemmed, representation, algorithm, num_clusters):
  
        self.stemmed = stemmed
        self.representation = representation
        self.matrix_name = representation + '/'
        if stemmed == 'true':
            self.matrix_name += 'stemmed_' 
        self.matrix_name += representation + '_matrix'
        
        if algorithm not in ['full', 'auto', 'elkan']:
            raise (algorithm + ' is not a valid option for algorithm')
        self.algorithm = algorithm
        self.num_clusters = num_clusters  
            
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

    def run_kmeans(self):
        print('finding ' + str(self.num_clusters) + ' clusters with ' + self.algorithm + ' algorithm')
        # setting an array element with a sequence
        # should I call to list on the bow data before saving it?
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, algorithm=self.algorithm).fit(self.matrix)
        self.index_cluster = zip(range(len(self.kmeans.labels_)), self.kmeans.labels_)
        self.index_cluster_dict = dict(self.index_cluster)
        self.cluster_index = list(zip(self.kmeans.labels_, range(len(self.kmeans.labels_))))
        self.cluster_list = dict()
        for i in range(self.num_clusters):
            self.cluster_list[i] = list(filter(lambda row: row[0] == i, self.cluster_index))
    
    # index is cluster index
    def intracluster_similarity(self, index):
        self.cluster_centroid = self.kmeans.cluster_centers_[index]
        dist_list = []
        cluster = self.cluster_list[index]
        for i in cluster:
            distance = np.linalg.norm(self.cluster_centroid-self.matrix[i[1]])
            dist_list.append(distance)

        #avg = sum_dist/(len(indices[index])-1)
        avg = np.average(dist_list)

        #variance = statistics.variance(dist_list)
        variance = np.var(dist_list)

        return avg, variance

    def calculate_intracluster_similarity(self):
        for i in range(self.num_clusters):
            avg, variance = self.intracluster_similarity(i)
            self.avg_list.append(avg)
            self.variance_list.append(variance)
    
    # index is cluster / cluster id
    def generate_raw_content_cluster_df(index):
        index_list = self.cluster_list[index]
        index_list = [x[1] for x in index_list]
        cluster_seed = self.df_merged.loc[index].to_frame().T
        #cluster_df = df_merged.loc[index_list[1:]]
        cluster_df = self.df_merged.loc[index_list[1:]]
        combined = pd.concat([cluster_seed, cluster_df])
        combined['cluster'] = index
        return combined

    def intercluster_dist(self):
        indices = range(len(self.kmeans.cluster_centers_))
        centroid_centroid_distance = {}
        values = []
        for i in indices:
            for j in indices:
                if i < j:
                    key = str(i) + "::" + str(j)
                    distance = np.linalg.norm(self.kmeans.cluster_centers_[i] - self.kmeans.cluster_centers_[j])
                    centroid_centroid_distance[key] = distance
                    values.append(distance)
        return centroid_centroid_distance, values
    
class bow_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'bow', algorithm, num_clusters)

class tfidf_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'tfidf', algorithm, num_clusters)
    
class lda_10_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'lda_10', algorithm, num_clusters)
        
class lda_25_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'lda_25', algorithm, num_clusters)

class lda_50_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'lda_50', algorithm, num_clusters)
        
class w2v_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'w2v', algorithm, num_clusters)
        
class gv_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'glove', algorithm, num_clusters)
        
class d2v_model(kmeans_clustering_and_metrics):
    def __init__(self, stemmed, algorithm, num_clusters):
        super().__init__(stemmed, 'd2v', algorithm, num_clusters)
    