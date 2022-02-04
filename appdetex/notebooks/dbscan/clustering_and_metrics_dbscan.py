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
from sklearn.cluster import DBSCAN
import statistics

class dbscan_clustering_and_metrics:
    matrix_name = None
    df_merged = None
    matrix = None
    intra_avg_list = None
    intra_variance_list = None
    index_centroid = None
    distances = None
    indices = None
    algorithm = None
    similarities_sparse = None
    dbscan = None
    eps = None
    min_samples = None
    num_clusters = None
    index_cluster = None
    index_cluster_dict = None
    cluster_index = None
    cluster_list = dict()
    avg_list = list()
    variance_list = list()
    intercluster_distances = None
    intercluster_values = None
        
    def __init__(self, stemmed, representation, eps, min_samples):
        self.stemmed = stemmed
        self.representation = representation
        self.matrix_name = representation + '/'
        if stemmed == 'true':
            self.matrix_name += 'stemmed_' 
        self.matrix_name += representation + '_matrix'
        self.eps = eps
        self.min_samples = min_samples
            
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
        if self.matrix_name == 'tfidf_matrix':
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

    def run_dbscan(self):
        print('finding clusters with ' + str(self.eps) + ' eps with ' + str(self.min_samples) + ' min_samples')
        # setting an array element with a sequence
        # should I call to list on the bow data before saving it?
        self.dbscan = DBSCAN(eps = self.eps, min_samples = self.min_samples).fit(self.matrix)
        self.num_clusters = max(self.dbscan.labels_)
        self.index_cluster = zip(range(len(self.dbscan.labels_)), self.dbscan.labels_)
        self.index_cluster_dict = dict(self.index_cluster)
        self.cluster_index = list(zip(self.dbscan.labels_, range(len(self.dbscan.labels_))))        
        for i in range(self.num_clusters):
            self.cluster_list[i] = list(filter(lambda row: row[0] == i, self.cluster_index))
        
    def generate_raw_content_cluster_df(self, index):
        index_list = list(self, self.indices[index])
        distance_list = list(self.distances[index])
        cluster_seed = df_merged.loc[index].to_frame().T
        cluster_seed['DISTANCE'] = 0
        cluster_df = df_merged.loc[index_list[1:]]
        cluster_df['DISTANCE'] = distance_list[1:]
        combined = pd.concat([cluster_seed, cluster_df.sort_values(by='DISTANCE', ascending=True)])
        return combined.style.set_properties(subset=['RAW_CONTENT'], **{'width-min': '100px'})


    # index is cluster index
    def intracluster_similarity(self, index):
        cluster_centroid = np.average(self.matrix[[i[1] for i in self.cluster_list[index]]], axis=0)
        dist_list = []
        cluster = self.cluster_list[index]
        for i in cluster:
            distance = np.linalg.norm(cluster_centroid-self.matrix[i[1]])
            dist_list.append(distance)

        #avg = sum_dist/(len(indices[index])-1)
        avg = np.average(dist_list)

        #variance = statistics.variance(dist_list)
        variance = np.var(dist_list)

        return avg, variance
    
    def calculate_intracluster_similarity(self):
        print('calculating intracluster similarity')
        for i in range(self.num_clusters):
            avg, variance = self.intracluster_similarity(i)
            self.avg_list.append(avg)
            self.variance_list.append(variance)
    
    def calculate_centroid(self, index):
        rows = self.matrix[index,:]
        row_mean = np.array(rows.mean(axis=0))
        if isinstance(row_mean[0], np.ndarray):
            return row_mean[0]
        else:
            return row_mean

    def generate_raw_content_cluster_df(self, index):
        index_list = self.cluster_list[index]
        index_list = [x[1] for x in index_list]
        cluster_seed = self.df_merged.loc[index].to_frame().T
        #cluster_df = df_merged.loc[index_list[1:]]
        cluster_df = self.df_merged.loc[index_list[1:]]
        combined = pd.concat([cluster_seed, cluster_df])
        combined['cluster'] = index
        return combined
    
    def calculate_intercluster_dist(self):
        print('calculating intercluster distances')
        indices = range(self.num_clusters)
        centroid_centroid_distance = {}
        values = []
        for i in indices:
            for j in indices:
                if i < j:
                    key = str(i) + "::" + str(j)
                    cluster_centroid_i = np.average(self.matrix[[x[1] for x in self.cluster_list[i]]], axis=0)
                    cluster_centroid_j = np.average(self.matrix[[x[1] for x in self.cluster_list[j]]], axis=0)
                    distance = np.linalg.norm(cluster_centroid_i - cluster_centroid_j)
                    centroid_centroid_distance[key] = distance
                    values.append(distance)
        self.intercluster_distances = centroid_centroid_distance
        self.intercluster_values = values
        
class bow_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'bow', eps, min_sample)
        
class tfidf_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'tfidf', eps, min_sample)
    
class lda_10_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'lda_10', eps, min_sample)
        
class lda_25_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'lda_25', eps, min_sample)

class lda_50_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'lda_50', eps, min_sample)
        
class w2v_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'w2v', eps, min_sample)
        
class gv_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'gv', eps, min_sample)
        
class d2v_model(dbscan_clustering_and_metrics):
    def __init__(self, stemmed, eps, min_sample):
        super().__init__(stemmed, 'd2v', eps, min_sample)
    