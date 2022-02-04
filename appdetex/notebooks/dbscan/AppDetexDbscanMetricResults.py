import pickle

class DbscanMetricResults:
    # argument list:
    # - dataset: AppDetex
    # - strategy: Dbscan
    # - stemmed: true or false
    # - representation: BOW, TFIDF, LDA_10, LDA_25, LDA_50, Word2Vec, Glove, Doc2Vec
    # - eps: 0.1, 0.25, 0.5
    # - min_sample: 5, 10, 20
    def __init__(self, stemmed, representation, eps, min_sample):
        self.dataset = 'AppDetex'
        self.strategy = 'Dbscan'
        self.stemmed = stemmed
        self.representation = representation
        self.eps = eps
        self.min_sample = min_sample
        self.filename = '../../../results/' + self.dataset + '_' + self.strategy + '_' + self.stemmed + '_' + representation + '_' + str(eps).replace('.', '') + '_' + str(min_sample) + '.metrics'
    
    def set_compute_time(self, compute_time_cluster, compute_time_metrics):
        self.compute_time_cluster = compute_time_cluster
        self.compute_time_metrics = compute_time_metrics
    
    def set_intra_metrics(self, average, variance):
        self.intra_average = average
        self.intra_variance = variance
        
    def set_inter_metrics(self, sample, mean):
        self.inter_distance_sample = sample
        self.inter_distance_mean = mean
        
    def set_silhouette(self, silhouette):
        self.silhouette = silhouette
        
    def save(self):
        file = open(self.filename, 'wb') 
        pickle.dump(self, file)