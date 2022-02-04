import pickle

class KmeansMetricResults:
    # argument list:
    # - dataset: AppDetex
    # - strategy: Kmeans
    # - stemmed: true or false
    # - representation: BOW, TFIDF, LDA_10, LDA_25, LDA_50, Word2Vec, Glove, Doc2Vec
    # - technique: full, alkan
    # - num_clusters: 10, 50, 100, 500, 1000
    def __init__(self, stemmed, representation, technique, num_clusters):
        self.dataset = 'AppDetex'
        self.strategy = 'Kmeans'
        self.stemmed = stemmed
        self.representation = representation
        self.technique = technique
        self.num_clusters = num_clusters
        self.filename = '../../../results/' + self.dataset + '_' + self.strategy + '_' + stemmed + '_' + representation + '_' + technique + '_' + str(num_clusters) + '.metrics'
        
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