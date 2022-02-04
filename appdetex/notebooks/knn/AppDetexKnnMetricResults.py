import pickle

class KnnMetricResults:
    # argument list:
    # - dataset: AppDetex
    # - strategy: Knn
    # - stemmed: true or false
    # - representation: BOW, TFIDF, LDA_10, LDA_25, LDA_50, Word2Vec, Glove, Doc2Vec
    # - technique: ball_tree, kd_tree, brute
    # - num_neighbors: 5, 10, 20, 40
    def __init__(self, stemmed, representation, technique, num_neighbors):
        self.dataset = 'AppDetex'
        self.strategy = 'Knn'
        self.stemmed = stemmed
        self.representation = representation
        self.technique = technique
        self.num_neighbors = num_neighbors
        self.filename = '../../../results/'+ self.dataset + '_' + self.strategy + '_' + stemmed + '_' + representation + '_' + technique + '_' + str(num_neighbors) + '.metrics'
    
    #def set_intra_metrics(self, average, variance):
    #    self.intra_average = average
    #    self.intra_variance = variance
    
    def set_intra_metrics(self, intra_average, intra_average_map, intra_variance, intra_variancee_map):
        self.intra_average = intra_average
        self.intra_average_map = intra_average_map
        self.intra_variance = intra_variance
        self.intra_variance_map = intra_variancee_map
        
    def set_inter_metrics(self, sample, mean):
        self.inter_distance_sample = sample
        self.inter_distance_mean = mean
        
    def set_silhouette(self, silhouette):
        self.silhouette = silhouette
        
    def save(self):
        file = open(self.filename, 'wb') 
        pickle.dump(self, file)