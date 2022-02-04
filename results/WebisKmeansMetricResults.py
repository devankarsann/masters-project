import pickle

class KmeansMetricResults:
    # argument list:
    # - dataset: Webis
    # - strategy: Kmeans
    # - stemmed: true or false
    # - representation: BOW, TFIDF, LDA_10, LDA_25, LDA_50, Word2Vec, Glove, Doc2Vec
    # - technique: full, alkan
    # - num_clusters: 10, 50, 100, 500, 1000
    def __init__(self, stemmed, representation, technique, num_clusters):
        self.dataset = 'Webis'
        self.strategy = 'Kmeans'
        self.stemmed = stemmed
        self.representation = representation
        self.technique = technique
        self.num_clusters = num_clusters
        self.filename = '../../../results/' + self.dataset + '_' + self.strategy + '_' + stemmed + '_' + representation + '_' + technique + '_' + str(num_clusters) + '.metrics'
            
    def set_compute_time(self, compute_time_cluster, compute_time_metrics):
        self.compute_time_cluster = compute_time_cluster
        self.compute_time_metrics = compute_time_metrics
            
    def set_intra_metrics(self, intra_average, intra_average_map, intra_variance, intra_variancee_map):
        self.intra_average = intra_average
        self.intra_average_map = intra_average_map
        self.intra_variance = intra_variance
        self.intra_variance_map = intra_variancee_map
        
    def set_inter_metrics(self, sample, mean):
        self.inter_distance_sample = sample
        self.inter_distance_mean = mean
        
    def set_silhouette(self, silhouette_score, silhouette_map):
        self.silhouette_score = silhouette_score
        self.silhuoette_map = silhouette_map
        
    def set_total_hit_metrics(self, checks_count, checks_map, hits_count, hits_map, hit_percent_value, hit_percent_map):
        self.total_checks_count = checks_count
        self.total_checks_map = checks_map
        
        self.total_hits_count = hits_count
        self.total_hits_map = hits_map
        
        self.total_hit_percent_value = hit_percent_value
        self.total_hit_percent_map = hit_percent_map
            
    def set_paraphrase_hit_metrics(self, checks_count, checks_map, hits_count, hits_map, hit_percent_value, hit_percent_map):
        self.paraphrase_checks_count = checks_count
        self.paraphrase_checks_map = checks_map
        
        self.paraphrase_hits_count = hits_count
        self.paraphrase_hits_map = hits_map
        
        self.paraphrase_hit_percent_value = hit_percent_value
        self.paraphrase_hit_percent_map = hit_percent_map
   
    def set_plagiarize_hit_metrics(self, checks_count, checks_map, hits_count, hits_map, hit_percent_value, hit_percent_map):
        self.plagiarize_checks_count = checks_count
        self.plagiarize_checks_map = checks_map
        
        self.plagiarize_hits_count = hits_count
        self.plagiarize_hits_map = hits_map
        
        self.plagiarize_hit_percent_value = hit_percent_value
        self.plagiarize_hit_percent_map = hit_percent_map
        
    def save(self):
        file = open(self.filename, 'wb') 
        pickle.dump(self, file)