from procs import processing
from util import CountVec, aggregate
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModel
from scipy import sparse
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

class Model:
    
    count_vec = None
    item_data = None
    processed_tokens = None
    tokenization = None
    pooling= None
    
    def __init__(self, pooling='mean'):
        self.pooling = pooling
       
    #def load_precounted(self, count_vec):
    #    self.count_vec = count_vec
    
    def load_preprocessed(self, processed_tokens):
        self.processed_tokens = processed_tokens
    
    #def get_countvec(self):
    #    return self.count_vec
    
    #def get_processed_tokens(self):
    #    return self.processed_tokens
    
    def preprocess(self, df, col, tokenize = True, lower = True, stop_words_remove = True, stemmed = True):
        proc = processing(lower, stop_words_remove, stemmed)
        self.processed_tokens = proc.generate_preprocessed_corpus(df,col)
    

class TFIDF(Model): 
  
    def fit(self):
        self.count_vec = CountVec().countvec(self.processed_tokens)
        tfidf_transformer = TfidfTransformer()
        tf_idf_mat = tfidf_transformer.fit_transform(self.count_vec)
        return tf_idf_mat

class LDA(Model):

    n_components = None
    random_state = None
    lda = None
    
    def __init__(self, tokenization='BOW', pooling='mean', n_components=20, random_state=0):
        self.tokenization = tokenization
        self.pooling = pooling
        self.n_components = n_components
        self.random_state = random_state
        self.lda = LatentDirichletAllocation(self.n_components, self.random_state)
    
    def fit(self):
        if self.count_vec is None:
            cv = CountVec()
            self.count_vec = cv.countvec(self.processed_tokens)
        lda_mat = self.lda.fit_transform(self.count_vec)
        #return sparse.csr_matrix(lda_mat)
        return lda_mat

class W2V(Model):
    
    embedding_model = None
    item_embeddings = None
    
    def load_pretrained(self, path='../../../utilities/pretrained/GoogleNews-vectors-negative300.bin'):
        W2Vmodel = KeyedVectors.load_word2vec_format(path, binary=True)
        self.embedding_model = W2Vmodel
        
    def extract_emb(self, content):
        embedding_matrix = []
        for token in content:
            try:
                emb_vec = self.embedding_model[token]
                embedding_matrix.append(emb_vec)
            except:
                continue
        return embedding_matrix

    def fit(self):
        ag = aggregate(self.pooling)
        encoded_input = self.processed_tokens
        model_output = encoded_input.apply(lambda x: self.extract_emb(x))
            
        self.item_embeddings = model_output.apply(lambda x: ag.pool(x))
        return self.item_embeddings
    
class Glove(Model):
    embedding_dict = None
    item_embeddings = None
        
    def load_pretrained(self, path='../../../utilities/pretrained/glove.840B.300d.txt'):
        embedding_dict={}
        with open(path,'r') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                try:
                    vectors = np.asarray(values[1:],'float32')
                    embedding_dict[word] = vectors
                except:
                    print(values)
                    print()
        f.close()
        self.embedding_dict=embedding_dict
        
    def extract_emb(self, content):
        embedding_matrix=[]
        for token in content:
            emb_vec = self.embedding_dict.get(token)
            if emb_vec is not None:
                embedding_matrix.append(emb_vec)
        return embedding_matrix

    def fit(self):
        ag = aggregate(self.pooling)
        encoded_input = self.processed_tokens
        model_output = encoded_input.apply(lambda x: self.extract_emb(x))
        self.item_embeddings = model_output.apply(lambda x: ag.pool(x))
        return self.item_embeddings
            
class DOC2VEC(Model):
    
    embedding_model = None
    item_embeddings = None
        
    def train(self):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        corpus = self.processed_tokens
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(corpus)]
        doc2vec = Doc2Vec(tagged_data, vector_size = 300, window = 2, min_count = 3, epochs = 10)
        self.embedding_model = doc2vec
        
    def extract_emb(self, content):
        try:
            return self.embedding_model.infer_vector(content)
        except:
            pass

    def fit(self):
        encoded_input = self.processed_tokens
        model_output = encoded_input.apply(lambda x: self.extract_emb(x))
        self.item_embeddings = model_output
        return self.item_embeddings
