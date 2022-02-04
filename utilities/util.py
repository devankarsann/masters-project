import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 

class CountVec:
    
    vect = None
    def __init__(self):
        pass

    def dummy(self, corpus):
        return corpus

    def countvec(self, corpus):
        vect = CountVectorizer(tokenizer = self.dummy, preprocessor= self.dummy)
        word_count = vect.fit_transform(corpus)
        self.vect = vect
        return word_count

class aggregate:
    
    pooling = None
    
    def __init__(self, pooling = 'mean'):
        self.pooling = pooling
        
    def pool(self, input_mat):
        if self.pooling=='mean':  
            return np.mean(input_mat, axis = 0)
        elif self.pooling=='max':  
            return np.max(input_mat, axis = 0)