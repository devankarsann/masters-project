import pandas as pd
import numpy as np
import itertools
from scipy import sparse as sps
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import logging
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from string import digits
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
#tokenizer = RegexpTokenizer(r'\w+')
from nltk import WhitespaceTokenizer
import string
tokenizer = WhitespaceTokenizer()
remove_digits = str.maketrans('', '', digits)
remove_punctuation = str.maketrans('', '', string.punctuation)
import pickle

class processing:
    
    corpus = None
    
    def __init__(self, tokenize = True, lower = True, stop_words_remove = True, stemmed = True):
        self.tokenize = tokenize
        self.lower = lower
        self.stop_words_remove = stop_words_remove
        self.stemmed = stemmed
        self.combined_stopwords = pickle.load(open("../../processed_files/combined_stopwords.pickle", "rb"))
        self.corpus_bigrams = pickle.load(open("../../processed_files/corpus_bigrams.pickle", "rb"))
        self.corpus_trigrams = pickle.load(open("../../processed_files/corpus_trigrams.pickle", "rb"))    
    
    def process(self, content, stem):
        
        #--------no digits--------#
        processed = content.translate(remove_digits)

        #--------remove punction--------#
        processed = processed.translate(remove_punctuation)
        
        #--------lower case--------#
        processed = content.lower()

        #--------remove trigrams--------#
        if self.stop_words_remove:
            for trigram in self.corpus_trigrams:
                processed = processed.replace(trigram.lower(), '')

        #--------remove bigrams--------#
        if self.stop_words_remove:
            for bigram in self.corpus_bigrams:
                processed = processed.replace(bigram.lower(), '') 

        #--------tokenize--------#
        processed = tokenizer.tokenize(processed)

        #--------remove stopwords--------#
        if self.stop_words_remove:
            processed = [token for token in processed if token not in self.combined_stopwords]

        #--------stem tokens--------#
        if stem:
            processed = [ps.stem(token) for token in processed]

        return processed
    
    def generate_preprocessed_corpus(self, df, col):
        corpus = df[col].progress_apply(lambda x: self.process(x))
        self.corpus = corpus 
        return self.corpus
        
    def __str__(self):
        return 'proc'