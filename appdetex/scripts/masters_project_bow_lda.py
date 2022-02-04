#!/usr/bin/env python
# coding: utf-8

# ## Libraries and Filesystem Setup

# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from IPython import get_ipython


# ### (already have imported / processed dataframe and stopwords)

# ## Parsing and Processing Content

# In[4]:


df_en = pd.read_csv('../processing_files/only_en.csv')
df_en.head()


# ### load stopword lists from MastersProjectStopwords.ipynb

# In[5]:


common_stopwords = pickle.load(open("../processing_files/common_stopwords.pickle", "rb"))
corpus_stopwords = pickle.load(open("../processing_files/corpus_stopwords.pickle", "rb"))
combined_stopwords = pickle.load(open("../processing_files/combined_stopwords.pickle", "rb"))


# ## Merging duplicate domain

# In[6]:


df_merged = pd.DataFrame(df_en.groupby('DOMAIN')['RAW_CONTENT'].agg('sum')).reset_index()
df_merged.head()


# In[7]:


number_domain = df_merged['DOMAIN'].nunique()
number_domain


# In[8]:


df_merged.shape


# In[24]:


with open('../processing_files/df_merged.pickle', 'wb') as file:
    pickle.dump(df_merged, file)


# ## TEXT Processing

# In[10]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from nltk.stem import PorterStemmer
ps = PorterStemmer()

from IPython.display import display

def process(content):
    processed = tokenizer.tokenize(content)
    processed = [token.lower() for token in processed]
    processed = [token for token in processed if token not in stop_words]
    return [ps.stem(token) for token in processed]


# In[11]:


#df_en['RAW_CONTENT_PARSED'] = df_en['RAW_CONTENT'].apply(lambda row: process(row))
#df_en.to_parquet('processed.parquet')


# ## count vectorizer

# In[12]:


vect = CountVectorizer(tokenizer=process)
corpus = df_merged['RAW_CONTENT'].tolist()
BOW = vect.fit_transform(corpus)


# In[13]:


with open('../processing_files/bow.pickle', 'wb') as file:
    pickle.dump(BOW, file)


# ## exploring LDA

# In[14]:


from sklearn.decomposition import LatentDirichletAllocation


# In[15]:


lda = LatentDirichletAllocation(n_components=20, random_state=0)
lda


# In[16]:


#get_ipython().run_cell_magic('time', '', 'LDA_mat = lda.fit_transform(BOW)')
LDA_mat = lda.fit_transform(BOW)


# In[17]:


LDA_mat.shape


# In[18]:


LDA_mat[0]


# In[19]:


lda_10 = LatentDirichletAllocation(n_components=10, random_state=0)


# In[20]:


#get_ipython().run_cell_magic('time', '', 'LDA_mat_10 = lda_10.fit_transform(BOW)')
LDA_mat_10 = lda_10.fit_transform(BOW)


# In[21]:


LDA_mat_10.shape


# In[22]:


LDA_mat_10[0]


# In[23]:


with open('../processing_files/LDA_mat.pickle', 'wb') as file:
    pickle.dump(LDA_mat_10, file)


# In[ ]:




