#!/usr/bin/env python
# coding: utf-8

# ## Libraries and Filesystem Setup

# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from collections import Counter
import string
import re
import pickle
from IPython import get_ipython


# ## Parsing and Processing Content

# In[4]:


df_en_2 = pd.read_csv('../processing_files/only_en.csv')


# In[5]:


df_en_2['RAW_CONTENT_BIGRAM'] = df_en_2['RAW_CONTENT'].apply(lambda row: [' '.join(phrase) for phrase in nltk.bigrams(row.lower().split())])
df_en_2.head()


# In[6]:


print(string.punctuation)
table = str.maketrans('', '', string.punctuation)


# In[7]:


df_en_2_bigram_tokens = [item for sublist in df_en_2['RAW_CONTENT_BIGRAM'].values for item in sublist]
df_en_2_bigram_tokens_stripped = [w.translate(table) for w in df_en_2_bigram_tokens]
df_en_2_bigram_tokens_stripped = [phrase.strip() for phrase in df_en_2_bigram_tokens_stripped if len(phrase.strip().split()) == 2]
df_en_2_bigram_tokens_stripped[:10]


# ## StopPhrases

# In[8]:


counts_bigrams = Counter(df_en_2_bigram_tokens_stripped)
counts_bigrams.most_common(30)


# ## Heaps' Law Bigrams

# In[9]:


df_en_2.head()


# In[10]:


df_en_2_new = df_en_2.reset_index()
df_en_2_new = df_en_2_new.drop(columns=['index'])
df_en_2_new.head()


# In[11]:


# number of documents processed, number of unique 
num_docs_num_unique = [(0,0)]
unique = set()
for index,row in df_en_2_new.iterrows():
    unique.update(row['RAW_CONTENT_BIGRAM'])
    num_docs = index + 1
    num_unique = len(unique)
    num_docs_num_unique.append((num_docs, num_unique))


# In[12]:


#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#x, y = zip(*num_docs_num_unique)
#plt.scatter(x, y)
#plt.show()


# In[13]:


most_to_least = sorted(counts_bigrams.items(), key=lambda item: item[1], reverse=True)
most_to_least_plot_items = []
for index, item in enumerate(most_to_least):
    most_to_least_plot_items.append((index, item[1]))


# In[14]:


#x, y = zip(*most_to_least_plot_items)
#plt.scatter(x, y)
#plt.show()


# ## Zipf's Law Bigrams

# In[15]:


most_to_least_2 = sorted(counts_bigrams.items(), key=lambda item: item[1], reverse=True)
most_to_least_bigrams_2 = []
for index, item in enumerate(most_to_least):
    if index < 0:
        continue
    if index >= 300:
        break
    most_to_least_bigrams_2.append((index, item[1]))


# In[16]:


#x2, y2 = zip(*most_to_least_bigrams_2)
#plt.scatter(x2, y2)
#plt.show()


# ## Stop Phrase lists

# In[17]:


corpus_bigrams = set([i[0] for i in counts_bigrams.most_common(30)])


# In[18]:


with open('../processing_files/corpus_bigrams.pickle', 'wb') as file:
    pickle.dump(corpus_bigrams, file)


# In[ ]:




