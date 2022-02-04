#!/usr/bin/env python
# coding: utf-8

# ## Libraries and Filesystem Setup

# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize 
import pandas as pd
from collections import Counter
import string
import pickle
from IPython import get_ipython


# ## Parsing and Processing Content

# In[5]:


df_en_2 = pd.read_csv('../processing_files/only_en.csv')


# In[6]:


df_en_2['RAW_CONTENT_TRIGRAM'] = df_en_2['RAW_CONTENT'].apply(lambda row: [' '.join(phrase) for phrase in nltk.trigrams(row.lower().split())])
df_en_2.head()


# In[7]:


print(string.punctuation)
table = str.maketrans('', '', string.punctuation)


# In[8]:


df_en_2_trigram_tokens = [item for sublist in df_en_2['RAW_CONTENT_TRIGRAM'].values for item in sublist]
df_en_2_trigram_tokens_stripped = [w.translate(table) for w in df_en_2_trigram_tokens]
df_en_2_trigram_tokens_stripped = [phrase.strip() for phrase in df_en_2_trigram_tokens_stripped if len(phrase.strip().split()) == 3]
df_en_2_trigram_tokens_stripped[:10]


# ## StopPhrases

# In[9]:


counts_trigrams = Counter(df_en_2_trigram_tokens_stripped)
counts_trigrams.most_common(30)


# ## Heaps' Law Trigrams

# In[10]:


df_en_2.head()


# In[11]:


df_en_2_new = df_en_2.reset_index()
df_en_2_new = df_en_2_new.drop(columns=['index'])
df_en_2_new.head()


# In[12]:


# number of documents processed, number of unique 
num_docs_num_unique = [(0,0)]
unique = set()
for index,row in df_en_2_new.iterrows():
    unique.update(row['RAW_CONTENT_TRIGRAM'])
    num_docs = index + 1
    num_unique = len(unique)
    num_docs_num_unique.append((num_docs, num_unique))


# In[13]:


#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#x, y = zip(*num_docs_num_unique)
#plt.scatter(x, y)
#plt.show()


# In[14]:


most_to_least = sorted(counts_trigrams.items(), key=lambda item: item[1], reverse=True)
most_to_least_trigrams = []
for index, item in enumerate(most_to_least):
    most_to_least_trigrams.append((index, item[1]))


# In[15]:


#x, y = zip(*most_to_least_trigrams)
#plt.scatter(x, y)
#plt.show()


# ## Zipf's Law Trigrams

# In[16]:


most_to_least_2 = sorted(counts_trigrams.items(), key=lambda item: item[1], reverse=True)
most_to_least_trigrams_2 = []
for index, item in enumerate(most_to_least):
    if index < 0:
        continue
    if index >= 400:
        break
    most_to_least_trigrams_2.append((index, item[1]))


# In[17]:


#x2, y2 = zip(*most_to_least_trigrams_2)
#plt.scatter(x2, y2)
#plt.show()


# In[18]:


most_to_least_trigrams_2[:30]


# ## Stop Phrase lists

# In[19]:


corpus_trigrams = set([i[0] for i in counts_trigrams.most_common(30)])


# In[21]:


with open('../processing_files/corpus_trigrams.pickle', 'wb') as file:
    pickle.dump(corpus_trigrams, file)


# In[ ]:




