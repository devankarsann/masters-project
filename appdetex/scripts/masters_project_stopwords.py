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

# In[4]:


df_en = pd.read_csv('../processing_files/only_en.csv')
df_en.head()


# In[5]:


# do not specify axis as parameter when running lambda on series
df_en['RAW_CONTENT_PARSED'] = df_en['RAW_CONTENT'].apply(lambda row: word_tokenize(row))
df_en.head()


# In[6]:


df_en_content_tokens = [item for sublist in df_en['RAW_CONTENT_PARSED'].values for item in sublist]


# In[7]:


counts = Counter(df_en_content_tokens)


# In[8]:


counts.most_common(10)


# In[9]:


df_en_2 = df_en.copy(deep=True)


# In[10]:


df_en_2['RAW_CONTENT_PARSED'] = df_en_2['RAW_CONTENT'].apply(lambda row: row.split())
df_en_2.head()


# In[11]:


df_en_2['RAW_CONTENT_PARSED'] = df_en_2['RAW_CONTENT_PARSED'].apply(lambda row: [token.lower() for token in row])
df_en_2.head()


# In[12]:


print(string.punctuation)
table = str.maketrans('', '', string.punctuation)


# In[13]:


table = str.maketrans('', '', string.punctuation)
df_en_2_content_tokens = [item for sublist in df_en_2['RAW_CONTENT_PARSED'].values for item in sublist]
#df_en_2_content_tokens


# In[14]:


df_en_2_content_tokens_stripped = [w.translate(table) for w in df_en_2_content_tokens]
df_en_2_content_tokens_stripped[:10]


# ## Stopwords

# In[15]:


counts_2 = Counter(df_en_2_content_tokens_stripped)
#nltk stopword list is 179 words
counts_2.most_common(18)


# In[16]:


from nltk.corpus import stopwords
print(stopwords.words('english'))


# In[17]:


len(stopwords.words('english'))


# In[18]:


#counts_2.most_common(179)


# In[19]:


common_stopwords = set(stopwords.words('english'))


# In[20]:


corpus_stopwords = set([i[0] for i in counts_2.most_common(179)])


# In[21]:


in_both_lists = common_stopwords.intersection(corpus_stopwords)


# In[22]:


len(in_both_lists)


# In[23]:


list(in_both_lists)[:10]


# ## Heaps' Law

# In[24]:


df_en_2.head()


# In[25]:


df_en_2_new = df_en_2.reset_index()
df_en_2_new = df_en_2_new.drop(columns=['index'])
df_en_2_new.head()


# In[26]:


# number of documents processed, number of unique 
num_docs_num_unique = [(0,0)]
unique = set()
for index,row in df_en_2_new.iterrows():
    unique.update(row['RAW_CONTENT_PARSED'])
    num_docs = index + 1
    num_unique = len(unique)
    num_docs_num_unique.append((num_docs, num_unique))


# In[27]:


#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#x, y = zip(*num_docs_num_unique)
#plt.scatter(x, y)
#plt.show()


# In[28]:


most_to_least = sorted(counts_2.items(), key=lambda item: item[1], reverse=True)
most_to_least_plot_items = []
for index, item in enumerate(most_to_least):
    most_to_least_plot_items.append((index, item[1]))


# In[29]:


#x, y = zip(*most_to_least_plot_items)
#plt.scatter(x, y)
#plt.show()


# ## Zipf's Law

# In[30]:


most_to_least_2 = sorted(counts_2.items(), key=lambda item: item[1], reverse=True)
most_to_least_plot_items_2 = []
for index, item in enumerate(most_to_least):
    if index < 20:
        continue
    if index >= 500:
        break
    most_to_least_plot_items_2.append((index, item[1]))


# In[31]:


#x2, y2 = zip(*most_to_least_plot_items_2)
#plt.scatter(x2, y2)
#plt.show()


# In[32]:


#most_to_least_plot_items_2


# ## Stopword lists

# In[58]:


common_stopwords = set(stopwords.words('english'))


# In[33]:


with open('../processing_files/common_stopwords.pickle', 'wb') as file:
    pickle.dump(common_stopwords, file)


# In[34]:


corpus_stopwords = set([i[0] for i in counts_2.most_common(100)])


# In[35]:


with open('../processing_files/corpus_stopwords.pickle', 'wb') as file:
    pickle.dump(corpus_stopwords, file)


# In[36]:


combined_stopwords = common_stopwords.union(corpus_stopwords)


# In[38]:


with open('../processing_files/combined_stopwords.pickle', 'wb') as file:
    pickle.dump(combined_stopwords, file)


# In[ ]:




