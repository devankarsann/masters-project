#!/usr/bin/env python
# coding: utf-8

# ## Libraries and Filesystem Setup

# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from collections import Counter
from IPython import get_ipython
from langdetect import detect


# In[4]:


#from google.colab import drive
#drive.mount('/content/drive/')


# In[5]:


#!ls "/content/drive/My Drive/"


# ## Importing Data

# In[6]:


import pandas as pd
df = pd.read_csv('../input_files/raw_content_100mb.csv') 


# In[7]:


df.head()


# ## Adding Language and Filtering

# In[8]:


def add_lang(row):
    try:
        return detect(row['RAW_CONTENT'])
    except:
        return 'Error'


# In[9]:


df['LANGUAGE'] = df.apply(lambda row: add_lang(row), axis=1)


# In[10]:


df.head()


# In[11]:


df[df['LANGUAGE'] == 'Error']


# In[12]:


df.groupby(['LANGUAGE']).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)[:10]


# In[13]:


df_en = df[df['LANGUAGE'] == 'en']
df_en.head()


# In[15]:


df_en.to_csv('../processing_files/only_en.csv',index=False)


# In[ ]:




