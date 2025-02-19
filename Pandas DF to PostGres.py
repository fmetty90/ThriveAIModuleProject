#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
import psycopg2


# In[ ]:


path_directory = "C://xxxx//xxxx//xxxx//xxxxx//xxxxx.csv"
df = pd.read_csv(path_directory) 


# In[ ]:


# Database connection string
engine = create_engine('postgresql+psycopg2://username:password@hostname:port/dbname')

# Write DataFrame to PostgreSQL
df.to_sql('your_table_name', 
          engine, 
          if_exists='replace', 
          index=False)

