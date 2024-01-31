#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
schools = np.array(
    [
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "Phillips Exeter",
        "Hotchkiss",
        "Lawrenceville",
        "St. Paul's",
        "Mt. Hermon",
    ]
)
# ArviZ ships with style sheets!
az.style.use("arviz-darkgrid")


# In[8]:


rng = np.random.default_rng()
az.plot_posterior(rng.normal(size=100_000));


# In[2]:


a = pd.read_html('https://baseball-data.com/player/t/')


# In[3]:


df = a[0]


# In[4]:


df


# In[5]:


df['年数'].str.replace('年','').astype(int).plot(kind='bar')


# In[6]:


df['出身地'].value_counts().plot(kind='bar')


# In[ ]:




