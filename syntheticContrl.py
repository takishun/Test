#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth


# In[7]:


# Data Requirements
#Import German Reunification data from paper
#Can be found in /datasets folder in repo
data = pd.read_csv("german_reunification.csv")
data = data.drop(columns="code", axis=1)
data.head()


# In[8]:


#Fit synthetic control
sc = Synth(data, "gdp", "country", "year", 1990, "West Germany", n_optim=100)


# In[ ]:




