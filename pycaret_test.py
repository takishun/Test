#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load sample dataset
from pycaret.datasets import get_data
data = get_data('diabetes')


# In[2]:


from pycaret.classification import *
s = setup(data, target = 'Class variable', session_id = 123)


# In[ ]:




