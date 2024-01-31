#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np

import statsmodels.api as sm


# In[12]:


data = sm.datasets.longley.load()
data.exog = sm.add_constant(data.exog)
print(data.exog.head())


# In[13]:


ols_resid = sm.OLS(data.endog, data.exog).fit().resid


# In[14]:


resid_fit = sm.OLS(
    np.asarray(ols_resid)[1:], sm.add_constant(np.asarray(ols_resid)[:-1])
).fit()
print(resid_fit.tvalues[1])
print(resid_fit.pvalues[1])


# In[15]:


rho = resid_fit.params[1]


# In[16]:


from scipy.linalg import toeplitz

toeplitz(range(5))


# In[17]:


order = toeplitz(range(len(ols_resid)))


# In[18]:


sigma = rho ** order
gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
gls_results = gls_model.fit()


# In[19]:


glsar_model = sm.GLSAR(data.endog, data.exog, 1)
glsar_results = glsar_model.iterative_fit(1)
print(glsar_results.summary())


# In[20]:


print(gls_results.params)
print(glsar_results.params)
print(gls_results.bse)
print(glsar_results.bse)


# In[ ]:




