#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
#pip install -r requirements.txt
#pip freeze > requirements.txt


# In[2]:


RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


# In[3]:


with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))


# In[4]:


model.basic_RVs


# In[5]:


model.free_RVs


# In[6]:


model.observed_RVs


# In[7]:


model.compile_logp()({"mu": 0})


# In[8]:


get_ipython().run_line_magic('timeit', 'model.compile_logp()({"mu": 0.1})')
logp = model.compile_logp()
get_ipython().run_line_magic('timeit', 'logp({"mu": 0.1})')


# In[9]:


with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)


# In[10]:


pm.logp(x, 0).eval()


# In[11]:


with pm.Model():
    obs = pm.Normal("x", mu=0, sigma=1, observed=rng.standard_normal(100))


# In[12]:


with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
    y = pm.Gamma("y", alpha=1, beta=1)
    plus_2 = x + 2
    summed = x + y
    squared = x**2
    sined = pm.math.sin(x)


# In[13]:


with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
    plus_2 = pm.Deterministic("x plus 2", x + 2)


# In[14]:


with pm.Model():
    # bad:
    x = [pm.Normal(f"x_{i}", mu=0, sigma=1) for i in range(10)]


# In[15]:


coords = {"cities": ["Santiago", "Mumbai", "Tokyo"]}
with pm.Model(coords=coords) as model:
    # good:
    x = pm.Normal("x", mu=0, sigma=1, dims="cities")


# In[16]:


with model:
    y = x[0] * x[1]  # indexing is supported
    x.dot(x.T)  # linear algebra is supported


# In[17]:


with pm.Model(coords={"idx": np.arange(5)}) as model:
    x = pm.Normal("x", mu=0, sigma=1, dims="idx")

model.initial_point()


# In[18]:


with pm.Model(coords={"idx": np.arange(5)}) as model:
    x = pm.Normal("x", mu=0, sigma=1, dims="idx", initval=rng.standard_normal(5))

model.initial_point()


# In[19]:


with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))

    idata = pm.sample(2000)


# In[20]:


idata.posterior.dims


# In[21]:


with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))

    idata = pm.sample(cores=4, chains=6)


# In[22]:


idata.posterior["mu"].shape


# In[23]:


with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=rng.standard_normal(100))

    idata = pm.sample()


# In[24]:


az.plot_trace(idata);


# In[25]:


az.summary(idata)


# In[26]:


az.plot_forest(idata, r_hat=True);


# In[27]:


az.plot_posterior(idata);


# In[28]:


with pm.Model(coords={"idx": np.arange(100)}) as model:
    x = pm.Normal("x", mu=0, sigma=1, dims="idx")
    idata = pm.sample()

az.plot_energy(idata);


# In[29]:


with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=rng.standard_normal(100))

    approx = pm.fit()


# In[ ]:




