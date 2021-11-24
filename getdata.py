#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go


# In[8]:


def get_data(tickers, start, end, interval, excel = True, output = "output.xslx"):
    
    data = yf.download(tickers = tickers, start = start, end = end, interval = interval)
    data = data.reset_index()
    data['Datetime'] = data['Datetime'].apply(lambda a: pd.to_datetime(a).date())
    if excel == True:
        data.to_excel(output)
    return data


# In[16]:





# In[ ]:




