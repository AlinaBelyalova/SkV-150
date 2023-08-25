#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import sys


# In[58]:


path = r'C:\Users\Alina\Desktop\filess'

# read an excel file and convert
# into a dataframe object
all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

list_of_dfs = [pd.read_csv(file, sep=';', encoding='cp1251') for file in all_files]

df = pd.concat(list_of_dfs, ignore_index=True)
print(sys.getsizeof(df))
sys.getsizeof(list_of_dfs)


# In[19]:


all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
[(pd.read_csv(file, sep = ';'), print(file)) for file in all_files]


# In[53]:


#[(df['Rhob_0-256'] , df['Depth'], df ['Vp'], df['Vs'], df['E'], df['v'])]
df.describe()


# In[59]:


Vp = df['Rhob_0-256']**4 / 0.31**4
Vs = Vp/np.sqrt(2)
E = df['Rhob_0-256'] * Vs**2 * (3*Vp**2 - 4*Vs**2)/(Vp**2 - Vs**2) #модуль Юнга
G = df['Rhob_0-256'] * Vs**2 #модуль сжатия
K = df['Rhob_0-256'] * (Vp**2 - 4/3*(Vs**2)) #коэффициент всестороннего сжатия
v = (Vp**2 - 2*Vs**2) / 2 * (Vp**2 - Vs**2) #коэффициент Пуассона


# In[60]:


df['Vp'] = True
df['Vs'] = True
df['E'] = True
df['G'] = True
df['K'] = True
df['v'] = True

df['Vp'] = Vp
df['Vs'] = Vs
df['E'] = E
df['G'] = G
df['K'] = K
df['v'] = v


# In[61]:


fig, axs = plt.subplots(1, 3, figsize = (8, 7), layout = 'constrained')

axs[0].plot(df['Rhob_0-256'],df['Depth'])  # Plot some data on the (implicit) axes.
axs[0].set_title('Плотность-глубина')
axs[0].set_xlabel('ρ, г/см³')
axs[0].set_ylabel('Глубина, м')

axs[1].plot(df['Vp'],df['Depth'], label = 'Vp')
axs[1].plot(df['Vs'],df['Depth'], label = 'Vs')
axs[1].set_title('Скорость-глубина')
axs[1].set_xlabel('V, м/c')
axs[1].set_ylabel('Глубина, м')
axs[1].legend()

axs[2].plot(df['E'],df['Depth'], label = 'E')
axs[2].plot(df['G'],df['Depth'], label = 'G')
axs[2].plot(df['K'],df['Depth'], label = 'K')
axs[2].plot(df['v'],df['Depth'], label = 'v')
axs[2].set_title('Модули-глубина')
axs[2].set_xlabel('Модули')
axs[2].set_ylabel('Глубина, м')
axs[2].legend()


# In[ ]:





# In[34]:


# Save content
df.to_csv('saved_skv150.csv',
          index=False)

