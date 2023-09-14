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


path = r"C:\Users\Alina\Desktop\csv"

# read an excel file and convert
# into a dataframe object
all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

list_of_dfs = [pd.read_csv(file, sep=';', encoding='cp1251') for file in all_files]

df = pd.concat(list_of_dfs, ignore_index=True)
print(sys.getsizeof(df))
sys.getsizeof(list_of_dfs)


# In[19]:


all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
[(pd.read_csv(file, sep = ';', encoding='cp1251'), print(file)) for file in all_files]


# In[53]:


#[(df['Rhob_0-256'] , df['Depth'], df ['Vp'], df['Vs'], df['E'], df['v'])]
df.describe()


# In[59]:
    
df_1 = df.dropna(subset=['Rhob_0-256'])
df_2 = df.dropna()
# %%
Rhob_0_256 = df['Rhob_0-256']
Rhob_0_256 = Rhob_0_256.dropna() # убрать nanы

Depth = df_1['Depth']

Vp = Rhob_0_256 **4 / 0.31**4
Vs = Vp/np.sqrt(2)
E = Rhob_0_256  * Vs**2 * (3*Vp**2 - 4*Vs**2)/(Vp**2 - Vs**2) #модуль Юнга
G = Rhob_0_256  * Vs**2 #модуль сжатия
K = Rhob_0_256  * (Vp**2 - 4/3*(Vs**2)) #коэффициент всестороннего сжатия
v = (Vp**2 - 2*Vs**2) / 2 * (Vp**2 - Vs**2) #коэффициент Пуассона

dtimes = np.array(Depth) / np.array(Vp)
times = np.cumsum(dtimes)
# %%

R1 = np.array(Rhob_0_256[1:]*Vp[1:])
R2 = np.array(Rhob_0_256[0:-1]*Vp[0:-1])
R = (R1 - R2) / (R1 + R2)
R_clear = np.copy(R)
R_clear[np.abs(R_clear) < 0.1] = 0.0


# %%
def rikker_func(t, T = 100000, T_mid = 230992/2):
    omega = 2*np.pi/T
    arg2 = (0.5*omega*(t - T_mid))**2.0
    return (1.0 - 2.0 * arg2) * np.exp(-arg2)

# w_rikker = rikker_func(2*np.pi/100000, np.arange(-len(R)//2,len(R)//2))
# w_rikker = rikker_func(times[0:-1])
w_rikker = rikker_func(times)

# %%
plt.figure(11)
plt.plot(R_clear, 'k')
plt.plot(w_rikker, '-r')
plt.show()
# %%
def conv_manual(f, g):
    """
    f and g=rikker_func - are 1d-arrays
    """
    n = len(f)
    h = np.zeros(n)
    
    for j in range(n):
        # g_last_j_inverted = g[-j:][::-1]
        h[j] = np.sum(f[0:j]*g[0:j][::-1])
    
    h = np.array([np.sum(f[0:j]*g[0:j][::-1]) for j in range(n)])
    return h


# %%

conv_full = np.convolve(R_clear, w_rikker, 'full')
# conv_same = np.convolve(R_clear, w_rikker, 'same')
# conv_valid = np.convolve(R_clear, w_rikker, 'valid')

# %%
conv_man = conv_manual(R_clear, w_rikker)

# %%
plt.figure(22)
plt.plot(conv_man, "--r")
# plt.plot(conv_same, "--b")
# plt.plot(conv_valid, "xk")
plt.legend(["manually", "same"])


# %%

conv_R_w_full = np.convolve(R, w_rikker, 'full')

# %%
# plt.figure(23)


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



# %%
plt.figure(1)

plt.plot(R, 'b')
plt.plot(w_rikker, '--r')
plt.show()

# %%

conv_top = np.copy(conv_full)
conv_bot = np.copy(conv_full)
conv_top[conv_full < 0] = 0.0
conv_bot[conv_full > 0] = 0.0

plt.figure(2)
# plt.plot(conv)
plt.plot(conv_top, 'r')
plt.plot(conv_bot, 'b')
plt.plot(conv_R_w_full, "-m")
plt.plot([0, len(R)], [0, 0], '--k')
plt.title("Convolution R_clear with W_rikker")
plt.legend(["R_clear >0", "R_clear < 0", "R"])

# %%
conv_sin = np.convolve(R, np.sin(np.arange(0,len(R))), 'same')

plt.figure(3)
plt.plot(conv_sin);





# In[ ]:





# In[34]:


# Save content
#df.to_csv('saved_skv150.csv',
          #index=False)

