#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np


# In[3]:


t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)


# In[4]:


print(t.dim())
print(t.shape)
print(t.size())


# In[5]:


print(t[0], t[1],t [-1])


# In[6]:


print(t[2:5], t[4:-1])


# In[7]:


t = torch.FloatTensor([1,2])
print(t.mean())


# In[8]:


t = torch.FloatTensor([[1,2],[3,4]])


# In[9]:


print(t)


# In[10]:


print(t.max(dim=0))


# In[11]:


print(t.max(dim=1))


# In[12]:


ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape)


# In[13]:


print(ft.squeeze())
print(ft.squeeze().shape)


# In[14]:


print(ft.squeeze(dim=0))


# In[15]:


print(ft.squeeze(dim=1))


# In[16]:


print(ft)
print(ft.shape)


# In[17]:


print(ft.view(1,-1))
print(ft.view(1,-1).shape)


# In[18]:


bt = torch.ByteTensor([True,False,False,True])
print(bt)


# In[19]:


print(bt.float())
print(bt.long())


# In[20]:


x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])


# In[23]:


print(torch.stack([x,y,z]))


# In[24]:


print(torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)]))


# In[25]:


x = torch.FloatTensor([[0,1,2],[2,1,0]])


# In[26]:


print(torch.ones_like(x))


# In[28]:


print(torch.zeros_like(x)) #CPU와 GPU는 동시에 같이 연산이 안됨


# In[29]:


x = torch.FloatTensor([[1,2],[3,4]])


# In[31]:


print(x.mul(2.))
print(x) # mul 은 그냥 값을 곱해서 출력 해주는거고.
print(x.mul_(2.))
print(x) #mul_은 값을 곱한 후 수정을 해주는거임.


# In[ ]:




