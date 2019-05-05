#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image


# In[22]:


boston = datasets.load_boston()


# In[23]:


panda = np.array(Image.open('panda.jpg').resize((224,224)))


# In[24]:


panda_tensor = torch.from_numpy(panda)


# In[25]:


panda_tensor.size()


# In[26]:


plt.imshow(panda)


# In[31]:


plt.imshow(panda_tensor[25:175,60:130,0].numpy())


# In[32]:


sales = torch.eye(3,3)


# In[33]:


print(sales)


# In[35]:


a = torch.rand(2,2)
b = torch.rand(2,2)


# In[36]:


c = a+b
d = torch.add(a,b)


# In[37]:


a.add_(5)


# In[38]:


a*b


# In[39]:


a.mul(b)


# In[40]:


a.mul_(b)


# In[41]:


a = torch.rand(10000,10000)
b = torch.rand(10000,10000)


# In[42]:


a.matmul(b)


# In[43]:


a = a.cuda()
b = b.cuda()


# In[44]:


a.matmul(b)


# In[ ]:




