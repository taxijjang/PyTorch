#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch
import torch.optim as optim


# In[27]:


x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])


# In[40]:


W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True) #requries_grad =True 학습 시킬거라고 명시


# In[41]:


optimizer = optim.SGD([W,b], lr = 0.01)

nb_epochs = 1000


# In[42]:


for epoch in range(1, nb_epochs +1):
    #값 계산
    hypothesis = x_train * W + b
    
    #cost값 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    #역전파 단계 전에 optimizer 객체를 사용하여
    #갱신할 Variable 들에 대한 모든 변화도를 0으로 만든다
    #.backword()를 호출할 때마다 변화도가 버퍼(Buffer)에 (덮어쓰지 않고) 누적되기
    # 때문이다.
    optimizer.zero_grad()
    
    #역전파 단계
    cost.backward() 
    
    #optimizer의 step함수를 호출하면 매개변수가 갱신 된다.
    print('before W : ', W , ' b : ' , b)
    optimizer.step()
    print('after W : ', W , ' b : ' , b)
    print(cost)

