#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:06:35 2022

@author: dliu
"""


import matplotlib.pyplot as plt
# from torchdiffeq import odeint
import numpy as np

# from _impl import odeint
from _impl_origin import odeint

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Memory(nn.Module):    
    def __init__(self):
        super(Memory, self).__init__()

        self.memory = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
            # nn.ReLU()
            nn.Sigmoid()
        )
    def forward(self, t):
        
        # t = torch.flip(t, [0]).reshape([-1,1])
        # t = t.repeat(1,2).reshape(-1,1)
        return self.memory(t)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.memory = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

        for m in self.memory.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
        self.mm = None        
        self.beta = 1.5
        self.gamma = 1
        
    def forward(self, t, y, integro):

        S, I, R = torch.split(y,1,dim=1)
        # print('asfafasfasfsafasfd', I.shape, integro.shape)

        dSdt = -self.beta * S * I + integro# + self.memory(I)#sum(pre*dist)*dx
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I - integro#- self.memory(I)#sum(pre*dist)*dx
        return torch.cat((dSdt,dIdt,dRdt),1)
    
    def integration(self, solution, K, dt):
        # print('sdfsfdsfsf', solution.shape, K.shape)
        
        S, I, R = torch.split(solution, 1, dim=2)
        # https://discuss.pytorch.org/t/one-of-the-variables-required-has-been-modified-by-inplace-operation/104328
        I = I.clone().transpose(1,0)  ######clone ???????
        # print('sfaf: ', I.shape, K.shape)

        integro = I*K
        # print('safdasfasfd', integro.shape)
        integro = torch.sum(integro, dim=1)*dt
        return integro
    
if __name__ == '__main__':

    # data = np.load('../data/train_sir_l.npy')
    # crop = 200
    # data = data[:,:crop,:]
    data = np.load('../data/train_sir.npy')

    k = 5
    t = torch.linspace(0., 80./k, 200//k).to(device)

    # data = data[[0,1], ::k, :]
    data = data[[0,1], :, :]

    func_m = Memory().to(device)
    func = ODEFunc().to(device)
    y = torch.tensor(data, dtype=torch.float32).to(device)
    
    # y0 = y[:,0,:].to(device)
    # pred_y = odeint(func, func_m, y0, t, method='euler').to(device)
    # plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
    # plt.legend()
    
    

    # optimizer = optim.Adam(func.parameters(), lr=1e-3)
    optimizer = optim.Adam([
                    {'params': func.parameters()},
                    {'params': func_m.parameters(), 'lr': 1e-3}
                ], lr=1e-4)
    
    batch_size = 2
    batch = data
    batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    batch_t = t
    for itr in range(1, 1000):
        idx = np.random.choice(np.arange(data.shape[0]),batch_size)
        batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)

        optimizer.zero_grad()
        pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
        pred_y = pred_y.transpose(1,0)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()
        
        if itr%1==0:
            print(loss.item())

    
    
    
    # optimizer = optim.LBFGS([
    #                 {'params': func.parameters()},
    #                 {'params': func_m.parameters()}
    #             ], lr=5e-2)

    # batch_size = 2
    # batch = data
    # batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
    # batch_y0 = batch_y[:,0,:].to(device)
    
    # batch_t = t
    # for itr in range(1, 1000):
    #     def closure():
    #         optimizer.zero_grad()
            
    #         pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
    #         pred_y = pred_y.transpose(1,0)
    #         loss = nn.functional.mse_loss(pred_y, batch_y)
            
    #         # loss = loss_diff + loss_mse# + .01*loss_sum
    #         print(f'loss: {loss.item():.3e}')
    #         loss.backward()
    #         return loss
    #     optimizer.step(closure)
        
    

            
    idx = np.random.choice(np.arange(data.shape[0]),batch_size)
    batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
    pred_y = pred_y.transpose(1,0)
    plt.plot(pred_y[0].detach().cpu())
    plt.plot(batch_y[0].detach().cpu())

    
    dist = np.load('../data/dist_l.npy')
    K = func_m(t.reshape(-1,1))
    plt.plot(K.detach().cpu().numpy(), label='dist pred')
    plt.plot(dist[:crop][::k], label='dist')
    plt.legend()