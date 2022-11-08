#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:44:05 2022

@author: dliu
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



dist = torch.tensor([[6.19101754e-28], [8.61125475e-28], [1.19773230e-27], [1.66587303e-27],
        [2.31692900e-27], [3.22234571e-27], [4.48146490e-27], [6.23241276e-27], [8.66723627e-27],
        [1.20529445e-26], [1.67607578e-26], [2.33067643e-26], [3.24084328e-26], [4.50631614e-26],
        [6.26574519e-26], [8.71186451e-26], [1.21125803e-25], [1.68402779e-25], [2.34125507e-25],
        [3.25487976e-25], [4.52488727e-25], [6.29023722e-25], [8.74404889e-25], [1.21547006e-24],
        [1.68951455e-24], [2.34836406e-24], [3.26403321e-24], [4.53658655e-24], [6.30505898e-24],
        [8.76262554e-24], [1.21776735e-23], [1.69230716e-23], [2.35168240e-23], [3.26785347e-23],
        [4.54078271e-23], [6.30932520e-23], [8.76635306e-23], [1.21797579e-22], [1.69216149e-22],
        [2.35086777e-22], [3.26586052e-22], [4.53680261e-22], [6.30209199e-22], [8.75390789e-22],
        [1.21590981e-21], [1.68881731e-21], [2.34555527e-21], [3.25754270e-21], [4.52392884e-21],
        [6.28235344e-21], [8.72387962e-21], [1.21137163e-20], [1.68199735e-20], [2.33535613e-20],
        [3.24235492e-20], [4.50139715e-20], [6.24903800e-20], [8.67476585e-20], [1.20415063e-19],
        [1.67140628e-19], [2.31985630e-19], [3.21971668e-19], [4.46839367e-19], [6.20100466e-19],
        [8.60496695e-19], [1.19402254e-18], [1.65672969e-18], [2.29861499e-18], [3.18901005e-18],
        [4.42405211e-18], [6.13703897e-18], [8.51278037e-18], [1.18074868e-17], [1.63763300e-17],
        [2.27116318e-17], [3.14957750e-17], [4.36745073e-17], [6.05584879e-17], [8.39639479e-17],
        [1.16407514e-16], [1.61376027e-16], [2.23700190e-16], [3.10071962e-16], [4.29760913e-16],
        [5.95605985e-16], [8.25388385e-16], [1.14373190e-15], [1.58473300e-15], [2.19560063e-15],
        [3.04169274e-15], [4.21348491e-15], [5.83621106e-15], [8.08319955e-15], [1.11943190e-14],
        [1.55014884e-14], [2.14639540e-14], [2.97170642e-14], [4.11397017e-14], [5.69474955e-14],
        [7.88216540e-14], [1.09087007e-13], [1.50958022e-13], [2.08878695e-13], [2.88992074e-13],
        [3.99788772e-13], [5.53002545e-13], [7.64846905e-13], [1.05772235e-12], [1.46257294e-12],
        [2.02213870e-12], [2.79544355e-12], [3.86398722e-12], [5.34028642e-12], [7.37965464e-12],
        [1.01964455e-11], [1.40864465e-11], [1.94577467e-11], [2.68732748e-11], [3.71094100e-11],
        [5.12367183e-11], [7.07311467e-11], [9.76271248e-11], [1.34728325e-10], [1.85897721e-10],
        [2.56456686e-10], [3.53733972e-10], [4.87820672e-10], [6.72608150e-10], [9.27214610e-10],
        [1.27794524e-09], [1.76098471e-09], [2.42609441e-09], [3.34168782e-09], [4.60179531e-09],
        [6.33561833e-09], [8.72063105e-09], [1.20005397e-08], [1.65098910e-08], [2.27077780e-08],
        [3.12239860e-08], [4.29221427e-08], [5.89860976e-08], [8.10380189e-08], [1.11299772e-07],
        [1.52813326e-07], [2.09741605e-07], [2.87778924e-07], [3.94710562e-07], [5.41175178e-07],
        [7.41702915e-07], [1.01612784e-06], [1.39150828e-06], [1.90473563e-06], [2.60607535e-06],
        [3.56396917e-06], [4.87154132e-06], [6.65540453e-06], [9.08756608e-06], [1.24015063e-05],
        [1.69138650e-05], [2.30536537e-05], [3.14015469e-05], [4.27426514e-05], [5.81372553e-05],
        [7.90155117e-05], [1.07303903e-04], [1.45593783e-04], [1.97365463e-04], [2.67285353e-04],
        [3.61598825e-04], [4.88647907e-04], [6.59550972e-04], [8.89091353e-04], [1.19687352e-03],
        [1.60881908e-03], [2.15908992e-03], [2.89254160e-03], [3.86782438e-03], [5.16125896e-03],
        [6.87161216e-03], [9.12587537e-03], [1.20860873e-02], [1.59571171e-02], [2.09950920e-02],
        [2.75157495e-02], [3.59013196e-02], [4.66034445e-02], [6.01378918e-02], [7.70640675e-02],
        [9.79380677e-02], [1.23221454e-01], [1.53117946e-01], [1.87295109e-01], [2.24425363e-01], 
        [2.61446588e-01], [2.92391844e-01], [3.06562343e-01], [2.85706313e-01],
        [1.99701869e-01], [0.00000000e+00]]).to(device)
dist = dist/dist.sum()
  
beta = 1.5 #.5
gamma = 1 #.1

class funcI(nn.Module):    
    def __init__(self):
        super(funcI, self).__init__()

        self.memory = nn.Sequential(
            nn.Linear(2, 20),
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



# fff = Memory().to(device)
# optimizer = optim.RMSprop(fff.parameters(), lr=1e-3)

# batch_y0 = batch_y[:,[0],:].to(device)
# for itr in range(1, 5000):
#     optimizer.zero_grad()
    
#     pred_dist = fff(batch_t)
#     loss = nn.functional.mse_loss(pred_dist, torch.tensor(dist[-40:],dtype=torch.float32).to(device))

#     loss.backward()
#     optimizer.step()
    
#     if itr%100==0:
#         print(loss.item())

# pred_dist = fff(batch_t)  
# plt.plot(pred_dist.cpu().detach())
# plt.plot(dist[-40:].cpu())


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # for m in self.memory.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)
                
        
        self.g = Memory().to(device)#(time)
        # self.g = fff
        
        self.funcI = funcI().to(device)
        
        
        # self.beta = 1.5#nn.Parameter(torch.tensor(0.1).to(device), requires_grad=True)
        # self.gamma = 1#nn.Parameter(torch.tensor(0.1).to(device), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.2).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.gamma = nn.Parameter(torch.tensor(0.2).to(device), requires_grad=True)
        
        # self.dx = 0.40578396351620016
    def forward(self, t, y):
        self.dt = t[0]-t[1]
        me = self.g(t)
        # me = dist
        self.me = me
        # series = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)        
        
        y = y.transpose(1,0)
        solution = y.clone()
        # print(solution.shape)
        
        diff = torch.zeros(t.shape[0], y.shape[1], 3, dtype=torch.float32).to(device)
        # print(diff.shape)

        diff[0,:,:] = self.ff(solution, me, 1)[0]
        # print(self.ff(solution, me, 1).shape)
        y1 = y + self.ff(solution, me, 1)*self.dt
            
        # print(y1.shape)
        solution = torch.cat((solution, y1), dim=0)
        # print(solution.shape)
        
        diff[1,:,:] = self.ff(solution, me, 2)[0]
        y2 = y1 + self.ff(solution, me, 2)*self.dt
        # print(y1.shape)

        solution = torch.cat((solution, y2), dim=0)

        diff[2,:,:] = self.ff(solution, me, 3)[0]
        y3 = y2 + self.ff(solution, me, 3)*self.dt
        solution = torch.cat((solution, y3), dim=0)
        # print(solution.shape)

        for ii in range(4,t.shape[0]):
            diff[ii-1,:,:] = self.ff(solution, me, ii)[0]
            y3 = y3 + self.ff(solution, me, ii)*self.dt
            solution = torch.cat((solution, y3), dim=0)


        return solution, diff
        # return torch.cat((dSdt,dIdt,dRdt),1)
    
    def ff(self, solution, me, j):
        beta = self.beta
        gamma = self.gamma
        
        S, I, R = torch.split(solution[[-1],:,:],1,dim=2)
        # print(S.shape)
        self.integro = self.integral(solution,me,j)
        # print('dd', self.integro.shape)

        dSdt = -beta * S * I + self.integro
        dIdt = beta * S * I - gamma * I
        # dIdt = self.funcI(torch.cat((S,R),2).transpose(1,0)).transpose(1,0)
        dRdt = gamma * I - self.integro
        
        # print('ddfsf', dRdt.shape, dIdt.shape)
        return torch.cat((dSdt,dIdt,dRdt),2)
    
    def integral(self, solution, me, j):
        S, I, R = torch.split(solution,1,dim=2)

        batchsize = S.shape[1]
        
        ME = me[-j:]
        
        integro = I[:,:,0] * ME * self.dt
        # print(integro.sum(0))

        return integro.sum(0).reshape(1, batchsize, 1)
    

k = 5
t = torch.linspace(0., 80./k, 200//k).to(device)
t = torch.flip(t, [0]).reshape([-1,1])
# t = t.repeat(1,2).reshape(-1,1)

func = ODEFunc().to(device)
y0 = torch.tensor([[[1, 0.001, 0]], [[0.66667574, 0.09769345, 0.23663081]]], dtype=torch.float32).to(device)
results, diff = func(t, y0)
results = results.transpose(1,0)
diff = diff.transpose(1,0)

plt.figure()
plt.plot(results[0,:,:].cpu().detach())




import numpy as np
traindata = np.load('./data/train_sir.npy')






'''
###################################
############## RMSprop ############
###################################

k = 5
t = torch.linspace(0., 80./k, 200//k).to(device)
t = torch.flip(t, [0]).reshape([-1,1])

# batch = traindata[np.arange(batch_size).reshape(-1,1)+np.arange(200//k)]
batch_y = torch.tensor(traindata[:,:200//k,:], dtype=torch.float32).to(device)
batch_y0 = batch_y[:,[0],:].to(device)
batch_t = t
diff_num = (batch_y[:,1:,:] - batch_y[:,:-1,:])/(t[1]-t[0])

func = ODEFunc().to(device)
optimizer = optim.Adam(func.parameters(), lr=1e-4)
for itr in range(1, 10000):
    optimizer.zero_grad()
    
    pred_y, diff = func(batch_t, batch_y0)
    pred_y = pred_y.transpose(1,0)
    diff = diff.transpose(1,0)

    loss_mse = torch.mean(torch.abs(pred_y - batch_y))
    loss_diff = torch.mean(torch.abs(diff[:,:-1,:] - diff_num))
    # loss = loss_mse
    loss = loss_diff+loss_mse

    loss.backward()
    optimizer.step()
    
    if itr%100==0:
        print(f'itr: {itr}. loss_diff: {loss_diff.item():.5e}, loss_mse: {loss_mse.item():.5e}')

'''


#################################
############## LBFGS ############
#################################
k = 5
t = torch.linspace(0., 80./k, 200//k).to(device)
t = torch.flip(t, [0]).reshape([-1,1])

batch_y = torch.tensor(traindata[:,:200//k,:], dtype=torch.float32).to(device)
batch_y0 = batch_y[:,[0],:].to(device)
batch_t = t
diff_num = (batch_y[:,1:,:] - batch_y[:,:-1,:])/(t[1]-t[0])

func = ODEFunc().to(device)
optimizer = optim.LBFGS(func.parameters(), lr=5e-2)
for itr in range(1,100):
    def closure():
        optimizer.zero_grad()
        
        pred_y, diff = func(batch_t, batch_y0)
        pred_y = pred_y.transpose(1,0)
        diff = diff.transpose(1,0)[:,:-1,:]
    
        loss_mse = nn.functional.mse_loss(pred_y, batch_y)
        loss_diff = nn.functional.mse_loss(diff, diff_num)
        loss_sum = nn.functional.mse_loss(pred_y.sum(2), torch.ones_like(pred_y.sum(2)))
        
        loss = loss_mse
        # loss = loss_diff + loss_mse# + .01*loss_sum
        print(f'loss_diff: {loss_diff.item():.3e}, loss_mse: {loss_mse.item():.3e}, beta:{func.beta.item():.2f}, gamma:{func.gamma.item():.2f}')
        loss.backward()
        return loss
    optimizer.step(closure)
    
#################################
#################################




pred_y, diff = func(batch_t, batch_y0)
pred_y = pred_y.transpose(1,0)
diff = diff.transpose(1,0)


jj = 20
fig, ax = plt.subplots(1,2,figsize=[12,4])
ax[0].plot(pred_y[jj,:,:].cpu().detach(), label=['predict S', 'predict I', 'predict R'])
ax[0].plot(batch_y[jj,:,:].cpu(), label=['S', 'I', 'R'])
ax[0].plot([],'.',label=f'MSE: {((batch_y-pred_y)**2).mean().item():.2e}')
ax[0].plot([],'.',label=f'beta(1.5): {func.beta.item():.2f}')
ax[0].plot([],'.',label=f'gamma(1): {func.gamma.item():.2f}')
ax[0].legend()

ax[1].plot(func.me.cpu().detach()*func.dt.cpu(), label='predict dist')
ax[1].plot(dist[-40:].cpu(), label='dist')
ax[1].plot([],'.',label=f'MSE: {((func.me.cpu().detach()-dist[-40:].cpu())**2).mean().item():.2e}')
ax[1].legend()
# fig.savefig('./figures/unkonwn_I_without_diff.png')
# fig.savefig('./figures/unkonwn_I_with_diff.png')
# fig.savefig('./figures/konwn_I_without_diff.png')
# fig.savefig('./figures/konwn_I_with_diff.png')

plt.figure()
plt.plot(diff[jj].cpu().detach())
plt.plot(diff_num[jj].cpu())







testdata = np.load('./data/test_sir.npy')

test_y = torch.tensor(testdata[:,:200//k,:], dtype=torch.float32).to(device)
test_y0 = test_y[:,[0],:].to(device)

pred_y, diff = func(batch_t, test_y0)
pred_y = pred_y.transpose(1,0)
diff = diff.transpose(1,0)


jj = 20
fig, ax = plt.subplots(1,2,figsize=[12,4])
ax[0].plot(pred_y[jj,:,:].cpu().detach(), label=['predict S', 'predict I', 'predict R'])
ax[0].plot(test_y[jj,:,:].cpu(), label=['S', 'I', 'R'])
ax[0].plot([],'.',label=f'MSE: {((test_y-pred_y)**2).mean().item():.2e}')
ax[0].plot([],'.',label=f'beta(1.5): {func.beta.item():.2f}')
ax[0].plot([],'.',label=f'gamma(1): {func.gamma.item():.2f}')
ax[0].legend()

ax[1].plot(func.me.cpu().detach()*func.dt.cpu(), label='predict dist')
ax[1].plot(dist[-40:].cpu(), label='dist')
ax[1].plot([],'.',label=f'MSE: {((func.me.cpu().detach()-dist[-40:].cpu())**2).mean().item():.2e}')
ax[1].legend()

plt.figure()
plt.plot(diff[jj].cpu().detach())
plt.plot(diff_num[jj].cpu())

print(f'MSE: {((pred_y-test_y)**2).mean():.2e}') #2.724662408581935e-05 with loss_diff
                                             #1.2362557981759892e-06 without loss_diff



# k = 4
# t = torch.linspace(0., 80./k, 200//k).to(device)
# t = torch.flip(t, [0]).reshape([-1,1])

# y0 = torch.tensor([[[1, 0.001, 0]], [[1.,0.002,0.]]], dtype=torch.float32).to(device)
# results, diff = func(t, y0)
# results = results.transpose(1,0)
# diff = diff.transpose(1,0)

# plt.figure()
# plt.plot(results[0,:,:].cpu().detach())


