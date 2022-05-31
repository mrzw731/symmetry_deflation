import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import pdb
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

C = .1
N_sites = 4
batch_size = 500
width = 800

#torch.set_default_dtype(torch.float)


###################### Define vector field f(z) and z #############################
def f(x):
    u = x[:, :N_sites]
    v = x[:, N_sites:]

    abs_uv_sqrd = u**2 + v**2
    # AL with periodic
    u_dot = -C * torch.cat([v[:,1:2]+v[:, -1:], v[:, :-2]+v[:, 2:], v[:, -2:-1]+v[:, :1]], dim=1)*(1 + abs_uv_sqrd/(2*C))
    v_dot = C * torch.cat([u[:,1:2]+u[:, -1:], u[:, :-2]+u[:, 2:], u[:, -2:-1]+u[:, :1]], dim=1)*(1+ abs_uv_sqrd/(2*C))

    # periodic
    #u_dot = -C * torch.cat([v[:,1:2]-2*v[:, 0:1]+v[:, -1:], v[:, :-2]+v[:, 2:]-2*v[:, 1:-1], v[:, -2:-1]-2*v[:, -1:]+v[:, :1]], dim=1) - abs_uv_sqrd*v
    #v_dot = C * torch.cat([u[:,1:2]-2*u[:, 0:1]+u[:, -1:], u[:, :-2]+u[:, 2:]-2*u[:, 1:-1], u[:, -2:-1]-2*u[:, -1:]+u[:, :1]], dim=1) + abs_uv_sqrd*u

    return torch.cat([u_dot, v_dot], dim=1)

# z is input
input_train = (np.random.rand(100000, 2*N_sites)-.5)*8
input_val = (np.random.rand(100000, 2*N_sites)-.5)*8
input_test = (np.random.rand(100000, 2*N_sites)-.5)*8


input_d = input_train.shape[1]


###################### Define the network for H #############################

class FCNET(nn.Module):
    def __init__(self,w=200):
        super(FCNET, self).__init__()
        self.l1 = nn.Linear(input_d,w)
        self.l2 = nn.Linear(w,w)
        self.l23 = nn.Linear(w,w)
        self.l3 = nn.Linear(w,1)
    
    def forward(self, x):
        bs = x.shape[0]
        f = nn.SiLU()
        self.x1 = f(self.l1(x))
        self.x2 = f(self.l2(self.x1))
        self.x23 = f(self.l23(self.x2))
        self.x3 = self.l3(self.x23) # scalar output
        return self.x3


###################### Define losses for H's #############################


def compute_loss_h_all(f, models, inputs, training=True):
    """
    f: vector field
    models: a list of [model_h1, model_h2, ..., model_hs], len(models) >=1
    inputs: array of size Nx4
    training: training or testing phase    
    """
    num_samples = inputs.shape[0]
    f_normalized = F.normalize(f(inputs))
    num_models = len(models)

    if num_models == 1:
        model_h1 = models[0]
        H = model_h1(inputs)
        grad_H,  = torch.autograd.grad(H, inputs, grad_outputs=H.data.new(H.shape).fill_(1),
                                       create_graph=training)
        grad_H_normalized = F.normalize(grad_H)
        loss = (((f_normalized * grad_H_normalized).sum(1))**2).mean()


    elif num_models == 2:
        model_h1, model_h2 = models
        # H1
        H1 = model_h1(inputs)
        grad_H1,  = torch.autograd.grad(H1, inputs, grad_outputs=H1.data.new(H1.shape).fill_(1),
                                    create_graph=False)
        grad_H1_normalized = F.normalize(grad_H1)
        # H2
        H2 = model_h2(inputs)
        grad_H2,  = torch.autograd.grad(H2, inputs, grad_outputs=H2.data.new(H2.shape).fill_(1),
                                        create_graph=training)
        grad_H2_normalized = F.normalize(grad_H2)

        # numerator 
        numerator = ((f_normalized * grad_H2_normalized).sum(1))**2 # squared of the inner products

        # denominator
        inner = (grad_H1_normalized * grad_H2_normalized).sum(1, keepdim=True)
        projected = grad_H2_normalized - inner*grad_H1_normalized
        denominator = ((projected**2).sum(1))**3 # L2^4 of the projected vectors
        
        loss = (numerator/denominator).mean()
        

    else:
        # num_models >= 3
        model_last = models.pop() # popping the last model from the list to train, the remaining models are fixed
        # H1 till H_{n-1}
        grad_H_fixed_list = []
        for model_fixed in models:
            H_fixed = model_fixed(inputs)
            grad_H_fixed,  = torch.autograd.grad(H_fixed, inputs, grad_outputs=H_fixed.data.new(H_fixed.shape).fill_(1),
                                               create_graph=False)
            grad_H_fixed_list.append(grad_H_fixed)

        #pdb.set_trace()
        grad_Hs_fixed = torch.stack(grad_H_fixed_list, dim=2)
        Q, R = torch.qr(grad_Hs_fixed)


        # H_last
        H_last = model_last(inputs)
        grad_H_last,  = torch.autograd.grad(H_last, inputs, grad_outputs=H_last.data.new(H_last.shape).fill_(1),
                                        create_graph=training)
        grad_H_last_normalized = F.normalize(grad_H_last)

        # numerator
        numerator = ((f_normalized * grad_H_last_normalized).sum(1))**2 # squared of the inner products

        # denominator
        inner = torch.matmul(grad_H_last_normalized.view(num_samples, 1, -1), Q)
        project_inplane = (torch.matmul(inner, torch.transpose(Q, 1, 2))).view(num_samples, -1)
        projected = grad_H_last_normalized - project_inplane

        denominator = ((projected**2).sum(1))**3 # L2^4 of the projected vectors
        
        loss = (numerator/denominator).mean()
        

    return loss
        
        
def sanity_check(f, models, inputs):

    num_samples = inputs.shape[0]
    f_normalized = F.normalize(f(inputs))
    num_models = len(models)

    grad_H_normalized_list = []
    for model in models:
        H = model(inputs)
        grad_H,  = torch.autograd.grad(H, inputs, grad_outputs=H.data.new(H.shape).fill_(1),
                                             create_graph=False)
        grad_H_normalized = F.normalize(grad_H)
        grad_H_normalized_list.append(grad_H_normalized)

    grad_Hs = torch.stack(grad_H_normalized_list, dim=2)

    inner = [np.arccos((grad_H * f_normalized).sum(1).data.numpy())/np.pi for grad_H in grad_H_normalized_list]

    _, S, V = torch.svd(grad_Hs)

    return inner, S.data.numpy()
    

###################### Training a network to minimize H1 #############################

cuda = torch.device('cuda')
cpu = torch.device('cpu')


np.random.seed(0)
torch.manual_seed(0)

model_h1 = FCNET(w=width).to(torch.float)
epochs = 1000
lr_decay_epoch = 500

n_train = input_train.shape[0]

lr = 1e-3
optimizer_h1 = optim.Adam(model_h1.parameters(), lr=lr)
#optimizer_h1 = optim.NAdam(model_h1.parameters(), lr=lr)

log = 1

losses_train_h1 = []
losses_test_h1 = []



for epoch in range(epochs):

    model_h1.train()
    optimizer_h1.zero_grad()
    choices = np.random.choice(n_train, batch_size)
    inputs = torch.tensor(input_train[choices], requires_grad=True, dtype=torch.float, device='cpu')


    if (epoch+1) % lr_decay_epoch == 0:
        for opt_param in optimizer_h1.param_groups:
            lr = lr * 0.5
            opt_param['lr'] = lr

    # computing loss
    loss = compute_loss_h_all(f, [model_h1], inputs)


    loss.backward()
    optimizer_h1.step()

    if epoch%log == 0:
        # compute test loss
        choices = np.random.choice(n_train, batch_size)
        inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)        
        #inputs = torch.tensor(input_test, requires_grad=True, dtype=torch.float, device='cpu')
        loss_test = compute_loss_h_all(f, [model_h1], inputs, False)
        losses_test_h1.append(loss_test.data.numpy())
        losses_train_h1.append(loss.data.numpy())
        print('Epoch:  %d | Loss_train: %.4f | Loss_test: %.4f' %(epoch, loss, loss_test))
        
# setting fixing h1
for p in model_h1.parameters():
    p.requires_grad=False


plt.plot(losses_train_h1)
plt.plot(losses_test_h1)
plt.legend(['Train loss', 'Test loss'])
plt.yscale('log')
plt.title('Learning H1, C = %.1f, Num sites = %d' %(C, N_sites))
plt.xlabel('Number of Iterations')
plt.savefig('Learning_H1_N=%d.png' % N_sites)
plt.close()


# sanity check
#choices = np.random.choice(n_train, 10)
#inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)

#inner, S = sanity_check(f, [model_h1, model_h1], inputs)



###################### Training a network to minimize H2 #############################
np.random.seed(2)
torch.manual_seed(2)

model_h2 = FCNET(w=width).to(torch.float)
epochs = 5000
lr_decay_epoch = 2500

n_train = input_train.shape[0]

lr = 1e-3
optimizer_h2 = optim.Adam(model_h2.parameters(), lr=lr)
#optimizer_h2 = optim.NAdam(model_h2.parameters(), lr=lr)

log = 1

losses_train_h2 = []
losses_test_h2 = []



for epoch in range(epochs):

    model_h2.train()
    optimizer_h2.zero_grad()
    choices = np.random.choice(n_train, batch_size)
    inputs = torch.tensor(input_train[choices], requires_grad=True, dtype=torch.float, device=cpu)


    if (epoch+1) % lr_decay_epoch == 0:
        for opt_param in optimizer_h2.param_groups:
            lr = lr * 0.5
            opt_param['lr'] = lr

    # computing loss
    loss = compute_loss_h_all(f, [model_h1, model_h2], inputs)


    loss.backward()
    optimizer_h2.step()

    if epoch%log == 0:
        # compute test loss
        choices = np.random.choice(n_train, batch_size)
        inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)
        #inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float)
        loss_test = compute_loss_h_all(f, [model_h1, model_h2], inputs, False)
        losses_test_h2.append(loss_test.data.numpy())
        losses_train_h2.append(loss.data.numpy())        
        print('Epoch:  %d | Loss_train: %.4f | Loss_test: %.4f' %(epoch, loss, loss_test))
        
# setting fixing h2
for p in model_h2.parameters():
    p.requires_grad=False

plt.plot(losses_train_h2)
plt.plot(losses_test_h2)
plt.legend(['Train loss', 'Test loss'])
plt.yscale('log')
plt.title('Learning H2, C = %.1f, Num sites = %d' %(C, N_sites))
plt.xlabel('Number of Iterations')
#plt.show()
plt.savefig('Learning_H2_N=%d.png' % N_sites)
plt.close()

# sanity check

#choices = np.random.choice(n_train, 10)
#inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)

#inner, S = sanity_check(f, [model_h1, model_h2], inputs)
#pdb.set_trace()


###################### Training a network to minimize H3 #############################
np.random.seed(2)
torch.manual_seed(2)

model_h3 = FCNET(w=width).to(torch.float)
epochs = 8000
lr_decay_epoch = 4000

n_train = input_train.shape[0]

lr = 1e-3
optimizer_h3 = optim.Adam(model_h3.parameters(), lr=lr)

log = 1

losses_train_h3 = []
losses_test_h3 = []



for epoch in range(epochs):

    model_h3.train()
    optimizer_h3.zero_grad()
    choices = np.random.choice(n_train, batch_size)
    inputs = torch.tensor(input_train[choices], requires_grad=True, dtype=torch.float, device=cpu)


    if (epoch+1) % lr_decay_epoch == 0:
        for opt_param in optimizer_h3.param_groups:
            lr = lr * 0.5
            opt_param['lr'] = lr

    # computing loss
    loss = compute_loss_h_all(f, [model_h1, model_h2, model_h3], inputs)



    loss.backward()
    optimizer_h3.step()

    if epoch%log == 0:
        # compute test loss
        choices = np.random.choice(n_train, batch_size)
        inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)
        loss_test = compute_loss_h_all(f, [model_h1, model_h2, model_h3], inputs, False)
        losses_test_h3.append(loss_test.data.numpy())
        losses_train_h3.append(loss.data.numpy())        
        print('Epoch:  %d | Loss_train: %.4f | Loss_test: %.4f' %(epoch, loss, loss_test))
        
# setting fixing h3
for p in model_h3.parameters():
    p.requires_grad=False


plt.plot(losses_train_h3)
plt.plot(losses_test_h3)
plt.legend(['Train loss', 'Test loss'])
plt.yscale('log')
plt.title('Learning H3, C = %.1f, Num sites = %d' %(C, N_sites))
plt.xlabel('Number of Iterations')
#plt.show()
plt.savefig('Learning_H3_N=%d.png' % N_sites)
plt.close()

# sanity check

#choices = np.random.choice(n_train, 10)
#inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)

#inner, S = sanity_check(f, [model_h1, model_h2, model_h3], inputs)
#pdb.set_trace()
