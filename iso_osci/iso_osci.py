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

###################### Define vector field f(z) and z #############################
def f(x):
    m = 1; k1 = 1; k2 = 1
    xx = x[:, 0] # x position
    px = x[:, 1] # x momentum
    yy = x[:, 2]
    py = x[:, 3]

    xx_dot = px/m
    px_dot = -k1*xx
    yy_dot = py/m
    py_dot = -k2*yy

    return torch.transpose(torch.stack([xx_dot, px_dot, yy_dot, py_dot]),0,1)

# z is input
input_train = (np.random.rand(10000, 4)-.5)*4
input_val = (np.random.rand(10000, 4)-.5)*4
input_test = (np.random.rand(10000, 4)-.5)*4


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
        denominator = ((projected**2).sum(1))**1 # L2^4 of the projected vectors
        
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

        denominator = ((projected**2).sum(1))**1 # L2^4 of the projected vectors
        
        loss = (numerator/denominator).mean()
        

    return loss
        
        


###################### Training a network to minimize H1 #############################

cuda = torch.device('cuda')
cpu = torch.device('cpu')


np.random.seed(0)
torch.manual_seed(0)

model_h1 = FCNET(w=400)
epochs = 400
lr_decay_epoch = 200

n_train = input_train.shape[0]
batch_size = 128

lr = 1e-3
optimizer_h1 = optim.Adam(model_h1.parameters(), lr=lr)

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
        inputs = torch.tensor(input_test, requires_grad=True, dtype=torch.float, device='cpu')
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
plt.title('Learning H1')
plt.xlabel('Number of Iterations')
plt.show()

###################### Training a network to minimize H2 #############################
np.random.seed(1)
torch.manual_seed(1)

model_h2 = FCNET(w=400)
epochs = 1000
lr_decay_epoch = 500

n_train = input_train.shape[0]
batch_size = 128

lr = 1e-3
optimizer_h2 = optim.Adam(model_h2.parameters(), lr=lr)

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
plt.title('Learning H2')
plt.xlabel('Number of Iterations')
plt.show()



###################### Training a network to minimize H3 #############################
np.random.seed(2)
torch.manual_seed(2)

model_h3 = FCNET(w=400)
epochs = 2000
lr_decay_epoch = 1000

n_train = input_train.shape[0]
batch_size = 128

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
plt.title('Learning H3')
plt.xlabel('Number of Iterations')
plt.show()


###################### Training a network to minimize H4 #############################
np.random.seed(3)
torch.manual_seed(3)

model_h4 = FCNET(w=400)
epochs = 2000
lr_decay_epoch = 1000

n_train = input_train.shape[0]
batch_size = 128

lr = 1e-3
optimizer_h4 = optim.Adam(model_h4.parameters(), lr=lr)

log = 1

losses_train_h4 = []
losses_test_h4 = []



for epoch in range(epochs):

    model_h4.train()
    optimizer_h4.zero_grad()
    choices = np.random.choice(n_train, batch_size)
    inputs = torch.tensor(input_train[choices], requires_grad=True, dtype=torch.float, device=cpu)


    if (epoch+1) % lr_decay_epoch == 0:
        for opt_param in optimizer_h4.param_groups:
            lr = lr * 0.5
            opt_param['lr'] = lr

    # computing loss
    loss = compute_loss_h_all(f, [model_h1, model_h2, model_h3, model_h4], inputs)



    loss.backward()
    optimizer_h4.step()

    if epoch%log == 0:
        # compute test loss
        choices = np.random.choice(n_train, batch_size)
        inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)
        loss_test = compute_loss_h_all(f, [model_h1, model_h2, model_h3, model_h4], inputs, False)
        losses_test_h4.append(loss_test.data.numpy())
        losses_train_h4.append(loss.data.numpy())        
        print('Epoch:  %d | Loss_train: %.4f | Loss_test: %.4f' %(epoch, loss, loss_test))
        
# setting fixing h4
for p in model_h4.parameters():
    p.requires_grad=False


plt.plot(losses_train_h4)
plt.plot(losses_test_h4)
plt.legend(['Train loss', 'Test loss'])
plt.yscale('log')
plt.title('Learning H4')
plt.xlabel('Number of Iterations')
plt.show()

    

