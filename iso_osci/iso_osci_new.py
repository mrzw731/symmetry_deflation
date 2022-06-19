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

batch_size = 128                             # num of training samples per iteration
width = 400                                  # width of the neural networks representing H's
num_Hs = 4                                   # number of H's to be learned
epochs_list = [400, 1000, 2000, 2000]        # num of epochs to learn each H


lr_decay_list = [x//2 for x in epochs_list]  # num of epochs after which lr is halfed
lr_init = 1e-3                               # initial learning rate

deflate = .5                                 # deflation power in the denominator



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
        denominator = ((projected**2).sum(1))**deflate # L2^4 of the projected vectors
        
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

        denominator = ((projected**2).sum(1))**deflate # L2^4 of the projected vectors
        
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


###################### Training networks to minimize all H's #############################

cuda = torch.device('cuda')
cpu = torch.device('cpu')
learned_models = []

for idx in range(num_Hs):
    np.random.seed(idx)
    torch.manual_seed(idx)

    model_now = FCNET(w=width).to(torch.float)
    epochs = epochs_list[idx]
    lr_decay_epoch = lr_decay_list[idx]

    n_train = input_train.shape[0]
    lr = lr_init
    optimizer_now = optim.Adam(model_now.parameters(), lr=lr)

    log = 1
    losses_train_now = []
    losses_test_now = []

    for epoch in range(epochs):

        model_now.train()
        optimizer_now.zero_grad()
        choices = np.random.choice(n_train, batch_size)
        inputs = torch.tensor(input_train[choices], requires_grad=True, dtype=torch.float, device='cpu')
        if (epoch+1) % lr_decay_epoch == 0:
            for opt_param in optimizer_now.param_groups:
                lr = lr * 0.5
                opt_param['lr'] = lr

        # computing loss
        loss = compute_loss_h_all(f, learned_models + [model_now], inputs)

        loss.backward()
        optimizer_now.step()

        if epoch%log == 0:
            # compute test loss
            choices = np.random.choice(n_train, batch_size)
            inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)        
            #inputs = torch.tensor(input_test, requires_grad=True, dtype=torch.float, device='cpu')
            loss_test = compute_loss_h_all(f, learned_models + [model_now], inputs, False)
            losses_test_now.append(loss_test.data.numpy())
            losses_train_now.append(loss.data.numpy())
            print('Epoch:  %d | Loss_train: %.4f | Loss_test: %.4f' %(epoch, loss, loss_test))
        
    # setting fixing h1
    for p in model_now.parameters():
        p.requires_grad=False

    plt.plot(losses_train_now)
    plt.plot(losses_test_now)
    plt.legend(['Train loss', 'Test loss'])
    plt.yscale('log')
    plt.title('Learning H%d' %(idx+1))
    plt.xlabel('Number of Iterations')
    plt.savefig('Learning_H%d.png' % (idx+1))
    plt.close()

    learned_models.append(model_now)
