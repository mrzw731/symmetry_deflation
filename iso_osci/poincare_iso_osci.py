import numpy as np
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--width', type=int, default=400)
parser.add_argument('--num_Hs', type=int, default=4)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr_init', type=float, default=1e-3)
parser.add_argument('--train_box', type=float, default=8.)
parser.add_argument('--nu', type=float, default=0.02)
args = parser.parse_args()


batch_size = args.batch_size                # num of training samples per iteration
width = args.width                          # width of the neural networks representing H's
num_Hs = args.num_Hs                        # number of H's to be learned
epochs = args.epochs                        # num of epochs to learn each H

lr_decay = epochs//2                        # num of epochs after which lr is halfed
lr_init = args.lr_init                      # initial learning rate

train_box = args.train_box                  # training domain
nu = args.nu                                # weight of the regularization term


path = './Poincare'
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)


###################### Define vector field f(z) and z #############################

# (q', p') = f(q, p) is in the cannonical coordinate.
# That is: q' = dH/dp, p' = -dH/dq
def f(x):
    m = 1; k1 = 1; k2 = 1
    xx = x[:, 0] # x position
    yy = x[:, 1] # y position
    px = x[:, 2] # x momentum
    py = x[:, 3] # y momentum

    xx_dot = px/m
    yy_dot = py/m    
    px_dot = -k1*xx
    py_dot = -k2*yy

    return torch.transpose(torch.stack([xx_dot, yy_dot, px_dot, py_dot]),0,1)

# z is input
input_train = (np.random.rand(100000, 4)-.5)*train_box
input_val = (np.random.rand(100000, 4)-.5)*train_box
input_test = (np.random.rand(100000, 4)-.5)*train_box


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

def p_bracket_sq(grad_f, grad_g):
    """
    grad_f, grad_g are normalized gradients, and are of size num_samples x (2s)
    returns the poisson bracket squared (of size num_samples x 1)
    """
    num_samples = inputs.shape[0]
    s = inputs.shape[1]//2
    
    dfdq = grad_f[:, :s]
    dfdp = grad_f[:, s:]
    
    dgdq = grad_g[:, :s]
    dgdp = grad_g[:, s:]

    sum1 = (dfdq * dgdp).sum(1)
    sum2 = (dfdp * dgdq).sum(1)
    p_bracket = sum1 - sum2

    return p_bracket**2 # returns p_bracket squared


def compute_loss_h_all(f, models, inputs, nu, training=True):
    """
    f: vector field
    models: type = nn.ModuleList, a list of [model_h1, model_h2, ..., model_hs]
    inputs: array of size Nxd
    nu: weight of the regularization
    training: training or testing phase    
    """
    num_samples = inputs.shape[0]
    f_normalized = F.normalize(f(inputs))
    num_models = len(models)
    list_grad_Hs = []
    loss_conserv = 0
    loss_reg = 0

    for kk in np.arange(num_models):
        model_now = models[kk]
        H_now = model_now(inputs)
        grad_H_now,  = torch.autograd.grad(H_now, inputs, grad_outputs=H_now.data.new(H_now.shape).fill_(1),create_graph=training)
        grad_H_now_normalized = F.normalize(grad_H_now)
        list_grad_Hs.append(grad_H_now_normalized)
        loss_conserv = loss_conserv + (((f_normalized * grad_H_now_normalized).sum(1))**2).mean()

    loss_conserv = loss_conserv/num_models

    for kk in np.arange(num_models):
        for ll in np.arange(kk):
            loss_reg = loss_reg + (((list_grad_Hs[kk] * list_grad_Hs[ll]).sum(1))**2).mean()

    loss_reg = nu * loss_reg * 2 / (num_models-1)/num_models

    loss = loss_conserv + loss_reg

    return loss

def compute_val_loss(f, models, inputs, training=False):
    """
    f: vector field
    models: type = nn.ModuleList, a list of [model_h1, model_h2, ..., model_hs]
    inputs: array of size Nxd
    training: training or testing phase    
    """
    num_samples = inputs.shape[0]
    f_normalized = F.normalize(f(inputs))
    num_models = len(models)
    list_grad_Hs = []
    
    loss_conserv = np.zeros(num_models)
    loss_involution = np.zeros([num_models, num_models])

    for kk in np.arange(num_models):
        model_now = models[kk]
        H_now = model_now(inputs)
        grad_H_now,  = torch.autograd.grad(H_now, inputs, grad_outputs=H_now.data.new(H_now.shape).fill_(1),create_graph=training)
        grad_H_now_normalized = F.normalize(grad_H_now)
        list_grad_Hs.append(grad_H_now_normalized)
        loss_now = (((f_normalized * grad_H_now_normalized).sum(1))**2).mean()
        loss_conserv[kk] = loss_now.data.numpy()

    for kk in np.arange(num_models):
        for ll in np.arange(num_models):
            loss_now = p_bracket_sq(list_grad_Hs[kk], list_grad_Hs[ll]).mean()
            loss_involution[kk, ll] = loss_now.data.numpy()

    return loss_conserv, loss_involution



###################### Training networks to minimize all H's #############################

cuda = torch.device('cuda')
cpu = torch.device('cpu')
losses_train = []
losses_test = []

models = []

for idx in range(num_Hs):
    models.append(FCNET(w=width).to(torch.float))

models = nn.ModuleList(models)

n_train = input_train.shape[0]
lr = lr_init
optimizer = optim.Adam(models.parameters(), lr=lr)
log = 1

for epoch in range(epochs):

    models.train()
    optimizer.zero_grad()

    choices = np.random.choice(n_train, batch_size)
    inputs = torch.tensor(input_train[choices], requires_grad=True, dtype=torch.float, device='cpu')
    if (epoch+1) % lr_decay == 0:
        for opt_param in optimizer.param_groups:
            lr = lr * 0.5
            opt_param['lr'] = lr
        
    # computing loss
    loss = compute_loss_h_all(f, models, inputs, nu)

    loss.backward()
    optimizer.step()

    if epoch%log == 0:
        # compute test loss
        choices = np.random.choice(n_train, batch_size)
        inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)
        loss_test = compute_loss_h_all(f, models, inputs, nu, False)
        losses_test.append(loss_test.data.numpy())
        losses_train.append(loss.data.numpy())
        print('Epoch:  %d | Loss_train: %.4f | Loss_test: %.4f' %(epoch, loss, loss_test))
    

plt.plot(losses_train)
plt.plot(losses_test)
plt.legend(['Train loss', 'Test loss'])
plt.yscale('log')
plt.title('AI Poincare Iso Osci')
plt.xlabel('Number of Iterations')
plt.savefig(path+'/Learning_curve.png')
plt.close()


losses_train = np.array(losses_train)
losses_test = np.array(losses_test)    
np.save(path+'/losses_train.npy', losses_train)
np.save(path+'/losses_test.npy', losses_test)
torch.save(models, path+'/models.pth')

###################### Plotting validation error for H's #############################

choices = np.random.choice(n_train, batch_size)
inputs = torch.tensor(input_test[choices], requires_grad=True, dtype=torch.float, device=cpu)
val_loss_conserv, val_loss_involution = compute_val_loss(f, models, inputs, False)

print('Conservation losses:\n', val_loss_conserv)
print('Involution losses:\n', val_loss_involution)
np.save(path+'/val_loss_conserv.npy', val_loss_conserv)
np.save(path+'/val_loss_involution.npy', val_loss_involution)
    



