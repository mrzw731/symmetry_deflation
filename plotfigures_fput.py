import numpy as np
import matplotlib.pyplot as plt
import pdb

for N_sites in range(3, 11):
    test_loss_fput_quad=np.load('fput_quad/N_sites=%d/losses_test_all.npy' % N_sites)
    test_loss_fput_quad = test_loss_fput_quad[:,-1]
    
    test_loss_toda=np.load('toda/N_sites=%d/losses_test_all.npy' % N_sites)
    test_loss_toda = test_loss_toda[:,-1]

    x = np.arange(1, 2*N_sites+1)

    plt.semilogy(x, test_loss_fput_quad)
    plt.semilogy(x, test_loss_toda)
    plt.xlabel('Learning the n-th H')
    plt.ylabel('Test error')
    plt.title('N_sites = %d' % N_sites)
    plt.legend(['fput', 'toda'])
    plt.xticks(x)
    plt.savefig('test_error_fput_toda_Nsites_%d.png'% N_sites)
    plt.close()

    
