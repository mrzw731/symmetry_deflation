import numpy as np
import matplotlib.pyplot as plt
import pdb

for N_sites in range(3, 11):
    test_loss_dnls=np.load('dnls/N_sites=%d/losses_test_all.npy' % N_sites)
    test_loss_dnls = test_loss_dnls[:,-1]
    
    test_loss_alnls=np.load('alnls/N_sites=%d/losses_test_all.npy' % N_sites)
    test_loss_alnls = test_loss_alnls[:,-1]

    x = np.arange(1, 2*N_sites+1)

    plt.semilogy(x, test_loss_dnls)
    plt.semilogy(x, test_loss_alnls)
    plt.xlabel('Learning the n-th H')
    plt.ylabel('Test error')
    plt.title('N_sites = %d' % N_sites)
    plt.legend(['dnls', 'AL'])
    plt.xticks(x)
    plt.savefig('test_error_dnls_al_Nsites_%d.png'% N_sites)
    plt.close()

    
