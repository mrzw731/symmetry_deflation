import numpy as np
import matplotlib.pyplot as plt
import pdb

for N_sites in range(3, 11):
    test_loss_fput_small_deflate=np.load('results_box_10/involution_N_sites=%d_box=10.0_deflate=0.5/losses_test_all.npy' % N_sites)
    test_loss_fput_large_deflate=np.load('results_box_10/involution_N_sites=%d_box=10.0_deflate=1.0/losses_test_all.npy' % N_sites)    
    test_loss_fput_small_deflate = test_loss_fput_small_deflate[:,-1]
    test_loss_fput_large_deflate = test_loss_fput_large_deflate[:,-1]


    x = np.arange(1, 2*N_sites+1)

    plt.semilogy(x, test_loss_fput_small_deflate)
    plt.semilogy(x, test_loss_fput_large_deflate)    
    plt.xlabel('Learning the n-th H')
    plt.ylabel('Test error')
    plt.title('N_sites = %d' % N_sites)
    plt.legend(['fput, deflate = 0.5', 'fput, deflate = 1.0'])
    plt.xticks(x)
    plt.savefig('test_error_box_10_fput_al_Nsites_%d.png'% N_sites)
    plt.close()

    
