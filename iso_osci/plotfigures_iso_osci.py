import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

d = 4
test_loss_small_deflate=np.load('involution_box=1000.0_deflate=0.5/losses_test_all.npy')
test_loss_large_deflate=np.load('involution_box=1000.0_deflate=1.0/losses_test_all.npy')    
test_loss_small_deflate = test_loss_small_deflate[:,-1]
test_loss_large_deflate = test_loss_large_deflate[:,-1]


x = np.arange(1, d+1)

plt.semilogy(x, test_loss_small_deflate, marker='o')
plt.semilogy(x, test_loss_large_deflate, marker='o')    
plt.xlabel('Learning the $k$-th conserved quantity $I_k$')
plt.ylabel('Validation error')
#plt.title('Harmonic Oscillator')
plt.legend([r'$\alpha = 0.5$', r'$\alpha = 1.0$'])
plt.xticks(x)
plt.savefig('test_error_isco_osci.svg')
plt.close()

    
