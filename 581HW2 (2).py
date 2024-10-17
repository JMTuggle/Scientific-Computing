import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

K = 1
#n0 = 50
L = 4
tol = 1e-4   
col = ['r', 'b', 'g', 'c', 'm', 'k']  # Colors for eigenfunctions

def shoot(y, x, E):
    return [y[1], (K * x**2 - E) * y[0]]

x = np.arange(-L, L + 0.1, 0.1)  # Simulate from -L to L

eigenvalues = []
eigenfunctions = []  

E_start = .1 # Initial guess for the eigenvalue

for modes in range(1, 6):
    E = E_start
    dE = E_start / 100
    for _ in range(1000):
        y0 = [1, np.sqrt(L**2 - E)]
        ys = odeint(shoot, y0, x, args=(E,))        

        if abs(ys[-1, 1] + np.sqrt(L**2 - E) * ys[-1, 0] - 0) < tol:
            eigenvalues.append(E)
            #eigenfunctions.append(np.abs(ys[:, 0])) # Save absolute values of eigenfunction
            break
        if (-1) ** (modes + 1) * (ys[-1, 1] + np.sqrt(L**2 - E) * ys[-1, 0]) > 0:
            E += dE        
        else:
            E -= dE / 2  
            dE /= 2  

    
    E_start = E + 0.1
    
    norm = np.trapz(ys[:, 0]**2, x) 
    eigenfunctions.append(np.abs(ys[:, 0] / np.sqrt(norm)))
    plt.plot(x, ys[:, 0] / np.sqrt(norm), col[modes - 1])
    

plt.show()
ei = np.array(eigenfunctions)# Stack columns to form a matrix
A1 = ei.T
A2 = eigenvalues 