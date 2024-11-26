import numpy as np
from scipy.sparse import spdiags
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time 
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

m = 300 # N value in x and y directions
n = m * m # total size of matrix
L = 20
dx = dy = L / m

e0 = np.zeros((n, 1)) # vector of zeros
e1 = np.ones((n, 1)) # vector of ones
e2 = np.copy(e1) # copy the one vector
e4 = np.copy(e0) # copy the zero vector
for j in range(1, m+1):
    e2[m*j-1] = 0 # overwrite every m^th value with zero
    e4[m*j-1] = 1 # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]
# Place diagonal elements

diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
e2.flatten(), -4 * e1.flatten(), e3.flatten(),
e4.flatten(), e1.flatten(), e1.flatten()]
offset = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
matA = spdiags(diagonals, offset, n, n) / (dx**2)#.tocsc()

#matA[0,0] = 2 / (dx**2)
# Modify the first diagonal directly
#matA.data[0][0] = 2 / (dx**2)

A = matA#.tocsc()

e1 = np.ones((n, 1))  
e2 = np.ones((n, 1))  
e3 = np.ones((n,1))  
e4 = np.ones((n, 1 ))  


diagonals_B = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_B = [-n+m, -m, m , n-m]  # Below and above neighbors (one step in y-direction)

matB = spdiags(diagonals_B, offsets_B, n, n) / ( 2 * dx )

B = matB

e1 = np.zeros((n, 1))  
e2 = np.ones((n, 1))  
e3 = np.copy(e2)  
e4 = np.copy(e1)  
for index in range(300):
    e1[300 * index]= 1
    e2[300 * index + 299] = 0
    e3[300 * index] = 0 
    e4[300 * index + 299]= 1


diagonals_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m+1, -1, 1 , m-1]  # Below and above neighbors (one step in y-direction)

matC = spdiags(diagonals_C, offsets_C, n, n) / ( 2 * dy)


C = matC

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2
import time


# Parameters for the simulation
nu = 0.001  # Viscosity coefficient
Lx = 30  # Length of the domain in the x direction
Ly = 30  # Length of the domain in the y direction
nx = 300  # Number of grid points in x
ny = 300  # Number of grid points in y
N = nx * ny  # Total number of grid points
t = np.arange(0, 20.5, 0.5)  # Time array

# Create the spatial grid
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Generate random vortices
num_vortices = 15  # You can increase this number
vortex_positions = np.random.uniform(-Lx/2, Lx/2, size=(num_vortices, 2))
vortex_strengths = np.random.uniform(5, 20, size=num_vortices)  # Random strengths
vortex_charges = np.random.choice([-1, 1], size=num_vortices)  # Random charges (positive or negative)
ellipticity_factors = np.random.uniform(0.5, 2, size=num_vortices)  # Random ellipticity factor

# Create the initial vorticity field
def initial_vorticity(nx, ny, X, Y, vortex_positions, vortex_strengths, vortex_charges, ellipticity_factors):
    w = np.zeros((nx, ny))
    
    # For each vortex, add a Gaussian centered at the vortex position
    for i in range(len(vortex_positions)):
        # Vortex properties
        x0, y0 = vortex_positions[i]
        strength = vortex_strengths[i]
        charge = vortex_charges[i]
        ellipticity = ellipticity_factors[i]
        
        # Gaussian function for the vortex
        X_shifted = X - x0
        Y_shifted = Y - y0
        
        # Apply ellipticity (stretch the Gaussian in one direction)
        gauss = charge * strength * np.exp(- (X_shifted**2 + (Y_shifted**2 / ellipticity**2)) / 2)
        
        w += gauss
    
    return w

# Initialize the vorticity field
w_init = initial_vorticity(nx, ny, X, Y, vortex_positions, vortex_strengths, vortex_charges, ellipticity_factors)
w2 = w_init.flatten()  # Flatten for use in the solver

# Set up the spatial grid for Fourier transforms
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# PDE system to solve
def pde(t, w2, nx, ny, A, B, C, K, nu):
    w = w2.reshape(nx, ny)
    wt = fft2(w)
    psit = -wt / K 
    psi = np.real(ifft2(psit)).reshape(nx * ny)
    psi_x = B @ psi 
    psi_y = C @ psi
    wx = B @ w2
    wy = C @ w2
    rhs = nu * A @ w2 + (wx * psi_y) - (psi_x * wy)
    return rhs

# Solve the PDE system
start_time = time.time()  # Record the start time
wsol = solve_ivp(pde, [t[0], t[-1]], w2, t_eval=t, args=(nx, ny, A, B, C, K, nu), method='RK45')
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Extract the solution
A1 = wsol.y

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(8, 6))

# Initial plot setup
im = ax.imshow(A1[:, 0].reshape(nx, ny), cmap='plasma', origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
ax.set_title(f"t = {t[0]:.2f}")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')

# Create the animation update function
def update(frame):
    w = A1[:, frame].reshape(nx, ny)
    im.set_array(w)
    ax.set_title(f"t = {t[frame]:.2f}")
    return [im]

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), interval=100, blit=True)

# Show the animation
plt.tight_layout()
plt.show()
