import numpy as np
A1 = [-1.6]
for j in range(100):
    f_xj = A1[j] * np.sin(3 * A1[j]) - np.exp(A1[j])
    f_der_xj = np.sin(3 * A1[j]) + 3 * A1[j] * np.cos(3 * A1[j]) - np.exp(A1[j])
    x_new = A1[j] - f_xj / f_der_xj
    A1 = np.append(A1, x_new)
    fc=A1[j]*np.sin(3*A1[j])-np.exp(A1[j])
    if abs(fc) < 1e-6:
        break 
iter_1 = j

xr = -0.4
xl = -0.7
xbi = []

for j in range(100):
    xc = (xr + xl)/2
    xbi.append(xc) 
    fc = (xc) * np.sin(3*(xc)) - np.exp(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc
    if  abs(fc) < 1e-6:
        break
iter_2 = j
A2 = xbi

A3 = [iter_1 + 1 , iter_2 + 1]

A = np.array([[1, 2] , [-1, 1]])
B = np.array([[2, 0] , [0, 2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1 , 0])
y = np.array([0 , 1])
z = np.array([1, 2, -1])

A4 = A + B
A5 = 3*x - 4*y
A6 = A @ x
A7 = B @(x-y)
A8 = D@x
A9 = D@y+z
A10 = A@B
A11 = B @ C
A12 = C @ D
