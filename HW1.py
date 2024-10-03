import numpy as np
x = np.array([-1.6])
for j in range(100):
    f_xj = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
    f_der_xj = np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])
    x_new = x[j] - f_xj / f_der_xj
    x = np.append(x, x_new)
    fc=x[j]*np.sin(3*x[j])-np.exp(x[j])
    if abs(fc) < 1e-6:
        break 

xr = -0.4
xl = -0.7
mid = np.array([])

for j in range(100):
    xc = (xr + xl)/2
    mid = np.append(mid, xc) 
    fc = (xc) * np.sin(3*(xc)) - np.exp(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc
    if  abs(fc) < 1e-6:
        display(j)
        break
mid

A3 = np.array([9, 16])

A = np.array([[1, 2] , [-1, 1]])
B = np.array([[2, 0] , [0, 2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([[1],[0]])
y = np.array([[0],[1]])
z = np.array([[1],[2],[-1]])

A4 = A + B
A5 = 3*x - 4*y
A6 = A @ x
A7 = B @(x-y)
A8 = D@x
A9 = D@x+z
A10 = A@B
A11 = B @ C
A12 = C @ D