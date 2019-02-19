import numpy as np

x1 = np.array([1,-1,1,-1,1,-1,1,-1])
x2 = np.array([1,1,1,1,-1,-1,-1,-1])
m1 = np.outer(x1,x1)
m2 = np.outer(x2,x2)
m3 = np.add(m1,m2)
print(m3)