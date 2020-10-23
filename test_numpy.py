import numpy as np
import random
a = np.array([1,2,3])
print(a)
print(a.shape)
print(a.ndim)
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(b)
print('Shape',b.shape)
print('Rank',b.ndim)
print('Size',b.size)
a = np.zeros((3,3))
print(a)
b = np.ones((2,3))
print(b)
c = np.full((3,2),9)
print(c)
d = np.eye(5)
print(d)
e = np.random.rand(4,5)
print(e)
f = np.arange(3,10,1)
print(f)
val = e[0,1]
print(val)
center = e[1:3,1:4]
print(center)
print('Shape of e:',e.shape)
print('Shape of center:',center.shape)
e[1:3,1:4] = 0
print(e)