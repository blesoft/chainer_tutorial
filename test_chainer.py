import chainer

#print(chainer.print_runtime_info())

from sklearn.datasets import load_iris
x,t = load_iris(return_X_y=True)

print('x=',x.shape)
print('t=',t.shape)
