# データセットの準備
from sklearn.datasets import load_iris

x,t = load_iris(return_X_y=True)
x = x.astype('float32')
t = t.astype('int32')

# Tuple Dataset
from chainer.datasets import TupleDataset
dataset = TupleDataset(x,t)
print(dataset[0])
print(dataset[:2])

