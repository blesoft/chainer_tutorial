# データセットの準備
from sklearn.datasets import load_iris

x,t = load_iris(return_X_y=True)
x = x.astype('float32')
t = t.astype('int32')

# Tuple Dataset
from chainer.datasets import TupleDataset
dataset = TupleDataset(x,t)
# print(dataset[0])
# print(dataset[:2])
## データセット分割
from chainer.datasets import split_dataset_random
train_val,test = split_dataset_random(dataset,int(len(dataset) * 0.7),seed=0)
train,valid = split_dataset_random(train_val,int(len(train_val) * 0.7),seed=0)
## SerialIterator
from chainer.iterators import SerialIterator
train_iter = SerialIterator(train,batch_size=4,repeat=True,shuffle=True)
minibatch = train_iter.next()
print(minibatch)