import numpy as np
import pickle
import os
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    #simple training
    self.Xtr = X
    self.ytr = y
  def findMajority(self,array):
      arr=[0,0,0,0,0,0,0,0,0,0]
      for index in array:
          classY = self.ytr[index]
          arr[classY]=arr[classY]+1
      return arr.index(max(arr))

  def predict(self, X,k):
    num_test = X.shape[0]
    ##boş olan indexlere 0 ekleniyor ve dimension boyutunun train ile aynı olması sağlanıyor.
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    xrange=range
    # loop over all test samples
    for i in xrange(num_test):
      #en yakın komşu bulunuyor
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      min_Kindex= distances.argsort()[:k]
      # print(min_Kindex)
      Ypred[i]= self.findMajority(min_Kindex)
      print(Ypred[i])
      # Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
    return Ypred
def load_pickle(f):
        return  pickle.load(f, encoding='latin1')
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,3):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte



Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072(32x32x3 = 3072 and we have 50000 train samples)
# print(len(Xtr_rows[0]))
# exit()
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072 (10000 test samples)
nn = NearestNeighbor()
arr=[40]
nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
for elem in arr:
    Yte_predict = nn.predict(Xte_rows,elem)  # predict labels on the test images
    # number of examples that are correctly predicted (i
    print('" %f' % (np.mean(Yte_predict == Yte)))
