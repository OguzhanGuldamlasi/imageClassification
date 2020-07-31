import pickle
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def get_test_features(batch):
    features = []
    labels = []

    for n in range(0, np.size(batch)):
        for row in range(0, np.shape(batch[b'data'])[0]):
            red = np.reshape(batch[b'data'][row][0:1024], (32, 32))
            blue = np.reshape(batch[b'data'][row][1024:2048], (32, 32))
            green = np.reshape(batch[b'data'][row][2048:3072], (32, 32))
            img = cv2.merge((blue, green, red))

            # Histogram feature vektörü
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, dst=None, dtype=cv2.CV_32F).flatten()

            # HOG feature vektörü
            hog_img = hog(img, feature_vector=True, multichannel=True)

            features.append(hist)  # histogram için hist, hog için hog_img parametre olarak verilir.

            labels.append(batch[b'labels'][row])

    return features, labels


def get_train_features(batch):
    features = []
    labels = []

    for n in range(0, np.size(batch)):
        for row in range(0, np.shape(batch[n][b'data'])[0]):
            red = np.reshape(batch[n][b'data'][row][0:1024], (32, 32))
            blue = np.reshape(batch[n][b'data'][row][1024:2048], (32, 32))
            green = np.reshape(batch[n][b'data'][row][2048:3072], (32, 32))
            img = cv2.merge((blue, green, red))

            # Histogram feature vektörü
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, dst=None, dtype=cv2.CV_32F).flatten()


            # HOG feature vektörü
            hog_img = hog(img, feature_vector=True, multichannel=True)
            features.append(hist)  # histogram için hist, hog için hog_img parametre olarak verilir.
            labels.append(batch[n][b'labels'][row])
    return features, labels


batch = []
batch.append(unpickle('cifar10/data_batch_1'))
batch.append(unpickle('cifar10/data_batch_2'))
batch.append(unpickle('cifar10/data_batch_3'))
batch.append(unpickle('cifar10/data_batch_4'))
batch.append(unpickle('cifar10/data_batch_5'))

train_features,train_labels=get_train_features(batch)

test_batch=unpickle('cifar10/test_batch')

test_features,test_labels=get_test_features(test_batch)
accuracy={}
kValues = [1,5,10,30,50,100]
for k in kValues:
   print("Started train and test")
   neigh=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
   neigh.fit(train_features,train_labels)
   acc=neigh.score(test_features,test_labels)
   accuracy[str(k)+'nearest-neighbor']=acc
   print(str(k)+'nearest-neighbor'+str(acc))

print(accuracy)