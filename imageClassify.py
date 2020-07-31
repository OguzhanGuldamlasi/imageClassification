from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD, Adam

import matplotlib.pyplot as plt
import numpy as np
# plt.plot([1,10,30,50,100],[0.3084,0.3383, 0.3422,0.3436, 0.3338])
# plt.savefig("asd.png")
# plt.show()
from tensorflow.python.keras.layers import Dropout

num_channels = 3
img_rows = 32
img_cols = 32

batch_size = 64
num_epochs = 200
num_classes = 10
validation_split = 0.2
optim = SGD(learning_rate=0.001,momentum =0.9)

# VERİ SETİ YÜKLENİR:

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#data preprocessing starts here
# one hot encoding yapıyıyoruz.
#for example

# red,	green,	blue
# 1,		0,		0
# 0,		1,		0
# 0,		0,		1

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)
##dataları floata çevirip 0 ve 1 arasında değerlere map ediyoruz.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
##this math we did increases our loss and decreases accuracy
# mean = np.mean(X_train,axis=(0,1,2,3))
# std = np.std(X_train,axis=(0,1,2,3))
# X_train = (X_train-mean)/(std + 1e-7)
# X_test = (X_test-mean)/(std+1e-7)
# np.std(X_train,axis=(0,1,2,3))

model =Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', ))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))


model.compile(loss='categorical_crossentropy', optimizer=optim,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size,
                    epochs=num_epochs, validation_split=validation_split,
                    verbose=1)

score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])




#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("accuracy2.png")
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("loss2.png")
# plt.show()