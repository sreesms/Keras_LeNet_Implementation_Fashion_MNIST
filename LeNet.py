import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import keras
from keras.datasets import fashion_mnist

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# plt.imshow(x_train[0])
# plt.show()

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

##padding the images by 2 pixels since LeNet Implementation is with input image of 32x32
x_train = np.pad(x_train,((0,0),(2,2),(2,2),(0,0)),'constant')
x_test = np.pad(x_test,((0,0),(2,2),(2,2),(0,0)),'constant')
# print(x_test.shape)

##standardization
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)
print(std_px)
x_train = (x_train - mean_px)/(std_px)

#one- hot encoding the labels
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

##LeNet Architecture
model = Sequential()

#Layer 1 - Convolution Layer
model.add(Conv2D(filters= 6,kernel_size =5,strides=1,activation = 'relu',input_shape = (32,32,1)))
model.add(AveragePooling2D(pool_size=2,strides=2))

##Layer 2 - Convolution Layer
model.add(Conv2D(filters=16,kernel_size = 5,strides=1,activation='relu',input_shape = (14,14,6)))
model.add(AveragePooling2D(pool_size=2,strides=2))
#Flattens the output of conv layer
model.add(Flatten())

##Layer 3 - Fully Connected Layer1
model.add(Dense(units=120,activation='relu'))

##Layer 4 - Fully Connected Layer2
model.add(Dense(units=84,activation='relu'))

##Output layer
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=15)

score = model.evaluate(x_test,y_test,batch_size=10)
print('Test Loss=',score)

#For the prediction of the input image
# predictions = model.predict(x_test)