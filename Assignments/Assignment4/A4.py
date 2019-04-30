from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K


def display_sample(num,X_train_gray):
    #Print the one-hot array of this sample's label 
    print(train_labels[num])  
    #Print the label converted back to a number
    label = train_labels[num].argmax(axis=0)
    #Reshape the 768 values to a 28x28 image
    image = X_train_gray[num].reshape([32,32])
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
    
########################################################################3
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  img = img.reshape([32,32])
  plt.imshow(img, cmap=plt.cm.binary)
  
  true_label = np.argmax(true_label)  
  predicted_label = np.argmax(predictions_array)
  
  print (true_label)
  print (predicted_label)
  
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("P={} {:2.0f}% (T={})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  
  predicted_label = np.argmax(predictions_array)
  true_label = np.argmax(true_label)    
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


##############################################################################3
class_names = ['0', '1', '2', '3', '4', 
               '5', '6', '7', '8', '9']




path='/home/mun/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment4/'

training_data = sio.loadmat(path+'train_32x32.mat')
print ( training_data) 
image_ind =6
x_train = training_data['X']
y_train = training_data['y']

# plt.imshow(x_train[:,:,:,image_ind])
# plt.show()


test_data= sio.loadmat(path+'test_32x32.mat')
x_test = test_data['X']
y_test = test_data['y']

print (y_train[image_ind])

print ( x_train.shape)
#====================================

# extract and reshape Xtrain and y train 
#( Num_observations, Dimentions , channels) 
X_train, y_train = x_train.transpose((3,0,1,2)), y_train[:,0]

print( X_train.shape)

# extract and reshape Xtest and Ytest
#( Num_observations, Dimentions , channels) 
X_test, y_test = x_test.transpose((3,0,1,2)), y_test[:,0]

print( X_test.shape)
print('')


    
##################################
#using validation set approachs
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=7, stratify = y_train)


y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

y_val[y_val == 10] = 0



#X_val =X_train
###########
# Convert all to  gray scall to save calcultaions 
X_train_gray= (np.dot(X_train, [0.2990, 0.5870, 0.1140])).astype('float32')
X_val_gray= (np.dot(X_val, [0.2990, 0.5870, 0.1140])).astype('float32')
X_test_gray= (np.dot(X_test, [0.2990, 0.5870, 0.1140])).astype('float32')


X_train_gray =X_train_gray/255
X_val_gray = X_val_gray/255
X_test_gray =X_test_gray/255

#input ("X_train_gray")
train_labels = keras.utils.to_categorical(y_train, 10)
test_labels = keras.utils.to_categorical(y_test, 10)
y_val_labels = keras.utils.to_categorical(y_val, 10)

display_sample(1234,X_train_gray)

if K.image_data_format() == 'channels_first':
    X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], 1, 32, 32)
    X_val_gray = X_val_gray.reshape(X_val_gray.shape[0], 1, 32, 32)
    X_test = X_test.reshape(X_test.shape[0], 1, 32, 32)
    input_shape = (1, 32, 32)
else:
    X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], 32, 32, 1)
    X_test_gray = X_test_gray.reshape(X_test.shape[0], 32, 32, 1)
    X_val_gray = X_val_gray.reshape(X_val_gray.shape[0], 32, 32, 1)
    input_shape = (32, 32, 1)
    

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape) )
                 

# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))


# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))

# Final categorization from 0-9 with softmax
model.add(Dense(10, activation='softmax'))

###################original#############


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train_gray, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(X_val_gray, y_val_labels))


score = model.evaluate(X_test_gray, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('Model.h5')

#########################################
predictions = model.predict(X_test_gray)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, X_test_gray)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  plt.xticks(range(10), class_names)

  
plt.show()
