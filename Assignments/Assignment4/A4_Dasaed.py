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
    


path='/home/mun/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment4/'

training_data = sio.loadmat(path+'train_32x32.mat')

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
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=7, stratify = y_train)

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
                 input_shape=input_shape))
# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))
# Another dropout
#model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
model.add(Dense(10, activation='softmax'))



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

model.save(path+'my_model.h5')












# 
# train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 32, 32)
# test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 32, 32)
# input_shape = (1, 32, 32)
# 
# # print ( "here")
# # print(X_train_gray.shape)
# # plt.imshow(X_train_gray[7,:,:])
# # plt.show()
# 
# 
# 
# train_greyscale = rgb2gray(X_train).astype(np.float32)
# test_greyscale = rgb2gray(X_test).astype(np.float32)
# val_greyscale = rgb2gray(X_val).astype(np.float32)
# 
# print ("before plot")
# plot_images(train_greyscale, y_train, 1, 10)
# 
# print ( train_greyscale.shape)
# # plt.imshow(train_greyscale[7,:,:])
# # plt.show()
# 
# 
# del X_train , X_val ,X_test
# 
# #############
# #Prepossing  normalize all values between [0,1]
# 
# 
# 
# ###################should change###########
# 
# # Calculate the mean on the training data
# train_mean = np.mean(train_greyscale, axis=0)
# 
# # Calculate the std on the training data
# train_std = np.std(train_greyscale, axis=0)
# 
# # Subtract it equally from all splits
# train_greyscale_norm = (train_greyscale - train_mean) / train_std
# test_greyscale_norm = (test_greyscale - train_mean)  / train_std
# val_greyscale_norm = (val_greyscale - train_mean) / train_std
# 
# 
# 
# enc = OneHotEncoder().fit(y_train.reshape(-1, 1))
# 
# # Transform the label values to a one-hot-encoding scheme
# y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
# y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
# y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
# 
# print("Training set", y_train.shape)
# print("Validation set", y_val.shape)
# print("Test set", y_test.shape)
# 
# 
# h5f = h5py.File(path+'SVHN_grey.h5', 'w')
# 
# # Store the datasets
# h5f.create_dataset('X_train', data=train_greyscale_norm)
# h5f.create_dataset('y_train', data=y_train)
# h5f.create_dataset('X_test', data=test_greyscale_norm)
# h5f.create_dataset('y_test', data=y_test)
# h5f.create_dataset('X_val', data=val_greyscale_norm)
# h5f.create_dataset('y_val', data=y_val)
# 
# # Close the file
# h5f.close()
# 
# 
# 
# 
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
# h5f = h5py.File('SVHN_grey.h5', 'r')
# 
# # Load the training, test and validation set
# X_train = h5f['X_train'][:]
# y_train = h5f['y_train'][:]
# X_test = h5f['X_test'][:]
# y_test = h5f['y_test'][:]
# X_val = h5f['X_val'][:]
# y_val = h5f['y_val'][:]
# 
# # Close this file
# h5f.close()
# 
# print('Training set', X_train.shape, y_train.shape)
# print('Validation set', X_val.shape, y_val.shape)
# print('Test set', X_test.shape, y_test.shape)
# 
# 
# 
# 
# 
# comp = 32*32
# tf.logging.set_verbosity(tf.logging.INFO)
# 
# # Our application logic will be added here
# x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1], name='Input_Data')
# y = tf.placeholder(tf.float32, shape = [None, 10], name='Input_Labels')
# y_cls = tf.argmax(y, 1)
# 
# discard_rate = tf.placeholder(tf.float32, name='Discard_rate')
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# 
# max_epochs = 2
# num_examples = X_train.shape[0]
# prepare_log_dir()
# 
# 
# 
# 
# 
