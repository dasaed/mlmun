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


import os
import time
# from __future__ import absolute_import
# from __future__ import print_function
from datetime import timedelta


import tensorflow as tf

import seaborn as sns

plt.rcParams['figure.figsize'] = (16.0, 4.0) # Set default figure size

def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Plot nrows * ncols images from images and annotate the images
    """
    # Initialize the subplotgrid
    fig, axes = plt.subplots(nrows, ncols)
    
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows*ncols)
    
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat): 
        
        # Predictions are not passed
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))
        
        # When predictions are passed, display labels + predictions
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])  
            
        # Display the image
        ax.imshow(images[i,:,:,0], cmap='binary')
        
        # Annotate the image
        ax.set_title(title)
        
        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])

def prepare_log_dir():
    '''Clears the log files then creates new directories to place
        the tensorbard log file.''' 
    if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
    tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)
    
def get_batch(X, y, batch_size=512):
    for i in np.arange(0, y.shape[0], batch_size):
        end = min(X.shape[0], i + batch_size)
        yield(X[i:end],y[i:end])   
         
def plot_images2(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])

def cnn_model_fn(features):
    """Model function for CNN."""
    
      # Input Layer
    input_layer = tf.reshape(features, [-1, 32, 32, 1], name='Reshaped_Input')

      # Convolutional Layer #1
    #with tf.name_scope('Conv1 Layer + ReLU'):
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

      # Pooling Layer #1
    #with tf.name_scope('Pool1 Layer'):
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

      # Convolutional Layer #2 and Pooling Layer #2
    #with tf.name_scope('Conv2 Layer + ReLU'): 
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        
    #with tf.name_scope('Pool2 Layer'):
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

      # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
         inputs=dense, rate=discard_rate)

      # Logits Layer
    #with tf.name_scope('Logits Layer'):
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)





############################################
path='/home/mun/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment4/'

train_data = sio.loadmat(path+'train_32x32.mat')

image_ind =6
x_train = train_data['X']
y_train = train_data['y']

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
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

##################################
#using validation set approachs
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=7, stratify = y_train)

############
# Convert all to  gray scall to save calcultaions 
# X_train_gray= (np.dot(X_train, [0.2990, 0.5870, 0.1140]))
# X_val_gray= (np.dot(X_val, [0.2990, 0.5870, 0.1140]))
# X_test_gray= (np.dot(X_test, [0.2990, 0.5870, 0.1140]))
# X_train_gray=X_train_gray.astype('float32')
# X_val_gray = X_val_gray.astype('float32')
# X_test_gray = X_test_gray.astype('float32')
# 
# X_train_gray =X_train_gray/255
# X_val_gray = X_val_gray/255
# X_test_gray =X_test_gray/255

# print ( "here")
# print(X_train_gray.shape)
# plt.imshow(X_train_gray[7,:,:])
# plt.show()



train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)
val_greyscale = rgb2gray(X_val).astype(np.float32)

print ("before plot")
plot_images(train_greyscale, y_train, 1, 10)

print ( train_greyscale.shape)
# plt.imshow(train_greyscale[7,:,:])
# plt.show()


del X_train , X_val ,X_test

#############
#Prepossing  normalize all values between [0,1]



###################should change###########

# Calculate the mean on the training data
train_mean = np.mean(train_greyscale, axis=0)

# Calculate the std on the training data
train_std = np.std(train_greyscale, axis=0)

# Subtract it equally from all splits
train_greyscale_norm = (train_greyscale - train_mean) / train_std
test_greyscale_norm = (test_greyscale - train_mean)  / train_std
val_greyscale_norm = (val_greyscale - train_mean) / train_std



enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

# Transform the label values to a one-hot-encoding scheme
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

print("Training set", y_train.shape)
print("Validation set", y_val.shape)
print("Test set", y_test.shape)


h5f = h5py.File(path+'SVHN_grey.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=train_greyscale_norm)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=test_greyscale_norm)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=val_greyscale_norm)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%


h5f = h5py.File('SVHN_grey.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

# Close this file
h5f.close()

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)





comp = 32*32
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1], name='Input_Data')
y = tf.placeholder(tf.float32, shape = [None, 10], name='Input_Labels')
y_cls = tf.argmax(y, 1)

discard_rate = tf.placeholder(tf.float32, name='Discard_rate')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

max_epochs = 2
num_examples = X_train.shape[0]
prepare_log_dir()





