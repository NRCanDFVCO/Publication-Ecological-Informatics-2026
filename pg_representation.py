#Edited by Mohamed Elhefnawy

#All the used imports
import time
import zipfile
from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd
from google.colab import drive
import zipfile
from sklearn.cluster import MeanShift
import cv2
import tensorflow as tf
import keras
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from IPython.display import clear_output
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
from patsy.contrasts import ContrastMatrix
from scipy import stats
import gc

"""# Constructing polygons"""

# Function for constructing polygons
# Input: n (number of polygon sides) and side length of the regular polygon (based on the standard deviation of the data)
# Output: Polygon coordinates, midpoints of polygon and unit vectors of each polygon side
def PG_constuct(n, side_length):
  radius = side_length/(2*math.sin(math.pi/n))
  center = [0,0]
  coord = [[center[0],center[1]+radius]]
  for i in range(1,n):
      coord.append([center[0]+(radius*math.sin(i*2*math.pi/n)),center[1]+(radius*math.cos(i*2*math.pi/n))])
  vertices = coord
  coord.append(coord[0])
  coord = np.asarray(coord)
  mid_points = np.zeros((n,2))
  unit_vectors = np.zeros((n,2))
  for i in range(0,n):
    mid_points[i,:] = (coord[i,:]+coord[i+1,:])/2
    unit_vectors[i,:] = (coord[i+1,:]-coord[i,:])/np.linalg.norm(coord[i+1,:]-coord[i,:])
  return (coord,mid_points,unit_vectors)

"""# Generating Hamiltonian Cycles"""

# Function for generating hamiltonian cycles
# Input: n (number of polygon sides)
# Output: Hamiltonian Cycles Matrix
def hamCycleMat(n):
  if (n % 2 != 0):
    n1 = n-1
  else:
    n1 = n
  ham_cycles = np.zeros((int(n1/2),n1))
  ham_cycles[0,0] = 0;
  for j in range(1,n1):
    ham_cycles[0,j] = (ham_cycles[0,j-1]+(pow(-1,j+1)*j)) % n1
  for k in range(1,int(n1/2)):
    for i in range(0,n1):
      ham_cycles[k,i] = (ham_cycles[k-1,i]+1) % n1
  ham_cycles = ham_cycles + 1
  seq_path = np.array(list(range(1,ham_cycles.shape[1]+1)))
  seq_path = np.expand_dims(seq_path, axis=0)
  ham_cycles = np.concatenate((ham_cycles, seq_path), axis=0)
  # in case of odd vertices
  if (n % 2 != 0):
    ham_cycles = np.concatenate((ham_cycles, n*np.ones((int(n1/2) +1,1))),axis=1)
  # convert paths into cycles
  last_point = ham_cycles[:,0]
  last_point = np.expand_dims(last_point, axis=1)
  ham_cycles = np.concatenate((ham_cycles, last_point),axis=1)
  ham_cycles = ham_cycles.astype(int)
  return ham_cycles

"""# Calculating the polygon coordinates of the data"""

# Function for calculating the polygon coordinates of the data
# Input: midpoints of polygon, unit vectors of each polygon side and standardized data
# Output: Data coordinates on polygon
def Calc_PG_data_coord(mid_points,unit_vectors,norm_data):
  no_obs = norm_data.shape[0]
  no_var = norm_data.shape[1]
  data_coord = np.zeros((no_obs,2,no_var))
  for j in range(0,no_var):
    for i in range(0,no_obs):
      data_coord[i,:,j] = mid_points[j,:] + (unit_vectors[j,:]*norm_data[i,j])
  return data_coord

"""# Polygon representation as images


"""

# Polygon representation as images
# Input: Polygon vertices, Data coordinates on polygon, Hamiltonian Cycles Matrix, class labels, saving path, images_properties (fig_size and dpi)
# Output: Images of each observation with each hamiltonian cycle
def get_PG_images(coord, data_coord,ham_cycles,class_labels,saving_path,fig_size,dpi):
  for j in range(0,ham_cycles.shape[0]):
    for i in range(0,data_coord.shape[0]):
      line_points = np.zeros((len(ham_cycles[j,:]),data_coord.shape[1]))
      for k in range(0,len(ham_cycles[j,:])):
        line_points[k,:] = data_coord[i,:,ham_cycles[j,k]-1]
      title_obs = str(int(class_labels[i]))+'_'+str(i+1)+'_'+str(j+1)
      xs, ys = zip(*coord) # create lists of x and y values
      # resolution increases upon increasing dpi and fig_size (RAM is the constraint) if 'agg' backend used, It will take time but no memory leak
      # with 'agg' backend, 600 dpi and 10*10 fig size --> approx. 1 figure/ sec
      # when fig_size is smaller (5*5) with same dpi, approx --> 3 fig/second
      # if dpi = 200 with 5*5 fig_size, approx -->  approx --> 7 fig/second
      # Time depends on the fig_size, dpi, and the number of observations and variables
      fig = plt.figure(figsize=fig_size,clear=True)
      plt.plot(xs,ys,color='black')
      # Plotting ham cycles inside polygons
      plt.plot(line_points[:,0],line_points[:,1],color='black')
      plt.axis('off')
      # Saving the images
      fig.savefig(saving_path+title_obs+'.jpg', bbox_inches='tight', dpi=dpi, pad_inches=0)
      plt.figure().clear()
      plt.close()
      plt.cla()
      plt.clf()
      plt.close("all")
      del fig
      gc.collect()

"""#Reading and preprocessing the generated polygon images, indices, ham_cycles and classes"""

def read_preprocess_PG_images(saving_path,im_size):
  classes =[]
  imgs = []
  ham_cycle = []
  imgs_index = []
  folder_path = saving_path
  for imgname in os.listdir(folder_path):
    f = Image.open(os.path.join(folder_path, imgname))
    imgs_index.append(int(imgname.split('_')[1]))
    ham_cycle.append((imgname.split('_'))[2].split('.')[0])
    img_read = np.array(f)
    if img_read.ndim == 4 or img_read.ndim == 3:
      assert img_read.shape[-1] == 3, 'Channel size should be 3!'
    else:
      raise Exception('Wrong dimensions!')
    greyscale_img = (img_read[..., 0] * 0.299 + img_read[..., 1] * 0.587 + img_read[..., 2] * 0.114).astype(img_read.dtype)
    images_grey_resized = (np.array(Image.fromarray(greyscale_img).resize(im_size))/ 127.5 - 1).astype(greyscale_img.dtype)
    images_grey_resized = np.asarray(images_grey_resized)
    imgs.append(images_grey_resized)
    classes.append(int(imgname.split('_')[0]))
  PG_imgs = np.asarray(imgs)
  PG_classes = np.asarray(classes)
  PG_ham_cycle = np.asarray(ham_cycle)
  PG_ham_cycle = PG_ham_cycle.astype(int)
  PG_index = np.asarray(imgs_index)
  PG_index = PG_index.astype(int)
  return (PG_imgs,PG_classes,PG_ham_cycle,PG_index)

"""# Simple Encoding for Nominal features"""

from patsy.contrasts import ContrastMatrix
def _name_levels(prefix, levels):
  return ["[%s%s]" % (prefix, level) for level in levels]
class Simple(object):
  def _simple_contrast(self, levels):
    nlevels = len(levels)
    contr = -1./nlevels * np.ones((nlevels, nlevels-1))
    contr[1:][np.diag_indices(nlevels-1)] = (nlevels-1.)/nlevels
    return contr
  def code_with_intercept(self, levels):
    contrast = np.column_stack((np.ones(len(levels)),self._simple_contrast(levels)))
    return ContrastMatrix(contrast, _name_levels("Simp.", levels))
  def code_without_intercept(self, levels):
    contrast = self._simple_contrast(levels)
    return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))

"""# Dealing with Categorical features"""

def Preprocess_categorical_train(train_data,categorical_features_indices,categorical_features_types):
  #Preprocessing of categorical data
  count_cat_nom_features = 0
  for i in range(len(categorical_features_indices)):
    if (categorical_features_types[i] == 'N'):
      cat_feat_values = np.unique(train_data[:,categorical_features_indices[i]])
      count_cat_nom_features = count_cat_nom_features + len(cat_feat_values)-1
  cat_var_all = np.empty([train_data.shape[0], categorical_features_types.count('O')+count_cat_nom_features])
  count_feat = 0
  nominal_indices = []
  count_nominal_feature_values = []
  ordinal_indices = []
  map_matrices_nominal = []
  map_matrices_ordinal = []
  unique_nominal_values = []
  unique_ordinal_values = []
  ordinal_means = []
  ordinal_stds=[]
  for i in range(len(categorical_features_indices)):
    cat_feat_values = np.unique(train_data[:,categorical_features_indices[i]])
    if (categorical_features_types[i] == 'N'):
      nominal_indices.append(i)
      count_nominal_feature_values.append(len(cat_feat_values))
      #Simple encoding
      contrast = Simple().code_without_intercept(cat_feat_values)
      map_matrix = contrast.matrix
      map_matrices_nominal.append(map_matrix)
      unique_nominal_values.append(cat_feat_values)
      for j in range(len(train_data[:,categorical_features_indices[i]])):
        cat_var_all[j,count_feat:count_feat+len(cat_feat_values)-1] = map_matrix[np.where(cat_feat_values == train_data[j,categorical_features_indices[i]]),:]
      count_feat = count_feat + len(cat_feat_values)-1
    else:
      #Ordinal labeling
      ordinal_indices.append(i)
      unique_ord_values = np.unique(train_data[:,categorical_features_indices[i]])
      map_matrix = np.array(range(0,len(unique_ord_values)))
      map_matrices_ordinal.append(map_matrix)
      unique_ordinal_values.append(unique_ord_values)
      for j in range(len(train_data[:,categorical_features_indices[i]])):
        cat_var_all[j,count_feat] = map_matrix[np.where(unique_ord_values == train_data[j,categorical_features_indices[i]])]
      # cat_var_all[:,count_feat]= stats.zscore(cat_var_all[:,count_feat])
      ordinal_means.append(np.mean(cat_var_all[:,count_feat]))
      ordinal_stds.append(np.std(cat_var_all[:,count_feat]))
      cat_var_all[:,count_feat]= (cat_var_all[:,count_feat]-np.mean(cat_var_all[:,count_feat]))/np.std(cat_var_all[:,count_feat])
      count_feat = count_feat + 1
  return (cat_var_all,nominal_indices,count_nominal_feature_values,ordinal_indices,map_matrices_nominal,map_matrices_ordinal,unique_nominal_values,unique_ordinal_values,ordinal_means,ordinal_stds)

def Preprocess_categorical_test(test_data,categorical_features_indices,categorical_features_types,nominal_indices,count_nominal_feature_values,ordinal_indices,map_matrices_nominal,map_matrices_ordinal,unique_nominal_values,unique_ordinal_values,ordinal_means,ordinal_stds):
  #Preprocessing of categorical data
  cat_var_all = np.empty([test_data.shape[0], categorical_features_types.count('O')+sum(count_nominal_feature_values)-len(count_nominal_feature_values)])
  count_feat = 0
  for h in range(len(categorical_features_indices)):
      if h in nominal_indices:
          index  = nominal_indices.index(h)
          # Simple encoding
          for j in range(len(test_data[:,categorical_features_indices[h]])):
            cat_var_all[j,count_feat:count_feat+count_nominal_feature_values[index]-1] = map_matrices_nominal[index][np.where(unique_nominal_values[index] == test_data[j,categorical_features_indices[h]]),:]
          count_feat = count_feat + count_nominal_feature_values[index]-1
      else:
          index  = ordinal_indices.index(h)
          # Ordinal labeling
          for m in range(len(test_data[:,categorical_features_indices[h]])):
              cat_var_all[m,count_feat] = map_matrices_ordinal[index][np.where(unique_ordinal_values[index] == test_data[m,categorical_features_indices[h]])]
          cat_var_all[:,count_feat]= (cat_var_all[:,count_feat]-ordinal_means[index])/ordinal_stds[index]
          count_feat = count_feat + 1
  return (cat_var_all)

"""# Data Standardization"""

def Standardize_data_train(train_data,categorical_features_indices,categorical_features_if,cat_var_all):
  norm_numerical_data = np.array(np.delete(train_data,categorical_features_indices, axis=1))
  #standardization of numerical features
  train_means = np.mean(norm_numerical_data.astype(float),axis=0)
  train_stds = np.std(norm_numerical_data.astype(float),axis=0)
  norm_numerical_data = (norm_numerical_data.astype(float)- np.mean(norm_numerical_data.astype(float),axis=0))/np.std(norm_numerical_data.astype(float),axis=0)
  num_numerical_features = norm_numerical_data.shape[1]
  if (categorical_features_if=='Y'):
    #concatenating the standardized numerical features with processed categorical data
    norm_train_data = np.hstack((norm_numerical_data,cat_var_all))
  else:
    norm_train_data = norm_numerical_data
  return (norm_train_data,train_means,train_stds,num_numerical_features)

def Standardize_data_test(test_data,categorical_features_indices,categorical_features_if,cat_var_all,train_means,train_stds):
  norm_numerical_data = np.array(np.delete(test_data,categorical_features_indices, axis=1))
  norm_numerical_data = (norm_numerical_data.astype(float)- train_means)/train_stds
  num_numerical_features = norm_numerical_data.shape[1]
  if (categorical_features_if=='Y'):
    #concatenating the standardized numerical features with processed categorical data
    norm_test_data = np.hstack((norm_numerical_data,cat_var_all))
  else:
    norm_test_data = norm_numerical_data
  return (norm_test_data)

"""# Demo for loading data and PG representation"""

#Connecting to drive to link the data
from google.colab import drive
from numpy import genfromtxt
import pandas as pd
drive.mount('/content/gdrive',force_remount=True)
# saving path of polygon images
saving_train_path = '/content/gdrive/My Drive/PG_demo/demo_train_images8/'
saving_test_path = '/content/gdrive/My Drive/PG_demo/demo_train_images8/'
# demo for training and testing data
# train_data = genfromtxt('/content/gdrive/My Drive/PG_demo/iris_norm.csv', delimiter = ',')
# norm_train_data = np.array(train_data)
file_path = "/content/gdrive/My Drive/Harvester/Copy of training_df_irving_nov2022.csv"
wine_data = pd.read_csv(file_path, delimiter=",")
train_data = np.array(wine_data)
test_data = train_data[0:1000,:] # for testing purposes only

# Ask if there are categorical features or not, if yes, enter an array of their indices, then enter in order if they are nominal ('N') or ordinal ('O')
categorical_features_indices = []
categorical_features_types = []
categorical_features_if = input ("Does the data include categroical features ?, if yes, enter 'Y', if no, enter 'N'\n")
if (categorical_features_if=='Y'):
  categorical_features_indices = input ("Enter the indices of the categorical features starting from 0, (e.g. 0,1,3)\n")
  categorical_features_types = input ("Enter the type of the categorical features in the order of their indices, if nominal, enter (N), if ordinal, enter (O), (e.g. N,O,N)\n")
  categorical_features_types = categorical_features_types.split(',')
  categorical_features_indices = categorical_features_indices.split(',')
  for i in range(len(categorical_features_indices)):
    # convert each item to int type
    categorical_features_indices[i] = int(categorical_features_indices[i])

(cat_var_all_train,nominal_indices,count_nominal_feature_values,ordinal_indices,map_matrices_nominal,map_matrices_ordinal,unique_nominal_values,unique_ordinal_values,ordinal_means,ordinal_stds) = Preprocess_categorical_train(train_data,categorical_features_indices,categorical_features_types)

cat_var_all_test = Preprocess_categorical_test(test_data,categorical_features_indices,categorical_features_types,nominal_indices,count_nominal_feature_values,ordinal_indices,map_matrices_nominal,map_matrices_ordinal,unique_nominal_values,unique_ordinal_values,ordinal_means,ordinal_stds)

(norm_train_data,train_means,train_stds,num_numerical_features) = Standardize_data_train(train_data,categorical_features_indices,categorical_features_if,cat_var_all_train)

norm_test_data = Standardize_data_test(test_data,categorical_features_indices,categorical_features_if,cat_var_all_test,train_means,train_stds)

norm_train_data.shape[1]

train_data.shape[1]

# demo for training and testing class labels
class_labels = np.ones((norm_train_data.shape[0],1))
class_labels[1000:2000] = 2*class_labels[1000:2000]
class_labels[2000:4176] = 3*class_labels[2000:4176]
train_class_labels = class_labels
test_class_labels = class_labels # for testing purposes only

# number of features
n = norm_train_data.shape[1]
# Side_length of the regular polygon - twice the maximum value of standardized data
# side_length = 2*max(np.amax(norm_train_data),np.amax(norm_test_data))
side_length = 10
# Modify nominal features accoring to the side length
for i in range(len(nominal_indices)):
  norm_train_data[:,nominal_indices[i]+num_numerical_features:nominal_indices[i]+num_numerical_features+count_nominal_feature_values[i]-1] = norm_train_data[:,nominal_indices[i]+num_numerical_features:nominal_indices[i]+num_numerical_features+count_nominal_feature_values[i]-1] * (0.5*side_length)
  norm_test_data[:,nominal_indices[i]+num_numerical_features:nominal_indices[i]+num_numerical_features+count_nominal_feature_values[i]-1] = norm_test_data[:,nominal_indices[i]+num_numerical_features:nominal_indices[i]+num_numerical_features+count_nominal_feature_values[i]-1] * (0.5*side_length)
# Constructing the polygon  (coordinates, midpoints and unit vectors)
(coord,mid_points,unit_vectors) = PG_constuct(n, side_length)
# Generating Hamiltonian Cycle Matrix
ham_cycles = hamCycleMat(n)
# Calculating PG coordinates of training and testing data
train_data_coord = Calc_PG_data_coord(mid_points,unit_vectors,norm_train_data)
test_data_coord = Calc_PG_data_coord(mid_points,unit_vectors,norm_test_data)
# Parameters of PG images (fig_size and dpi)
fig_size = (3,3)
dpi = 200

saving_train_path = '/content/gdrive/My Drive/PG_demo/demo_train_images8/'
saving_test_path = '/content/gdrive/My Drive/PG_demo/demo_train_images8/'

# Generating the training PG images
get_PG_images(coord, train_data_coord[0:1,:,:],ham_cycles,train_class_labels[0:1,:],saving_train_path,fig_size,dpi)

# Generating the testing PG images
get_PG_images(coord, test_data_coord,ham_cycles,test_class_labels,saving_test_path,fig_size,dpi)

# image_size for CNN training
im_size = (64,64)
# Reading PG images from the folder path and preprocessing them for CNN training
(PG_train_imgs,PG_train_classes,PG_train_ham_cycle,PG_train_index) = read_preprocess_PG_images(saving_train_path,im_size)
(PG_test_imgs,PG_test_classes,PG_test_ham_cycle,PG_test_index) = read_preprocess_PG_images(saving_test_path,im_size)

"""# Training CNNs with PG images"""

#getting the train and val set, adding a new dimension for the CNN to work
train_imgs = PG_train_imgs[:,:,:,np.newaxis]
test_imgs = PG_test_imgs[:,:,:,np.newaxis]

#one-hot encode target column
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_encoder_train = LabelEncoder()
integer_encoded_train = label_encoder_train.fit_transform(PG_train_classes)
label_encoder_test = LabelEncoder()
integer_encoded_test = label_encoder_test.fit_transform(PG_test_classes)
print(integer_encoded_train)
print(integer_encoded_test)
# binary encode
onehot_encoder_train = OneHotEncoder(sparse=False)
integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
train_classes = onehot_encoder_train.fit_transform(integer_encoded_train)
print(train_classes)
onehot_encoder_test = OneHotEncoder(sparse=False)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
test_classes = onehot_encoder_test.fit_transform(integer_encoded_test)
print(test_classes)

# number of classes
No_classes = 3
# train_imgs = train_imgs/255
# test_imgs = test_imgs/255

label_encoder_train.classes_

np.unique(integer_encoded_train)

train_classes.shape

def cnn_create_model(activation='relu',kernel_size=3,filters = 8,pool_size=(2, 2),optimizer='adam'):
  #create model
  model = Sequential()
  #add model layers
  model.add(Conv2D(filters, kernel_size=kernel_size, activation=activation, input_shape=(train_imgs[1].shape[0],train_imgs[1].shape[1],1),padding='same'))
  model.add(Conv2D(filters, kernel_size=kernel_size, activation=activation,padding='same'))
  model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
  model.add(Conv2D(2*filters, kernel_size=kernel_size, activation=activation,padding='same'))
  model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
  model.add(Conv2D(4*filters, kernel_size=kernel_size, activation=activation,padding='same'))
  model.add(Conv2D(4*filters, kernel_size=kernel_size, activation=activation,padding='same'))
  model.add(Flatten())
  model.add(Dense(No_classes, activation='softmax'))
  #compile model using accuracy to measure model performance
  model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  #train the model
  #model.fit(train_imgs, train_classes, validation_data=(test_imgs, test_classes), epochs=100,batch_size=300,verbose=0)
  return model

# Implement a model with the best parameters
#cnn_model = vgg16_create_model()
#cnn_model.compile(optimizer= 'Adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
cnn_model = cnn_create_model(activation='relu',kernel_size=3,filters = 16,pool_size=(2, 2),optimizer='Adadelta')
cnn_model.fit(train_imgs, train_classes, epochs=15,batch_size=300,verbose=1)

#test_results = test_results.argmax(1)
# Testing the best model
test_results = cnn_model.predict(test_imgs,verbose=1)
test_accuracy = cnn_model.evaluate(test_imgs, test_classes, verbose = 1)[1]
test_results = test_results.argmax(1)
test_classes = test_classes.argmax(1)
# tn, fp, fn, tp = confusion_matrix (test_classes,test_results).ravel()
f1_score_new = f1_score(test_classes,test_results, average = None)
f1_score_new

No_ham_cycle = max(PG_test_ham_cycle)
No_test_obs = int(test_results.shape[0]/No_ham_cycle)
Majority_voting_results = np.zeros(No_test_obs)
True_results = np.zeros(No_test_obs)
count = 0
for i in range(1,No_test_obs+1):
  indices_index = np.where(PG_test_index == i)
  Majority_voting_results[count]= np.argmax(np.bincount(test_results[indices_index]))
  True_results[count] = np.argmax(np.bincount(test_classes[indices_index]))
  count = count + 1

f1_score_majority_voting = f1_score(True_results,Majority_voting_results, average = None)

f1_score_majority_voting
