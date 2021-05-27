"""
from google.colab import drive
drive.mount('/content/gdrive')
import keras,os
import numpy as np

import tensorflow as tf
%load_ext tensorboard
!rm -rf ./logs/ 
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, MaxPool2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
#from SignatureDataGenerator import SignatureDataGenerator
import getpass as gp
import sys
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
import datetime
import cv2
import glob
from tensorboard import notebook
notebook.list()

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


#def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
#	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
#	squaredPreds = K.square(preds)
#	squaredMargin = K.square(K.maximum(margin - preds, 0))
#	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
#	return loss


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy_roc(predictions, labels):
  
  predictions = tf.cast(predictions, tf.float32)
  labels = tf.cast(labels, tf.float32)
	
  dmax = K.max(predictions)
  dmin = K.min(predictions)
  nsame = K.sum(labels == 1)
  ndiff = K.sum(labels == 0)

  step = 0.01
  max_acc = 0

  for d in K.arange(dmin, dmax + step, step):
    idx1 = predictions.ravel() <= d
    idx2 = predictions.ravel() > d

    tpr = float(K.sum(labels[idx1] == 1)) / nsame
    tnr = float(K.sum(labels[idx2] == 0)) / ndiff
    acc = 0.5 * (tpr + tnr)
		#       print ('ROC', acc, tpr, tnr)

    if (acc > max_acc):
      max_acc = acc

  return max_acc

def make_pairs():
  images_forged1 = []
  images_genuine1 = []
  images_forged2 = []
  images_genuine2 = []
  images_forged3 = []
  images_genuine3 = []
  images_forged4 = []
  images_genuine4 = []
  pairImages = []
  pairLabels = []
  numClasses = 2
  dirs1 = os.walk('/content/gdrive/My Drive/123/Genuine/Final/')
  dirs2 = os.walk('/content/gdrive/My Drive/123/Forged/Final')
  for root, dirs, files in dirs2:
    z = 0 
    for i in dirs:
      z += 1
      for file1 in glob.glob('/content/gdrive/My Drive/123/Forged/Final/' + i +'/'+ '*jpg'):
        img_forg = cv2.imread(file1)
        img_forg = cv2.cvtColor(img_forg, cv2.COLOR_RGB2GRAY)
        img_forg = cv2.resize(img_forg, (110, 110), interpolation=cv2.INTER_CUBIC)
        if z == 1 :
          images_forged1.append(img_forg.astype(np.float32))
        if z == 2:
          images_forged2.append(img_forg.astype(np.float32))
        if z == 3:
          images_forged3.append(img_forg.astype(np.float32))
        if z == 4 :
          images_forged4.append(img_forg.astype(np.float32))
        

  for root1, dirs_1, files1 in dirs1:
    z = 0
    for i in dirs_1:
      z += 1
      for file in glob.glob('/content/gdrive/My Drive/123/Genuine/Final/' + i +'/'+ '*jpg'):
        img_gen = cv2.imread(file)
        img_gen = cv2.cvtColor(img_gen, cv2.COLOR_RGB2GRAY)
        img_gen = cv2.resize(img_gen, (110, 110), interpolation=cv2.INTER_CUBIC)
        if z == 1:
          images_genuine1.append(img_gen.astype(np.float32))
        if z == 2:
          images_genuine2.append(img_gen.astype(np.float32))
        if z == 3:
          images_genuine3.append(img_gen.astype(np.float32))
        if z == 4:
          images_genuine4.append(img_gen.astype(np.float32))
        
    for h in range(len(images_genuine1)):
      for j in range(len(images_genuine1)):
        i = images_genuine1[h]
        i2 = images_genuine1[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged1)):
        i = images_genuine1[h]
        i2 = images_forged1[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

    for h in range(len(images_genuine2)):
      for j in range(len(images_genuine2)):
        i = images_genuine2[h]
        i2 = images_genuine2[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged2)):
        i = images_genuine2[h]
        i2 = images_forged2[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

    for h in range(len(images_genuine3)):
      for j in range(len(images_genuine3)):
        i = images_genuine3[h]
        i2 = images_genuine3[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged3)):
        i = images_genuine3[h]
        i2 = images_forged3[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

    for h in range(len(images_genuine4)):
      for j in range(len(images_genuine4)):
        i = images_genuine4[h]
        i2 = images_genuine4[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged4)):
        i = images_genuine4[h]
        i2 = images_forged4[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

  del images_forged1, images_genuine1, images_forged2, images_genuine2, images_forged3, images_genuine3, images_forged4, images_genuine4
  return np.array(pairImages), np.array(pairLabels)


pairImages, pairLabels = make_pairs()

input_shape = (110, 110, 1) # 155
left_input = Input(input_shape)
right_input = Input(input_shape)

model = Sequential()
model.add(Conv2D(input_shape=(110,110,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())

encoded_image_1 = model(left_input)
encoded_image_2 = model(right_input)

distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([encoded_image_1, encoded_image_2])
#prediction = Dense(units=1, activation='sigmoid')(distance)
model_final = Model(inputs=[left_input, right_input], outputs=distance)
model_final.summary()

#rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
#adadelta = Adadelta()
model_final.compile(loss=contrastive_loss, optimizer='adam')

#model_final.compile(loss=contrastive_loss, optimizer="adam", metrics=["accuracy"])
log_dir = "/content/gdrive/My Drive/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

history = model_final.fit([pairImages[:, 0], pairImages[:, 1]], pairLabels[:].astype(np.float32), batch_size=60, epochs=10, callbacks=[tensorboard_callback])
%tensorboard --logdir "/content/gdrive/My Drive/logs/fit"

model_final.save("/content/gdrive/My Drive/signes1/Siames_Trained_Signet.h5")
model_final.save_weights("/content/gdrive/My Drive/signes1/Siames_Trained_Weights_Signet.h5")
"""
"""
from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
import keras,os
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
import datetime
import glob

def make_pairs():
  images_forged1 = []
  images_genuine1 = []
  images_forged2 = []
  images_genuine2 = []
  images_forged3 = []
  images_genuine3 = []
  images_forged4 = []
  images_genuine4 = []
  pairImages = []
  pairLabels = []
  numClasses = 2
  dirs1 = os.walk('/content/gdrive/My Drive/123/Genuine/Final/')
  dirs2 = os.walk('/content/gdrive/My Drive/123/Forged/Final')
  for root, dirs, files in dirs2:
    z = 0 
    for i in dirs:
      z += 1
      for file1 in glob.glob('/content/gdrive/My Drive/123/Forged/Final/' + i +'/'+ '*jpg'):
        img_forg = cv2.imread(file1)
        img_forg = cv2.cvtColor(img_forg, cv2.COLOR_RGB2GRAY)
        img_forg = cv2.resize(img_forg, (110, 110), interpolation=cv2.INTER_CUBIC)
        if z == 1 :
          images_forged1.append(img_forg.astype(np.float32))
        if z == 2:
          images_forged2.append(img_forg.astype(np.float32))
        if z == 3:
          images_forged3.append(img_forg.astype(np.float32))
        if z == 4 :
          images_forged4.append(img_forg.astype(np.float32))
        

  for root1, dirs_1, files1 in dirs1:
    z = 0
    for i in dirs_1:
      z += 1
      for file in glob.glob('/content/gdrive/My Drive/123/Genuine/Final/' + i +'/'+ '*jpg'):
        img_gen = cv2.imread(file)
        img_gen = cv2.cvtColor(img_gen, cv2.COLOR_RGB2GRAY)
        img_gen = cv2.resize(img_gen, (110, 110), interpolation=cv2.INTER_CUBIC)
        if z == 1:
          images_genuine1.append(img_gen.astype(np.float32))
        if z == 2:
          images_genuine2.append(img_gen.astype(np.float32))
        if z == 3:
          images_genuine3.append(img_gen.astype(np.float32))
        if z == 4:
          images_genuine4.append(img_gen.astype(np.float32))
        
    for h in range(len(images_genuine1)):
      for j in range(len(images_genuine1)):
        i = images_genuine1[h]
        i2 = images_genuine1[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged1)):
        i = images_genuine1[h]
        i2 = images_forged1[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

    for h in range(len(images_genuine2)):
      for j in range(len(images_genuine2)):
        i = images_genuine2[h]
        i2 = images_genuine2[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged2)):
        i = images_genuine2[h]
        i2 = images_forged2[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

    for h in range(len(images_genuine3)):
      for j in range(len(images_genuine3)):
        i = images_genuine3[h]
        i2 = images_genuine3[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged3)):
        i = images_genuine3[h]
        i2 = images_forged3[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

    for h in range(len(images_genuine4)):
      for j in range(len(images_genuine4)):
        i = images_genuine4[h]
        i2 = images_genuine4[j]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([1])
      for g in range(len(images_forged4)):
        i = images_genuine4[h]
        i2 = images_forged4[g]
        i = np.asarray(i)
        i2 = np.asarray(i2)
        pairImages.append([i, i2])
        pairLabels.append([0])

  del images_forged1, images_genuine1, images_forged2, images_genuine2, images_forged3, images_genuine3, images_forged4, images_genuine4
  return np.array(pairImages), np.array(pairLabels)


pairImages, pairLabels = make_pairs()

imge1 = cv2.imread('/content/drive/My Drive/123/Genuine/Final/Genuine_Irina/21.jpg', 0)
#imge2 = cv2.imread('/content/drive/My Drive/123/Genuine/Final/Genuine_Irina/25.jpg', 0)
imge2 = cv2.imread('/content/drive/My Drive/123/Forged/Final/Forged_Irina/1.jpg', 0)

imge1 = cv2.resize(imge1, (110, 110), interpolation=cv2.INTER_CUBIC)
imge2 = cv2.resize(imge2, (110, 110), interpolation=cv2.INTER_CUBIC)

image1 = imge1.reshape(110, 110, 1).astype(np.float32)
image2 = imge2.reshape(110, 110, 1).astype(np.float32)

gg = []
gg.append([image1, image2])
gg = np.array(gg)

print(image1.shape)

print(image2.shape)

model1 = keras.models.load_model("/content/drive/My Drive/signes1/Siames_Trained_Signet.h5", compile=False)

model1.summary()
pr = model1.predict([gg[:, 0], gg[:, 1]])
print(pr[0][0])

fig = plt.figure("Pair #{}".format(1), figsize=(4, 2))
plt.suptitle("Distance: {:.2f}".format(pr[0][0]))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(imge1)
plt.axis("off")

ax = fig.add_subplot(1, 2, 2)
plt.imshow(imge2)
plt.axis("off")
	# show the plot
plt.show()
"""