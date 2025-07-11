# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:06:48 2020

@author: KIIT
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from  tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from  tensorflow.keras.layers import Flatten
from  tensorflow.keras.layers import Dense
from  tensorflow.keras.layers import Input
from  tensorflow.keras.models import Model
from  tensorflow.keras.optimizers import Adam
from  tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from  tensorflow.keras.preprocessing.image import img_to_array
from  tensorflow.keras.preprocessing.image import load_img
from  tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
#load the images and initialize the calss images
print("Loading the images....")
img_Paths=list(paths.list_images("D:\\traing data set\\face-mask-detector\\dataset"))
img_data=[]
img_labels=[]

#looping over the directors
for img_path in img_Paths:
    # extracting lable from file name i.e masked or unmasked
    label=img_path.split(os.path.sep)[-2]
    # load and preprocess the image 
    img=load_img(img_path,target_size=(224,224))
    img=img_to_array(img)
    img=preprocess_input(img)
    # insert in the img_data and img_label
    img_data.append(img)
    img_labels.append(label)
    
img_data=np.array(img_data,dtype="float32")
img_lables=np.array(img_labels)
# performing  one-hot encoding on the img_labels
label_binarizer=LabelBinarizer()
img_labels=label_binarizer.fit_transform(img_labels)
img_labels=to_categorical(img_labels)
#creating tarining  and testing splits using 80% for training 
# and 20% for testing
(trainX, testX, trainY, testY)=train_test_split(img_data,img_labels,test_size=0.20, stratify=img_labels, random_state=25)

# creating training image generator fro data augmentation
augmentation=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

# loading the MobileNetV2 network

base_model=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

head_modle=base_model.output
head_modle=AveragePooling2D(pool_size=(7,7))(head_modle)
head_modle=Flatten(name="flatten")(head_modle)
head_modle=Dense(128,activation="relu")(head_modle)
head_modle=Dropout(0.5)(head_modle)
head_modle=Dense(2,activation="softmax")(head_modle)

model=Model(inputs=base_model.input,outputs=head_modle)

for layer in base_model.layers:
	layer.trainable=False
print("compiling the model.....")
option=Adam(lr=1e-4,decay=1e-4/20)
model.compile(loss="binary_crossentropy",optimizer=option,metrics=["accuracy"])
print("traing the head_modle.....")
H=model.fit(augmentation.flow(trainX,trainY,batch_size=32),steps_per_epoch=len(trainX) // 32,validation_data=(testX, testY),validation_steps=len(testX) // 32,epochs=20)
   
print("predicting the network.....")
pred=model.predict(testX,batch_size=32)
pred=np.argmax(pred,axis=1)

print(classification_report(testY.argmax(axis=1), pred,
	target_names=label_binarizer.classes_))
model.save("D:\\traing data set\\face-mask-detector\\mask_model",save_format="h5")
# visualizing the tarining loss and accuracy
N = 32
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("D:\\traing data set\\face-mask-detector\\traing_plot")



