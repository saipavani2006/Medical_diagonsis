#Import libraries
import os
import tensorflow

#Import ImageDataGenerator class for image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Step 1: Define base dataset directory
#__file__ is the current script file
# os.path.abspath(...) gets the absolute path to the 'dataset' folder
base_dir=os.path.abspath(os.path.join(os.getcwd(),'dataset'))

#Step 2: Define sub-directories for training,validation,testing data
train_dir=os.path.join(base_dir,'train')
val_dir=os.path.join(base_dir,'val')
test_dir=os.path.join(base_dir,'test')

#Step 3: Create ImageDataGenerator for training data
#It rescales pixel values and copies data augmentation (rotation,flip)
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True
)

#Step 4: ImageDataGenerator for validation data
val_datagen=ImageDataGenerator(rescale=1./255)

#Step 5: ImageDataGenerator for test data
test_datagen=ImageDataGenerator(rescale=1./255)

#Step 6: Create data loaders from the folders using `.flow_from_directory()`
#This loads images in batches,assigns labels from loader names, and resizes them 
#loading training data
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

#loading validation data
val_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False                   #No need to shuffle data
)

#loading test data
test_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#Step 7: Fetch one batch of images and labels to check
images,labels=next(train_generator)

#Print the shape of loaded batch
print("Batch Shape: ",images.shape)
print("Labels Shape: ",labels.shape)