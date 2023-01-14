# Necessary Imports
import tensorflow as tf
import numpy as np 
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.utils import plot_model
from Generator import Generator
from Discriminator import Discriminator
from Training import train
from Load_images import Load_images
from GAN import GAN


# Check versions
print('tensorflow: %s' %tf.__version__)
print('numpy: %s' % np.__version__) 
print('sklearn: %s' % sklearn.__version__)

# Installs 
import cv2

import graphviz
# Visualization

import cv2 # 
print('OpenCV: %s' % cv2.__version__) 
import matplotlib 
import matplotlib.pyplot as plt 
print('matplotlib: %s' % matplotlib.__version__) 
import graphviz 
print('graphviz: %s' % graphviz.__version__) 




import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Assign working directory to a variable
working_dir=os.path.dirname(sys.path[0])


# Locate Images
ImgLocation=working_dir +'\Papers-Implementation\images\\'



# Choose Images Class 'the artist'
artist_class = set(["Edvard_Munch"])


# Store Img paths in a list
ImgPaths=[]
for artist in artist_class:
    for img in list(os.listdir(ImgLocation+artist)):
        ImgPaths=ImgPaths+[ImgLocation + artist + '\\'+ img]



# Load images 
# resize to 64 by 64
imgs_data=[]
for img in ImgPaths:
    img = cv2.imread(img)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs_resized = cv2.resize(img, (64, 64))
    imgs_data.append(imgs_resized)
    
# Create Numpy array of images standardized 
imgs_data = np.array(imgs_data, dtype="float") / 255.0

# Show data shape
print("Shape of data_lowres: Number of Samples is :", imgs_data.shape[0],".\nImages Shape is", imgs_data.shape[1:])


# Display 20 images from data
fig, axs = plt.subplots(5, 4, sharey=False, tight_layout=True, figsize=(20,20), facecolor='white')
n=0
for i in range(0,5):
    for j in range(0,4):
        axs[i,j].matshow(imgs_data[n])
        n=n+1
plt.show() 


# Instantiate Scaler
scaler=MinMaxScaler(feature_range=(-1, 1))

# Select images 
data=imgs_data.copy()
print("Original shape of the data: ", data.shape)

# Data must be reshaped ro one column to be scaled using MinMaxScaler
# Reshape array
data=data.reshape(-1, 1)
print(data)
print("Reshaped data: ", data.shape)

# Fit the scaler
scaler.fit(data)

# Scale the array  
data=scaler.transform(data)

    # Reshape back to the original shape
data=data.reshape(imgs_data.shape[0], 64, 64, 3)
print("Shape of the scaled array: ", data.shape)



# Latent Space or output space of generator
# Latent space is the space of points to feed the generator
latent_dim=200

# 
gen_model = Generator(latent_dim)

# Show Generator model summary and plot model diagram
gen_model.summary()

# Plot Generator Diagram
plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400)



# Instantiate Discriminator
dis_model = Discriminator()

# Show Discriminator model summary and plot its diagram
dis_model.summary()

plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400)

# Instantiate GAN model
gan_model = GAN(gen_model, dis_model)


# Train DCGAN model
train(gen_model, dis_model, gan_model, data, latent_dim)