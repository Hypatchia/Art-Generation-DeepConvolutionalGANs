from tensorflow import keras
def Discriminator(input_shape=(64,64,3)):

    model = keras.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2, 2), padding='same', input_shape=input_shape, name='Discriminator-Input-Layer'),
        keras.layers.LeakyReLU(alpha=0.2, name='Discriminator-H-Layer1'),
        keras.layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2, 2), padding='same', name='Discriminator-H-Layer2'),
        keras.layers.LeakyReLU(alpha=0.2, name='Discriminator-H-Layer3'),
        keras.layers.Flatten(name='Discriminator-Flatten-Layer') ,
        keras.layers.Dropout(0.3, name='Discriminator-Dropout'), 
        keras.layers.Dense(1, activation='sigmoid', name='Discriminator-Output-Layer') ,
        ],name="Discriminator") 
    return model
    
   
