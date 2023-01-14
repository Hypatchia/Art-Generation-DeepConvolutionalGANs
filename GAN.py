from tensorflow import keras
import Generator 
import Discriminator 
# Define GAN 
def GAN(Generator, Discriminator):

    GAN = keras.models.Sequential([Generator,Discriminator])

    # Deactive training for Discriminator 
    Discriminator.trainable = False

    # Compile GAN
    
    GAN.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    return GAN