from tensorflow import keras
def Generator (latent_dim):

    n_input_nodes = 8 * 8 * 128
    model = keras.Sequential(
    [
    # Layer 1
    keras.layers.Dense(n_input_nodes, input_dim=latent_dim, name='Generator-H-Layer-1'),
    keras.layers.Reshape((8, 8, 128), name='Generator-H-Layer-Reshape-1'),
    
    # Layer 2
    keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-H-Layer-2'),
    keras.layers.ReLU(name='Generator-Hidden-Layer-Activation-2'),
                              
    # Layer 3
    keras.layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-H-Layer-3'),
    keras.layers.ReLU(name='Generator-Hidden-Layer-Activation-3'),
    
    # Layer 4
    keras.layers.Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), padding='same', name='Generator-H-Layer-4'),
    keras.layers.ReLU(name='Generator-H-Layer-Activation-4'),
    
    # Output Layer
    keras.layers.Conv2D(filters=3, kernel_size=(5,5), activation='tanh', padding='same', name='Generator-Output-Layer'),
    ],name="Generator")
    return model
