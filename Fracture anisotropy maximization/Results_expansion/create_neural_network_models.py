import numpy as np
import tensorflow as tf


## Using Model.summary() in the console is a good way to visualize the networks somewhat

class Autoencoder():
    def __init__(self, nelx, nely):
        self.nelx = int(nelx)
        self.nely = int(nely)
    
    def get_networks(self, encoding_dim):
        '''
        Create a convolutional autoencoder for signed distance field encoding.
    
        Parameters
        ----------
        nelx : int
            Number of horizontal dimensions in design space.
        nely : int
            Number of vertical dimensions in design space.
        encoding_dim : int
            Number of latent space dimensions.
    
        Returns
        -------
        Encoder_out : Model
            Convolutional outer part of the encoder.
        Encoder_in : Model
            Dense inner part of the encoder.
        Decoder_in : Model
            Dense inner part of the decoder.
        Decoder_out : Model
            Convolutional outer part of the decoder.
    
        '''
        ## Set network parameter like filter size and stride values:
        filter_size = 4
        stride = 2
        increase_in_channels = 2
        
        ## Determine the dimensionality of each layer of the convolutional network part
        x0 = self.nelx
        y0 = self.nely
        c0 = 1
        x1 = (x0 - filter_size) / stride + 1
        y1 = (y0 - filter_size) / stride + 1
        c1 = c0 * increase_in_channels
        dim1 = c1 * np.ceil(y1) * np.ceil(x1)
        depth = 0
        while np.ceil(min(x1,y1)) >= 1 and dim1 > encoding_dim:
            x0 = x1
            y0 = y1
            c0 = c1
            x1 = (x0-filter_size)/stride+1
            y1 = (y0-filter_size)/stride+1
            c1 = c0 * increase_in_channels
            dim1 = c1 * np.ceil(y1) * np.ceil(x1)
            depth = depth + 1
        shape = []
        max_channel = c0
        shape.append([int(np.ceil(y0)), int(np.ceil(x0)), int(c0)])
        for d in range(depth):
            [y0,x0,c0] = shape[-1]
            y1 = int((y0 - 1) * stride + filter_size)
            x1 = int((x0  -1) * stride + filter_size)
            c1 = int(c0 / increase_in_channels)
            shape.append([y1, x1, c1])
        size = int(np.product(shape[0]))
        sizem = int((size*encoding_dim) ** (0.5))
        for i in range(len(shape)-1):
            shape[i][2] = (max(shape[i][2], max_channel))
        
        
        
        ## Design convolutional outer part of the decoder
        latent_space_out= tf.keras.layers.Input(shape=(size,))
        decoded_out = tf.keras.layers.Reshape(shape[0])(latent_space_out)
        for i in range(depth):
            shape_out = shape[i+1]
            shape_in = shape[i]
            if i < depth - 1:
                decoded_out = tf.keras.layers.Conv2DTranspose(shape_out[2], (filter_size, filter_size), strides = stride, 
                                              activation='linear', padding='valid', output_padding=(0,0))(decoded_out)
                decoded_out = tf.keras.layers.LeakyReLU(alpha = 0.3)(decoded_out)
            else:
                decoded_out = tf.keras.layers.Conv2DTranspose(shape_out[2], (filter_size, filter_size), strides = stride, 
                                              activation='tanh', padding='valid', output_padding=(0,0))(decoded_out)
                decoded_out = tf.keras.layers.Conv2D(shape_out[2], (shape_out[0] - self.nely + 1, shape_out[1] - self.nelx + 1), 
                                     strides = 1, activation='linear', use_bias=False)(decoded_out)
                decoded_out = tf.keras.layers.Dense(1, activation='tanh', use_bias=False)(decoded_out)
        decoded_out = tf.keras.layers.Reshape((self.nely, self.nelx))(decoded_out)
        Decoder_out = tf.keras.models.Model(latent_space_out, decoded_out)
        
        
        ## Design convolutional outer part of the encoder
        design_space_out = tf.keras.layers.Input(shape=(self.nely, self.nelx))
        encoded_out = tf.keras.layers.Reshape((self.nely, self.nelx, 1))(design_space_out)
        for i in range(depth):
            shape_inner = shape[depth-i]
            shape_outer = shape[depth-i-1]
            if i == 0:
                encoded_out = tf.keras.layers.Conv2DTranspose(1, (shape_inner[0] - self.nely + 1, shape_inner[1] - self.nelx + 1), 
                                                              strides=1, activation='linear', padding='valid', output_padding=(0,0))(encoded_out)
                encoded_out = tf.keras.layers.LeakyReLU(alpha = 0.3)(encoded_out)
                encoded_out = tf.keras.layers.Conv2D(shape_outer[2], (filter_size, filter_size), strides=stride, 
                                     activation='linear')(encoded_out)
                encoded_out = tf.keras.layers.LeakyReLU(alpha = 0.3)(encoded_out)
            else:
                encoded_out = tf.keras.layers.Conv2D(shape_outer[2], (filter_size, filter_size), strides=stride, 
                                     activation='linear')(encoded_out)
                encoded_out = tf.keras.layers.LeakyReLU(alpha=0.3)(encoded_out)
        encoded_out = tf.keras.layers.Flatten()(encoded_out)
        Encoder_out = tf.keras.models.Model(design_space_out, encoded_out)
        
        
        ## Design dense inner part of the decoder
        latent_space_in = tf.keras.layers.Input(shape=(encoding_dim,))
        decoded_in = tf.keras.layers.Dense(sizem,activation='linear')(latent_space_in) 
        decoded_in = tf.keras.layers.LeakyReLU(alpha=0.3)(decoded_in)
        decoded_in = tf.keras.layers.Dense(size,activation='linear')(decoded_in)
        decoded_in = tf.keras.layers.LeakyReLU(alpha=0.3)(decoded_in)
        Decoder_in = tf.keras.models.Model(latent_space_in,decoded_in)
        
        
        
        ## Design dense inner part of the encoder
        design_space_in = tf.keras.layers.Input((size,))
        encoded_in = tf.keras.layers.Dense(sizem, activation='linear')(design_space_in)
        encoded_in = tf.keras.layers.LeakyReLU(alpha=0.3)(encoded_in)
        encoded_in = tf.keras.layers.Dense(encoding_dim, activation='sigmoid')(encoded_in)
        Encoder_in = tf.keras.models.Model(design_space_in,encoded_in,name='Encoder')
        
        EL = [Encoder_out, Encoder_in]
        DL = [Decoder_out, Decoder_in]
        
        return EL, DL 
        
        
    def create_discriminator(encoding_dim):
        '''
        Create a network that maps from latent space to an scalar [0,1].
        This can be used as a discriminator netowork or surrogate model network.
    
        Parameters
        ----------
        encoding_dim : int
            Number of latent space dimensions.
    
        Returns
        -------
        Discriminator : Model
            Dense neural network.
    
        '''
        latent_space = tf.keras.layers.Input((encoding_dim,))
        dis = tf.keras.layers.Dense(50,activation='tanh')(latent_space)
        dis = tf.keras.layers.Dense(50,activation='tanh')(dis)
        dis = tf.keras.layers.Dense(50,activation='tanh')(dis)
        dis = tf.keras.layers.Dense(50,activation='tanh')(dis)
        dis = tf.keras.layers.Dense(50,activation='tanh')(dis)
        dis = tf.keras.layers.Dense(1,activation='sigmoid')(dis)
        Discriminator = tf.keras.models.Model(latent_space,dis)
        return Discriminator

