import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Input, Reshape, Flatten, Conv2DTranspose, Layer, Multiply, Subtract, Lambda, Add, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K 
from keras.optimizers import Adam
from keras import initializers


## Using Model.summary() in the console is a good way to visualize the networks somewhat


def create_model_max(nelx,nely,encoding_dim):
    '''
    Create a convolutional autoencoder for density field encoding.

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
    filter_size=4
    filter_conv=3
    filter_max=2
    stride=2
    increase_in_channels=2
    
    ## Determine the dimensionality of each layer of the convolutional network part
    x0=nelx
    y0=nely
    c0=1
    x1=(x0-filter_size)/stride+1
    y1=(y0-filter_size)/stride+1
    c1=c0*increase_in_channels
    dim1=c1*np.ceil(y1)*np.ceil(x1)
    depth=0
    while np.ceil(min(x1,y1))>=1 and dim1>encoding_dim:
        x0=x1
        y0=y1
        c0=c1
        x1=(x0-filter_size)/stride+1
        y1=(y0-filter_size)/stride+1
        c1=c0*increase_in_channels
        dim1=c1*np.ceil(y1)*np.ceil(x1)
        depth=depth+1
    shape=[]
    max_channel=c0
    shape.append([int(np.ceil(y0)),int(np.ceil(x0)),int(c0)])
    for d in range(depth):
        [y0,x0,c0]=shape[-1]           
        y1=int((y0-1)*stride+filter_size)
        x1=int((x0-1)*stride+filter_size)
        c1=int(c0/increase_in_channels)
        shape.append([y1,x1,c1])
    size=int(np.product(shape[0]))
    sizem=int((size*encoding_dim)**(1/2))
    for i in range(len(shape)-1):
        shape[i][2]=(max(shape[i][2],max_channel))
        
    
    ## Design dense inner part of the decoder
    latent_space_in = Input(shape=(encoding_dim,))
    decoded_in = Dense(sizem,activation='linear')(latent_space_in) 
    decoded_in = LeakyReLU(alpha=0.3)(decoded_in)
    decoded_in = Dense(size,activation='linear')(decoded_in)
    decoded_in = LeakyReLU(alpha=0.3)(decoded_in)
    Decoder_in = Model(latent_space_in,decoded_in)
    
    ## Design convolutional outer part of the decoder
    latent_space_out = Input(shape=(size,))
    decoded_out = Reshape(shape[0])(latent_space_out)
    for i in range(depth):
        shape_out=shape[i+1]
        shape_in=shape[i]
        if i<depth-1:
            decoded_out = Conv2DTranspose(shape_out[2],(filter_size,filter_size), strides=stride,activation='linear', padding='valid', output_padding=(0,0))(decoded_out)
            decoded_out = LeakyReLU(alpha=0.3)(decoded_out)
        else:
            decoded_out = Conv2DTranspose(shape_out[2],(filter_size,filter_size), strides=stride,activation='linear', padding='valid', output_padding=(0,0))(decoded_out)
            decoded_out = LeakyReLU(alpha=0.3)(decoded_out)
            decoded_out = Conv2D(shape_out[2],(shape_out[0]-nely+1,shape_out[1]-nelx+1), strides=1,activation='linear')(decoded_out)
            decoded_out = Dense(1,activation='sigmoid',use_bias=False)(decoded_out)
    decoded_out=Reshape((nely,nelx))(decoded_out)
    Decoder_out = Model(latent_space_out,decoded_out)
    
    
    ## Design convolutional outer part of the encoder
    design_space_out =Input(shape=(nely,nelx))
    encoded_out=Reshape((nely,nelx,1))(design_space_out)
    for i in range(depth):
        shape_in=shape[depth-i]
        shape_out=shape[depth-i-1]
        if i==0:
            encoded_out = Conv2DTranspose(1,(shape_in[0]-nely+1,shape_in[1]-nelx+1), strides=1,activation='linear', padding='valid', output_padding=(0,0))(encoded_out)
            encoded_out = LeakyReLU(alpha=0.3)(encoded_out)
            encoded_out = Conv2D(shape_out[2], (filter_conv, filter_conv),strides=1, activation='linear')(encoded_out)
            encoded_out = LeakyReLU(alpha=0.3)(encoded_out)
            encoded_out = MaxPooling2D(pool_size=(filter_max, filter_max),strides=stride)(encoded_out)
        else:
            encoded_out = Conv2D(shape_out[2], (filter_conv, filter_conv),strides=1, activation='linear')(encoded_out)
            encoded_out = LeakyReLU(alpha=0.3)(encoded_out)
            encoded_out = MaxPooling2D(pool_size=(filter_max, filter_max),strides=stride)(encoded_out)
    encoded_out = Flatten()(encoded_out)
    Encoder_out = Model(design_space_out,encoded_out)
    
    ## Design dense inner part of encoder
    design_space_in = Input((size,))
    encoded_in = Dense(sizem, activation='linear')(design_space_in)
    encoded_in = LeakyReLU(alpha=0.3)(encoded_in)
    encoded_in = Dense(encoding_dim, activation='sigmoid')(encoded_in)
    Encoder_in=Model(design_space_in,encoded_in,name='Encoder')
    
 
    return Encoder_out, Encoder_in, Decoder_in, Decoder_out






def create_model_sdf(nelx,nely,encoding_dim):
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
    filter_size=4
    stride=2
    increase_in_channels=2
    
    ## Determine the dimensionality of each layer of the convolutional network part
    x0=nelx
    y0=nely
    c0=1
    x1=(x0-filter_size)/stride+1
    y1=(y0-filter_size)/stride+1
    c1=c0*increase_in_channels
    dim1=c1*np.ceil(y1)*np.ceil(x1)
    depth=0
    while np.ceil(min(x1,y1))>=1 and dim1>encoding_dim:
        x0=x1
        y0=y1
        c0=c1
        x1=(x0-filter_size)/stride+1
        y1=(y0-filter_size)/stride+1
        c1=c0*increase_in_channels
        dim1=c1*np.ceil(y1)*np.ceil(x1)
        depth=depth+1
    shape=[]
    max_channel=c0
    shape.append([int(np.ceil(y0)),int(np.ceil(x0)),int(c0)])
    for d in range(depth):
        [y0,x0,c0]=shape[-1]
        y1=int((y0-1)*stride+filter_size)
        x1=int((x0-1)*stride+filter_size)
        c1=int(c0/increase_in_channels)
        shape.append([y1,x1,c1])
    size=int(np.product(shape[0]))
    sizem=int((size*encoding_dim)**(1/2))
    for i in range(len(shape)-1):
        shape[i][2]=(max(shape[i][2],max_channel))
    
    
    ## Design dense inner part of the decoder
    latent_space_in = Input(shape=(encoding_dim,))
    decoded_in = Dense(sizem,activation='linear')(latent_space_in) 
    decoded_in = LeakyReLU(alpha=0.3)(decoded_in)
    decoded_in = Dense(size,activation='linear')(decoded_in)
    decoded_in = LeakyReLU(alpha=0.3)(decoded_in)
    Decoder_in = Model(latent_space_in,decoded_in)
    
    ## Design convolutional outer part of the decoder
    latent_space_out= Input(shape=(size,))
    decoded_out = Reshape(shape[0])(latent_space_out)
    for i in range(depth):
        shape_out=shape[i+1]
        shape_in=shape[i]
        if i<depth-1:
            decoded_out = Conv2DTranspose(shape_out[2],(filter_size,filter_size), strides=stride,activation='linear', padding='valid', output_padding=(0,0))(decoded_out)
            decoded_out = LeakyReLU(alpha=0.3)(decoded_out)
        else:
            decoded_out = Conv2DTranspose(shape_out[2],(filter_size,filter_size), strides=stride,activation='tanh', padding='valid', output_padding=(0,0))(decoded_out)
            decoded_out = Conv2D(shape_out[2],(shape_out[0]-nely+1,shape_out[1]-nelx+1), strides=1,activation='linear',  use_bias=False)(decoded_out)
            decoded_out = Dense(1,activation='linear',use_bias=False)(decoded_out)
    decoded_out=Reshape((nely,nelx))(decoded_out)
    Decoder_out = Model(latent_space_out,decoded_out)
    
    
    ## Design convolutional outer part of the encoder
    design_space_out =Input(shape=(nely,nelx))
    encoded_out=Reshape((nely,nelx,1))(design_space_out)
    for i in range(depth):
        shape_in=shape[depth-i]
        shape_out=shape[depth-i-1]
        if i==0:
            encoded_out = Conv2DTranspose(1,(shape_in[0]-nely+1,shape_in[1]-nelx+1), strides=1,activation='linear', padding='valid', output_padding=(0,0))(encoded_out)
            encoded_out = LeakyReLU(alpha=0.3)(encoded_out)
            encoded_out = Conv2D(shape_out[2], (filter_size, filter_size),strides=stride, activation='linear')(encoded_out)
            encoded_out = LeakyReLU(alpha=0.3)(encoded_out)
        else:
            encoded_out = Conv2D(shape_out[2], (filter_size, filter_size),strides=stride, activation='linear')(encoded_out)
            encoded_out = LeakyReLU(alpha=0.3)(encoded_out)
    encoded_out = Flatten()(encoded_out)
    Encoder_out=Model(design_space_out, encoded_out)
    
    ## Design dense inner part of the encoder
    design_space_in = Input((size,))
    encoded_in = Dense(sizem, activation='linear')(design_space_in)
    encoded_in = LeakyReLU(alpha=0.3)(encoded_in)
    encoded_in = Dense(encoding_dim, activation='sigmoid')(encoded_in)
    Encoder_in=Model(design_space_in,encoded_in,name='Encoder')
    
    
    return Encoder_out, Encoder_in, Decoder_in, Decoder_out 
    
    
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
    latent_space=Input((encoding_dim,))
    dis=Dense(50,activation='tanh')(latent_space)
    dis=Dense(50,activation='tanh')(dis)
    dis=Dense(50,activation='tanh')(dis)
    dis=Dense(50,activation='tanh')(dis)
    dis=Dense(50,activation='tanh')(dis)
    dis=Dense(1,activation='sigmoid')(dis)
    Discriminator=Model(latent_space,dis)
    return Discriminator

