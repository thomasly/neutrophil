'''
Filename: ResNet.py
Python Version: 3.6.5
Project: Neutrophil Identifier
Author: Yang Liu
Created date: Sep 5, 2018 4:13 PM
-----
Last Modified: Oct 4, 2018 4:03 PM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

from keras.layers import Input, Add, Dense
from keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D
from keras.layers import MaxPooling2D, ReLU, PReLU, LeakyReLU
from keras.models import Model
from keras.initializers import glorot_uniform
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the \
    main path
    filters -- python list of integers, defining the number of filters in \
    the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position \
    in the network
    block -- string/character, used to name the layers, depending on \
    their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the
    # main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(
        filters=F1, kernel_size=(1, 1),
        strides=(1, 1), padding='valid', name=conv_name_base + '2a',
        kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = PReLU()(X)

    # Second component of main path
    X = Conv2D(
        F2, (f, f), strides=(1, 1), padding="same", name=conv_name_base+"2b",
        kernel_initializer=glorot_uniform(seed=2))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = PReLU()(X)

    # Third component of main path
    X = Conv2D(
        F3, (1, 1), strides=(1, 1), padding="valid",
        name=conv_name_base + "2c",
        kernel_initializer=glorot_uniform(seed=3))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # Final step: Add shortcut value to main path, and pass it through a
    # RELU activation
    X = Add()([X, X_shortcut])
    X = PReLU()(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for \
    the main path
    filters -- python list of integers, defining the number of filters \
    in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their \
    position in the network
    block -- string/character, used to name the layers, depending on \
    their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # MAIN PATH
    # First component of main path
    X = Conv2D(
        F1, (1, 1), strides=(s, s), padding="valid",
        name=conv_name_base + "2a", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = PReLU()(X)

    # Second component of main path
    X = Conv2D(
        F2, (f, f), strides=(1, 1), padding="same",
        name=conv_name_base + "2b", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = PReLU()(X)

    # Third component of main path
    X = Conv2D(
        F3, (1, 1), strides=(1, 1), padding="valid",
        name=conv_name_base + "2c", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    # SHORTCUT PATH
    X_shortcut = Conv2D(
        F3, (1, 1), strides=(s, s), padding="valid",
        name=conv_name_base + "1",
        kernel_initializer=glorot_uniform())(X_shortcut)
    X_shortcut = BatchNormalization(
        axis=3, name=bn_name_base + "1")(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a
    # RELU activation
    X = Add()([X, X_shortcut])
    X = PReLU()(X)

    return X


def ResNet50(input_shape=(299, 299, 3), classes=2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 \
    -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Maxpooling added by Yang Liu
    # X = MaxPooling2D((3, 3), strides=(2, 2))(X_input)
    
    # 299 x 299 x 3
    # Stage 1
    X = Conv2D(
        32, (3, 3), strides=(2, 2), name='conv1',
        kernel_initializer=glorot_uniform(),
        padding='valid')(X_input)
    X = PReLU()(X)
    
    # 149 x 149 x 32
    X = Conv2D(
        32, (3, 3), strides=(1, 1), name='conv2',
        kernel_initializer=glorot_uniform(),
        padding='valid')(X)
    X = PReLU()(X)

    # 147 x 147 x 32
    X = Conv2D(
        64, (3, 3), strides=(1, 1), name='conv3',
        kernel_initializer=glorot_uniform(),
        padding='valid')(X)
    X = PReLU()(X)
    
    # 145 x 145 x 64
    X = Conv2D(
        64, (3, 3), strides=(2, 2), name='conv4',
        kernel_initializer=glorot_uniform(),
        padding='valid')(X)
    X = PReLU()(X)
    
    # 72 x 72 x 64
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    
    # 36 x 36 x 64
    # Stage 2
    X = convolutional_block(
        X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    # 36 x 36 x 256
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # 36 x 36 x 256
    # Stage 3
    X = convolutional_block(
        X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    # 18 x 18 x 512
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')
    
    # 18 x 18 x 512
    # Stage 4
    X = convolutional_block(
        X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    # 9 x 9 x 1024
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    X = MaxPooling2D(
        (3, 3), strides=(2, 2), padding='valid', name="max_pool")(X)
    
    # 4 x 4 x 1024
    # Stage 5
    X = convolutional_block(
        X, f=1, filters=[512, 512, 2048], stage=5, block='a', s=1)
    # 4 x 4 x 2048
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')
    
    # 4 x 4 x 2048
    # MaxPool
    X = MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same', name="max_pool_last")(X)

    # 2 x 2 x 2048
    # output layer
    X = Flatten()(X)
    X = Dense(
        classes,
        activation='softmax',
        name='fc' + str(classes),
        kernel_initializer=glorot_uniform())(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model
