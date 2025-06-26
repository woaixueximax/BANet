
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, MaxPooling2D,Activation,Dropout
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D,Bidirectional,LSTM
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute,MultiHeadAttention
from tensorflow.keras.regularizers import L2
from tensorflow.keras import backend as K
import numpy as np



def mha_block(input_feature, key_dim=8, num_heads=16, dropout=0.5, vanilla=True):
    """Multi Head self Attention (MHA) block.

    Here we include two types of MHA blocks:
            The original multi-head self-attention as described in https://arxiv.org/abs/1706.03762
            The multi-head local self attention as described in https://arxiv.org/abs/2112.13492v1
    """
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(input_feature)


    x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)

    x = Dropout(0.3)(x)
    # Skip connection
    mha_feature = Add()([input_feature, x])

    return mha_feature


def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay=0.009, maxNorm=0.6, dropout=0.25):
    """ Conv_block_ from Ingolfsson et al 2020.
    See details at https://arxiv.org/abs/2006.00622

    The original code for this model is available at https://github.com/iis-eth-zurich/eeg-tcnet

        Notes
        -----
        Selected modules and structures, and made changes

        References
        ----------
        .. Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N.,
           Cavigelli, L., & Benini, L. (2020, October).
           Eeg-tcnet: An accurate temporal convolutional network
           for embedded motor-imagery brain–machine interfaces.
           In 2020 IEEE International Conference on Systems,
           Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
    """

    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    )(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, in_chans),
                             depth_multiplier=D,
                             data_format='channels_last',
                             )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                     padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

def TCN_block_Inception(input_layer, filters, dropout,input_dimension, depth, kernel_size,
               weightDecay=0.009, maxNorm=0.6, activation='relu'):
    """ TCN_block_Inception from Bai et al 2018
        Temporal Convolutional Network (TCN)

        Notes
        -----
        using different regularization methods
        mutli-scale decoding
    """

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   padding='causal')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   padding='causal')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)

    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1,
                      padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',

                       padding='causal', kernel_initializer='he_uniform')(block)

        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out

def  D1CNN(input,filters,drop_num=0.4):
    cnn = Conv1D(filters=filters,kernel_size=1,
                      kernel_regularizer=L2(0.01),
                      )(input)
    cnn = BatchNormalization()(cnn)
    cnn = mha_block(cnn)
    cnn = Dropout(drop_num)(cnn)
    return cnn

def Bridge_block(block1,bridge_number,eegn_dropout):
    block2 = block1
    for i in range(bridge_number):
        cnn1 = D1CNN(block1, 64)
        concat2 = Concatenate(axis=-1)([block2, cnn1])
        lstm1 = Bidirectional(LSTM(units=32, return_sequences=True, dropout=eegn_dropout, recurrent_dropout=0))(
            concat2)
        concat3 = Concatenate(axis=-1)([lstm1, cnn1])
        block1 = concat3
        block2 = lstm1
    # cnn1 = D1CNN(block1,64)
    concat = Concatenate(axis=-1)([block1, block2])
    drop = Dropout(eegn_dropout)(concat)
    return drop

def Bridge(n_classes, in_chans=22, in_samples=1125, n_windows=2, attention='mha',
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average',encoder = 'None',decoder ='LSTM',Bridge_number = 2):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)
    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                             kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                             weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                             in_chans=in_chans, dropout=eegn_dropout)

    block2 = Lambda(lambda x: x[:, :, -1, :])(block1)

    block2 = Bridge_block(block2, Bridge_number, eegn_dropout)

    block3_1 = TCN_block_Inception(input_layer=block2, input_dimension=128, depth=tcn_depth,
                                   kernel_size=tcn_kernelSize, filters=tcn_filters,
                                   weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                                   dropout=tcn_dropout, activation=tcn_activation)
    block3_1 = Lambda(lambda x: x[:, -1, :])(block3_1)
    block3_1 = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3_1)

    block3_2 = TCN_block_Inception(input_layer=block2, input_dimension=128, depth=tcn_depth,
                                   kernel_size=tcn_kernelSize * 2, filters=tcn_filters,
                                   weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                                   dropout=tcn_dropout, activation=tcn_activation)
    block3_2 = Lambda(lambda x: x[:, -1, :])(block3_2)
    block3_2 = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3_2)

    block3_3 = TCN_block_Inception(input_layer=block2, input_dimension=128, depth=tcn_depth,
                                   kernel_size=tcn_kernelSize * 3, filters=tcn_filters,
                                   weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                                   dropout=tcn_dropout, activation=tcn_activation)
    block3_3 = Lambda(lambda x: x[:, -1, :])(block3_3)
    block3_3 = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3_3)
    sw_concat = tf.keras.layers.Average()([block3_1, block3_2, block3_3])

    out = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=out)

