import keras
from keras.layers import Input, Dense, Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K

class Decoder:
    def __init__(self, latent_dim, conv_shape, num_channels):
        self.latent_dim = latent_dim
        self.conv_shape = conv_shape
        self.num_channels = num_channels
        self.decoder = self.build_decoder()

    def build_decoder(self):
        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')
        x = Dense(self.conv_shape[1] * self.conv_shape[2] * self.conv_shape[3], activation='relu')(decoder_input)
        x = Reshape((self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]))(x)
        x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2DTranspose(self.num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

        decoder = Model(decoder_input, x, name='decoder')
        return decoder