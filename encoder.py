from keras.layers import Conv2D, Input, Flatten, Dense, Lambda
from keras.models import Model
from keras import backend as K

class Encoder:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.conv_shape = None
        self.model = self.build_encoder()

    def build_encoder(self):
        input_shape = self.input_shape[-3:]
        input_img = Input(shape=input_shape, name='encoder_input')
        # x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
        # x = Conv2D(64, 3, padding='same', activation='relu', strides=(2,2))(x)
        # x = Conv2D(64, 3, padding='same', activation='relu')(x)
        # x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(32, 15, padding='same', activation='relu')(input_img)
        x = Conv2D(128, 11, padding='same', activation='relu', strides=(2,2))(x)
        x = Conv2D(128, 9, padding='same', activation='relu')(x)
        x = Conv2D(128, 7, padding='same', activation='relu')(x)
        x = Conv2D(128, 9, padding='same', activation='relu')(x)
        x = Conv2D(128, 7, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)

        self.conv_shape = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)

        z_mu = Dense(self.latent_dim, name='latent_mu')(x)
        z_sigma = Dense(self.latent_dim, name='latent_sigma')(x)

        def sample_z(args):
            z_mu, z_sigma = args
            eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
            return z_mu + K.exp(z_sigma / 2) * eps

        z = Lambda(sample_z, output_shape=(self.latent_dim,), name='z')([z_mu, z_sigma])

        encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')
        return encoder