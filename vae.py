import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from encoder import Encoder
from decoder import Decoder

class CustomLayer(Layer):
    def vae_loss(self, x, z_decoded, z_mu, z_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss
        recon_loss = tf.keras.metrics.binary_crossentropy(x, z_decoded)

        # KL divergence
        kl_loss = -3e-5 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=1)

        beta = 2.5
        return K.mean(recon_loss + beta * kl_loss)

    def call(self, inputs):
        x, z_decoded, z_mu, z_sigma = inputs
        loss = self.vae_loss(x, z_decoded, z_mu, z_sigma)
        self.add_loss(loss, inputs=inputs)
        return x

class VariationalAutoencoder:
    def __init__(self, input_shape, latent_dim, num_channels):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_vae()

    def build_encoder(self):
        return Encoder(self.input_shape, self.latent_dim)

    def build_decoder(self):
        return Decoder(self.latent_dim, self.encoder.conv_shape, self.num_channels)

    def build_vae(self):
        # Combine encoder and decoder
        input_shape = self.input_shape[-3:]
        input_img = Input(shape=input_shape, name='encoder_input')
        z_mu, z_sigma, z = self.encoder.model(input_img)
        z_decoded = self.decoder.model(z)

        # Create custom layer
        custom_layer = CustomLayer()
        y = custom_layer([input_img, z_decoded, z_mu, z_sigma])

        # Define VAE model
        vae = Model(input_img, y, name='vae')
        vae.compile(optimizer='adam', loss=None)

        return vae