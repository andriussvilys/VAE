from load_mnist import MNISTDataLoader

mnist = MNISTDataLoader()

x_train, x_test = mnist.data
x_train.shape
x_test.shape
mnist.plot_samples()

from encoder import Encoder
# input_shape = x_train.shape 
input_shape = (None, 28, 28, 1)
latent_dim = 2
encoder = Encoder(input_shape=input_shape, latent_dim=latent_dim)

encoder.model.summary()
print(encoder.conv_shape)

from decoder import Decoder

decoder = Decoder(latent_dim=latent_dim, conv_shape=encoder.conv_shape, num_channels=1)

from vae import VariationalAutoencoder

vae = VariationalAutoencoder(latent_dim=latent_dim, input_shape=input_shape, num_channels=1)
vae.model.summary()

vae.model.fit(x_train, None, epochs=1, batch_size=32, validation_split=0.2)

import matplotlib.pyplot as plt
mu, _, _ = encoder.model.predict(x_test)
plt.figure(figsize=(10,10))
plt.scatter(mu[:,0], mu[:, 1], cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar()
plt.show()