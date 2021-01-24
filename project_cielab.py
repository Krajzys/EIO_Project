import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer, UpSampling2D,  Conv2D, Dense
from tensorflow.keras.optimizers import Adam
from skimage.color import rgb2lab, lab2rgb
from matplotlib import pyplot as plt


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__(name='encoder')

        self.layer_1 = InputLayer((32, 32, 1))
        self.layer_2 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.layer_3 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.layer_4 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.layer_5 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.layer_6 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.layer_7 = Dense(256, activation='relu')
        self.layer_8 = Dense(256, activation='relu')
        self.layer_9 = Dense(256, activation='relu')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__(name='decoder')

        self.layer_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.layer_2 = UpSampling2D((2, 2))
        self.layer_3 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer_4 = UpSampling2D((2, 2))
        self.layer_5 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.layer_6 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.layer_7 = UpSampling2D((2, 2))
        self.layer_8 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.layer_9 = Conv2D(2, (3, 3), activation='tanh', padding='same')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__(name='autoencoder')

        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def main():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Load data
    (y_train, _), (y_test, _) = cifar100.load_data()
    y_train = y_train.astype('float32') / 255.
    y_test = y_test.astype('float32') / 255.
    x_train = tf.expand_dims(rgb2lab(y_train)[:, :, :, 0], axis=-1)
    y_train = rgb2lab(y_train)[:, :, :, 1:] / 128.
    x_test = tf.expand_dims(rgb2lab(y_test)[:, :, :, 0], axis=-1)
    y_test = rgb2lab(y_test)[:, :, :, 1:] / 128.

    # Print shapes of datasets
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Create, compile and train model
    model = Autoencoder()
    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))
    # model.save("autoencoder_cielab_imagenet")
    # model = tf.keras.models.load_model("autoencoder_cielab")

    for i in range(1):
        # Test model on examplary input
        y = model(x_test[i:i+1])

        # Plot examplary gray and colored image
        plt.subplot(131)
        plt.imshow(tf.squeeze(x_test[i]), cmap='gray')
        plt.title("Gray image")
        plt.subplot(132)
        img = np.zeros((32, 32, 3))
        img[:, :, 0] = tf.squeeze(x_test[i])
        img[:, :, 1:] = tf.squeeze(y * 128)
        plt.imshow(lab2rgb(img))
        plt.title("Colorized image")
        plt.subplot(133)
        img[:, :, 1:] = tf.squeeze(y_test[i] * 128)
        plt.imshow(lab2rgb(img))
        plt.title("Original image")
        plt.show()


if __name__ == '__main__':
    main()
