import os
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

wav_dir = "data/wav-loops"
target_sr = 44100/2

def preprocess_wav(filename):
    y, sr = librosa.load(os.path.join(wav_dir, filename))

    if sr != target_sr:
        print(f"Input sample rate {sr} does not match target ({target_sr}).")

    y = librosa.util.normalize(y)

    return y

preprocessed_data = []
for filename in os.listdir(wav_dir):
    if filename.endswith(".wav"):
        preprocessed_data.append(preprocess_wav(filename))


latent_dim = 32 

class VAE(keras.Model):
    def __init__(self, audio_length):
        super(VAE, self).__init__()

        self.encoder = keras.Sequential([
            layers.Conv1D(32, kernel_size=3, strides=2, activation="relu", input_shape=(audio_length, 1)),
            layers.Conv1D(64, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim * 2),
        ])

        self.decoder = keras.Sequential([
            layers.Dense(units=audio_length * 16),
            layers.Reshape((audio_length, 16)),
            layers.Conv1DTranspose(32, kernel_size=3, strides=2, activation="relu"),
            layers.Conv1DTranspose(1, kernel_size=3, strides=2, activation="sigmoid"),
        ])

    def call(self, inputs):
        z_mean_var = self.encoder(inputs)

        z_mean, z_log_var = tf.split(z_mean_var, num_split=2, axis=1)
        epsilon = keras.backend.random_normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        reconstructed = self.decoder(z)

        return reconstructed

def vae_loss(y_true, y_pred):
    reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss, axis=1)

    total_loss = reconstruction_loss + kl_loss
    return total_loss


seconds = 4
audio_length = int(target_sr * seconds)

vae = VAE(audio_length)

vae.compile(optimizer="adam", loss=vae_loss)


def train_vae(vae, dataset, epochs, validation_split=0.1):
    if validation_split > 0:
        total_size = len(dataset)
        train_size = int(total_size * (1 - validation_split))
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
    else:
        train_dataset = dataset

    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = vae.fit(train_dataset, epochs=epochs, validation_data=val_dataset if validation_split > 0 else None, callbacks=[early_stopping])
    return history



def sample_and_save(vae, audio_length):

    z = tf.random.normal(shape=(1, latent_dim))
    z = tf.reshape(z, (1, audio_length, latent_dim))

    generated_audio = vae(z)[0, :, 0]  

    generated_audio = generated_audio.numpy()

    generated_audio = generated_audio.astype(np.float32)

    librosa.output.write_wav("new_sample.wav", generated_audio, target_sr)



sample_and_save(vae, audio_length)