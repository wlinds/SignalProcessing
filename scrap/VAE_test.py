import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import soundfile as sf

def preprocess_audio(audio_path, sample_rate=44100, duration=4):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration, mono=True)
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")
    
    if sr != sample_rate:
        audio = librosa.resample(audio, sr, sample_rate)
        sr = sample_rate
    
    audio = librosa.util.normalize(audio)
    
    return audio

def create_x_train_from_folder(folder_path, sample_rate=44100, duration=4):
    x_train = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            audio_path = os.path.join(folder_path, file)
            audio = preprocess_audio(audio_path, sample_rate, duration)
            x_train.append(audio)
    x_train = np.array(x_train)
    return x_train

folder_path = 'data/wav-loops'
x_train = create_x_train_from_folder(folder_path)


latent_dim = 128

original_dim = max(len(audio) for audio in x_train)


# Encoder
encoder_inputs = keras.Input(shape=(original_dim,))
x = layers.Dense(512, activation='relu')(encoder_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

# Sampler
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(512, activation='relu')(decoder_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(decoder_inputs, outputs, name='decoder')

outputs = decoder(z)
vae = keras.Model(encoder_inputs, outputs, name='vae')

reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(kl_loss)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(x_train, epochs=200, batch_size=32)

num_samples = 10
sample_rate = 44100
latent_samples = np.random.normal(size=(num_samples, latent_dim))
synthetic_data = decoder.predict(latent_samples)

output_folder = 'synthetic_data'
os.makedirs(output_folder, exist_ok=True)

for i, sample in enumerate(synthetic_data):
    output_path = os.path.join(output_folder, f'synthetic_{i}.wav')
    sf.write(output_path, sample, sample_rate)