import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten

def preprocess_wav(wav_path):

    audio_bytes = tf.io.read_file(wav_path)

    audio, _ = tf.audio.decode_wav(audio_bytes, desired_channels=2)

    audio = tf.cast(audio, tf.float32)

    audio = audio / (2**15 - 1)  # 16-bit

    return audio


def load_wavs(data_dir, num_samples=4):

    wav_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]
    wav_paths = wav_paths[:num_samples]
    audio_batch = []
    for wav_path in wav_paths:
        audio = preprocess_wav(wav_path)

        audio = tf.expand_dims(audio, axis=0)
        audio_batch.append(audio)

    return tf.concat(audio_batch, axis=0)

model = tf.keras.Sequential([
LSTM(256, return_sequences=True, input_shape=(None, 88200, 2)),
LSTM(128),
Dense(88200 * 2, activation='tanh')
])

model.compile(loss='mse', optimizer='adam')
print(model.summary())

data_dir = "data/wav-loops"
num_samples = 4

audio_batch = load_wavs(data_dir, num_samples)

model.train_on_batch(audio_batch, audio_batch)