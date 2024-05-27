
import os
import numpy as np
import librosa
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model

import sounddevice as sd


def split_data(features, labels, test_size=0.2, val_size=0.2, random_state=None):
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
        features (numpy array): Input features (e.g., MFCCs).
        labels (numpy array): Target labels.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.2).
        val_size (float): The proportion of the dataset to include in the validation split (default: 0.2).
        random_state (int or None): Seed for random number generator (default: None).
    
    Returns:
        X_train (numpy array): Training features.
        y_train (numpy array): Training labels.
        X_val (numpy array): Validation features.
        y_val (numpy array): Validation labels.
        X_test (numpy array): Test features.
        y_test (numpy array): Test labels.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    # Further split training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# Feature Extraction (MFCC)

def get_mfcc(base_dir="/Users/helvetica/SignalProcessing/data/one-shots/", audio_files=[]):
    mfccs_list = []
    labels = []

    for audio_file in audio_files:
        y, sr = librosa.load(base_dir + audio_file, sr=None)  # sr=None to keep the original sampling rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
        mfccs_list.append(mfccs.T)  # Transpose to match shape for split_data
        track_name = os.path.splitext(os.path.basename(audio_file))[0]  # Extract track name from file name
        labels += [track_name] * mfccs.shape[1]  # Add track name as label

    features = np.concatenate(mfccs_list, axis=0)
    labels = np.array(labels)

    return features, labels


# Data preparation

def get_data(features, labels):
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(features, labels)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Assuming 1 MFCC coefficient per time step
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    num_classes = len(label_encoder.classes_)

    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_val_categorical = to_categorical(y_val_encoded, num_classes=num_classes)

    return X_train, y_train_categorical, X_val, y_val_categorical, X_test, y_test, num_classes, label_encoder


 # Model architecture

def get_rnn(X_train, num_classes):
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)

    model = Sequential()
    model.add(LSTM(units=128))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_classes, activation='softmax'))

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, save_model=True):
    history = model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_val, y_val_categorical))

    if save_model:
        model.save("RNN_0.keras")
        np.save("RNN_0_label_encoder.npy", label_encoder.classes_)

    return history

def eval_model(X_test, y_test, num_classes):
    # Convert integer labels to one-hot encoded categorical labels for test data
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)

    # Evaluate model on test data
    loss, accuracy = model.evaluate(X_test, y_test_categorical)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# Predict on wav files

def preprocess_data(audio_file):
    mfccs_list = []
    y, sr = librosa.load(audio_file, sr=None)  # sr=None to keep the original sampling rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
    mfccs_list.append(mfccs.T)  # Transpose to match shape for model input

    preprocessed_data = np.concatenate(mfccs_list, axis=0)
    preprocessed_data = preprocessed_data.reshape(preprocessed_data.shape[0], preprocessed_data.shape[1], 1)  # Assuming 1 MFCC coefficient per time step

    return preprocessed_data

def predict_on_new_data(model, wav_file, label_encoder):
    # Perform prediction on the new data
    predictions = model.predict(wav_file)

    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Map predicted labels to label names
    predicted_label_names = label_encoder.inverse_transform(predicted_labels)

    # Count occurrences of each label
    label_counts = {}
    for label in predicted_label_names:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Find the most common label and its count
    most_common_label = max(label_counts, key=label_counts.get)
    most_common_count = label_counts[most_common_label]

    # Calculate the certainty (percentage of occurrences of the most frequent label)
    certainty = (most_common_count / len(predicted_label_names)) * 100

    return most_common_label, certainty





# Real time predictions

def preprocess_audio(audio_data, sr=44100, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Preprocesses audio data for prediction.

    Args:
    - audio_data: Audio data as numpy array
    - sr: Sampling rate (default: 44100 Hz)
    - n_mfcc: Number of MFCC coefficients to extract (default: 13)
    - hop_length: Hop length for MFCC calculation (default: 512)
    - n_fft: Number of FFT points for MFCC calculation (default: 2048)

    Returns:
    - preprocessed_data: Preprocessed audio data with shape (num_samples, num_timesteps, num_features)
    """
    # Initialize list to store preprocessed batches
    preprocessed_batches = []

    # Determine the number of samples and the size of each batch
    num_samples = len(audio_data)
    batch_size = 10000 
    
    # Process audio data in batches
    for i in range(0, num_samples, batch_size):
        # Extract MFCC features for the current batch
        mfccs = librosa.feature.mfcc(y=audio_data[i:i+batch_size], sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        mfccs = mfccs.T  # Transpose to match shape for model input
        preprocessed_batches.append(mfccs)

    # Concatenate preprocessed batches
    preprocessed_data = np.concatenate(preprocessed_batches, axis=0)

    # Reshape data to match model input shape
    preprocessed_data = preprocessed_data.reshape(preprocessed_data.shape[0], preprocessed_data.shape[1], 1)

    return preprocessed_data


# Function to make predictions on audio data
def predict_audio(model, audio_data, label_encoder):
    """
    Predicts labels for audio data using a trained model.

    Args:
    - model: Trained Keras model
    - audio_data: Audio data as numpy array
    - label_encoder: LabelEncoder object used for encoding labels during training

    Returns:
    - predicted_label: Predicted label for the audio data
    - predicted_label_name: Predicted label name for the audio data
    """
    # Preprocess audio data
    preprocessed_data = preprocess_audio(audio_data)

    # Make prediction
    prediction = model.predict(preprocessed_data)

    # Convert prediction to label
    predicted_label = np.argmax(prediction)

    # Print predicted label and classes in label encoder
    print("Predicted Label:", predicted_label)

    # Check if predicted label index is in the LabelEncoder's classes
    if predicted_label < len(label_encoder.classes_):
        # Map predicted label to label name
        predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
    else:
        # Handle previously unseen label
        predicted_label_name = "Unknown"

    return predicted_label, predicted_label_name

def stream_audio_and_predict(model, label_encoder, duration=10, samplerate=44100):
    print("Streaming audio...")

    def callback(indata, frames, time, status):
        if status:
            print(status)
        if any(indata):
            # Make prediction on the incoming audio data
            predicted_label, predicted_label_name = predict_audio(model, indata[:, 0], label_encoder)
            
            print(f"Predicted Label: {predicted_label}, Label Name: {predicted_label_name}")

    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate):
        sd.sleep(int(duration * 1000))



if __name__ == "__main__":
    train = 0

    if train:
        features, labels = get_mfcc(audio_files=['piano_c3.wav', 'riddim_screech.wav', 'violin_c4.wav'])
        X_train, y_train_categorical, X_val, y_val_categorical, X_test, y_test, num_classes, label_encoder = get_data(features, labels)

        model = get_rnn(X_train, num_classes)

        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical)

        eval_model(X_test, y_test, num_classes)
    
    else:
        model = load_model("RNN_0.keras")
        label_classes = np.load("RNN_0_label_encoder.npy")
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_classes

        stream_audio_and_predict(model, label_encoder)

    audio_files = ['/Users/helvetica/SignalProcessing/data/one-shots/piano_c3.wav',
                '/Users/helvetica/SignalProcessing/data/one-shots/riddim_screech.wav',
                '/Users/helvetica/SignalProcessing/data/one-shots/violin_tremolo_c4.wav']

    for wav in audio_files:
        preprocessed_data = preprocess_data(wav)
        predictions = predict_on_new_data(model, preprocessed_data, label_encoder)
        print(predictions)