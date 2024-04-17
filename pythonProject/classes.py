import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
from collections import Counter
import os

# Load your trained model
model = keras.models.load_model("C:/Users/TANAY/Downloads/lstm_model.h5")

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["bhairav", "yaman", "malkans"])
class_names = ["bhairav", "yaman", "malkans"]


# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_histogram = np.sum(magnitudes, axis=1)
    return np.mean(mfccs, axis=1), np.mean(chroma, axis=1), pitch_histogram


def predict_raga(audio_file_path):
    # Extract features from the audio file
    mfccs, chroma, pitch_histogram = extract_features(audio_file_path)

    # Reshape features for model input
    features = np.concatenate([mfccs.reshape(1, -1), chroma.reshape(1, -1), pitch_histogram.reshape(1, -1)], axis=1)
    features = features.reshape((1, 1, features.shape[1]))

    # Make prediction
    prediction = model.predict(features)

    # Decode prediction
    predicted_label_index = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_label_index)[0]

    return predicted_label

