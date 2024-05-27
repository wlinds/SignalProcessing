import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_audio(audio):
    raw, sr = librosa.load(audio)
    return raw, sr 


def seconds_to_samples(sr, sec):
    return int(sr * sec)


def get_mfccs(audio, sr, n_mfcc=13, frame_length=0.025, hop_length=0.01):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, 
                                 n_fft=frame_length, hop_length=hop_length)

    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    return mfccs.T, mfccs_delta.T, mfccs_delta2.T

def estimate_pitch(audio, sr, hop_length=0.01):
    pitches, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                      fmin=librosa.note_to_hz('C2'), 
                                                      fmax=librosa.note_to_hz('C5'), 
                                                      hop_length=hop_length)
    return pitches, voiced_flag, voiced_probs


def mask_silence(arr, mask, target_val):
    a = arr.copy()
    a[~mask] = target_val
    return a


def estimate_formants_librosa(audio, sr, frame_length=0.025, hop_length=0.01, n_formants=5):
    num_frames = (len(audio) - frame_length) // hop_length + 1
    formants = np.zeros((num_frames, n_formants))

    for i, frame in enumerate(librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T):
        # Apply LPC
        lpc_coeffs = librosa.lpc(frame, order=2 + sr // 1000)
        # Calculate roots of the LPC polynomial
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angz * (sr / (2 * np.pi))
        freqs = np.sort(freqs)
        formants[i, :min(n_formants, len(freqs))] = freqs[:n_formants]  # Fill with formant frequencies

    return formants


def smooth_formants(formants, window_size=10):
    f1_smooth = np.convolve(formants[:, 0], np.ones(window_size)/window_size, mode='valid')
    f2_smooth = np.convolve(formants[:, 1], np.ones(window_size)/window_size, mode='valid')
    return f1_smooth, f2_smooth


def calculate_zcr(audio, sr, frame_length=0.025, hop_length=0.01):
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    return zcr.T

def calculate_sc(audio, sample_rate, frame_length=0.025, hop_length=0.01):
    sc = librosa.feature.spectral_centroid(y=audio, n_fft=frame_length, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, n_fft=frame_length, hop_length=hop_length)[0]

    return sc.T, rolloff.T


def extract_features(audio, sr, frame_length, hop_length, n_mfcc, n_formants):
    mfccs, mfccs_delta, mfccs_delta2 = get_mfccs(audio, sr, frame_length=frame_length, hop_length=hop_length)
    pitches, voiced_flag, voiced_probs = estimate_pitch(audio, sr, hop_length=hop_length)
    formants = estimate_formants_librosa(audio, sr, frame_length=frame_length, hop_length=hop_length, n_formants=n_formants)
    zcr = calculate_zcr(audio, sr, frame_length=frame_length, hop_length=hop_length)
    sc, rolloff = calculate_sc(audio, sr, frame_length=frame_length, hop_length=hop_length)

    masked_pitches = mask_silence(pitches, voiced_flag, target_val=0)
    
    if formants.shape[0] < mfccs.shape[0]:
        formants = np.pad(formants, ((0, mfccs.shape[0] - formants.shape[0]), (0, 0)), 'constant')
    
    features = {"MFCCs": mfccs,
            "Delta MFCCs": mfccs_delta,
            "Delta-Delta MFCCs": mfccs_delta2,
            "Pitch": masked_pitches[:, np.newaxis],
            "ZCR": zcr[:, np.newaxis],
            "Formants": formants,
            "Spectral Centroid": sc[:, np.newaxis],
            "Spectral Roll-off": rolloff[:, np.newaxis]
            }

    for name, val in features.items():
        print(f"{name:<{max(len(name) for name in features.keys())}} : {val.shape}")

    for name, val in features.items():
        if np.any(np.isnan(val)):
            print(f"\nNaN values present in the {name} array: ", end='')
            print(len(np.argwhere(np.isnan(val))))

            features.update({name:np.nan_to_num(val, nan=-1.0)})
            print(f"All NaN in {name} replaced with -1.0")
    

    combined_features = np.hstack([i for i in features.values()])

    return combined_features


def extract_speaker_embeddings(features):
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=19)  # You can adjust the number of components
    features_reduced = pca.fit_transform(features_scaled)

    # Apply KMeans for clustering
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(features_reduced)

    # Evaluate the clustering
    sil_score = silhouette_score(features_reduced, clusters)
    print(f'Silhouette Score: {sil_score}')

    # Visualize the results
    plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=clusters, cmap='viridis')
    plt.title('PCA-Reduced Feature Space with Cluster Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return clusters

if __name__ == "__main__":
    Y, sr = load_audio("data/speak/sample2_sveriges_radio-ok_boomer_20221207_0924400151.wav")

    Y = Y[:10_000]

    frame_length = seconds_to_samples(sr, 0.025)
    hop = seconds_to_samples(sr, 0.025)

    n_mfcc = 13
    n_formants = 5

    features = extract_features(Y, sr, frame_length, hop, n_mfcc, n_formants)

    print(features.shape)
    extract_speaker_embeddings(features)