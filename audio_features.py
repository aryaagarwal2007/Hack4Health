"""
Shared audio feature extraction.
Used by phase3b training, phase4 fusion, and the Streamlit app so that all
three consume an IDENTICAL feature vector for the audio classifier.
"""

import numpy as np
import librosa


def extract_features_from_waveform(y, sr=22050):
    """Extract the full feature vector from a waveform array."""
    feats = []

    # MFCCs: mean + var of coefficients, deltas, delta-deltas
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats += list(np.mean(mfccs, axis=1))
    feats += list(np.var(mfccs, axis=1))
    d_mfcc = librosa.feature.delta(mfccs)
    feats += list(np.mean(d_mfcc, axis=1))
    feats += list(np.var(d_mfcc, axis=1))
    dd_mfcc = librosa.feature.delta(mfccs, order=2)
    feats += list(np.mean(dd_mfcc, axis=1))
    feats += list(np.var(dd_mfcc, axis=1))

    # Pitch (YIN)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"),
                     fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 0:
        feats += [np.mean(f0_valid), np.std(f0_valid),
                  np.max(f0_valid) - np.min(f0_valid),
                  np.median(f0_valid)]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    feats += [np.mean(zcr), np.var(zcr)]

    # Spectral features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    feats += [np.mean(cent), np.var(cent),
              np.mean(rolloff), np.var(rolloff),
              np.mean(bandwidth), np.var(bandwidth),
              np.mean(flatness), np.var(flatness)]

    # RMS energy
    rms = librosa.feature.rms(y=y)
    feats += [np.mean(rms), np.var(rms)]

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += list(np.mean(chroma, axis=1))
    feats += list(np.var(chroma, axis=1))

    return np.array(feats, dtype=np.float64)


def extract_features(fpath):
    """Load a file and extract the feature vector."""
    y, sr = librosa.load(fpath, sr=22050, duration=5.0)
    return extract_features_from_waveform(y, sr)
