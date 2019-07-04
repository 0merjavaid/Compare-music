import librosa
import numpy as np
import cv2


def wav_to_jpg(wav_path):
    clip, sample_rate = librosa.load(wav_path, sr=None)
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude, stft_phase = librosa.magphase(stft)
    stft_magnitude_db = librosa.amplitude_to_db(
        stft_magnitude, ref=np.max)
    n_mels = 256

    fmin = 0
    fmax = sample_rate/2

    mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels, sr=sample_rate, power=1.0,
                                              fmin=fmin, fmax=fmax)

    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.abs(mel_spec_db)
    mel_spec_db = ((mel_spec_db/np.max(mel_spec_db))
                   * 255).astype("uint8")
    return cv2.applyColorMap(mel_spec_db, cv2.COLORMAP_JET)
