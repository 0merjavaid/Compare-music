import os
import cv2
import glob
import librosa
import numpy as np


class MusicToImage:

    def __init__(self, m_path, i_path):
        self.m_path = m_path
        self.i_path = i_path

    def wav_ro_jpg(self, wav_path):
        clip, sample_rate = librosa.load(wav_path, sr=None)
        cropped = clip[clip != 0]
        clip = clip[0:cropped.shape[0]]
        clip = np.hstack((clip, clip))
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
        # return cv2.applyColorMap(mel_spec_db, cv2.COLORMAP_JET)
        return mel_spec_db

    def convert_all(self):
        assert os.path.exists(self.m_path)
        for folder in os.listdir(self.m_path):
            print (folder)
            folder = os.path.join(self.m_path, folder)
            wavs = glob.glob(folder + "/*wav")
            for wav in wavs:
                image = self.wav_ro_jpg(wav)
                width = str(image.shape[1])
                image_path = wav.replace(
                    "TrainM", "TrainI").replace(".wav", "_" + width + ".png")
                directory = "/".join(image_path.split("/")[:-1])
                os.makedirs(directory, exist_ok=True)
                cv2.imwrite(image_path, image)

if __name__ == "__main__":
    music_to_image = MusicToImage("datasets/TrainM", "datasets/TrainI")
    music_to_image.convert_all()
