import os
import librosa
import time
import torch
import utils.utils as utils
import numpy as np
import torch.nn as nn
from PIL import Image
from models import musiccnn
from torchvision import transforms as transforms
import torch.nn.functional as F


class MusicAnalysis():

    def __init__(self, weights_path):
        assert os.path.exists(weights_path)
        self.model = musiccnn.resnet18(weights_path)
        self.normalize = transforms.Normalize(mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([

            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])
        self.model.eval()

    def get_pairs(self, embeddings):
        left = list()
        right = list()
        for i in range(0, embeddings.shape[0]-1):
            for j in range(i+1, embeddings.shape[0]):
                left.append(embeddings[i])
                right.append(embeddings[j])
        left = torch.stack(left)
        right = torch.stack(right)
        pairs = [left, right]
        return pairs

    def process_music_chunks(self, path_to_wavs, series):
        """
        input:
            path_to_wavs: a list of paths to wav files, at least two
            series: int, the number of series. 1-8

        output:
            A dictionary containing file path as key:[predicted series, confidence]
            A dictionary containing pairs of wav as key:[sync/not_sync , confidence]
        """

        for i in range(10):
            start_time = time.time()
            print ("starting....")
            assert isinstance(series, int)
            assert series < 9 and series > 0
            assert len(path_to_wavs) >= 2
            tensor_images = list()
            music_images = list()
            for path in path_to_wavs:
                assert os.path.exists(path)
            for path in path_to_wavs:
                music_images.append(utils.wav_to_jpg(path))
            for image in music_images:
                tensor_images.append(self.preprocess(image))
            tensor_images = torch.stack(tensor_images)
            embeddings, predictions = self.model(
                [tensor_images], get_embedding=True)

            pairs = self.get_pairs(embeddings)
            sync_results = self.model(pairs, process_embedding=True)
            print (predictions.shape)
            prediction_confidence = F.softmax(predictions)
            predictions = torch.argmax(prediction_confidence, dim=1)
            print (predictions.shape)
            print ((sync_results > 0.5), predictions, prediction_confidence)
            print (time.time() - start_time)
            break
