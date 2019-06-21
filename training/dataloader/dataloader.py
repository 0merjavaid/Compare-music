from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
import numpy as np
import utils
from torch.utils import data
import glob
import logging
import sys
import os
import utils
from torchvision import transforms
import random
from imgaug import augmenters as iaa
import random
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logger = logging.getLogger('global')


class MUSIC_LOADER(data.Dataset):

    def __init__(self, data_path, format="*png", mode="Train", experiment_name=""):
        # self.logger = utils.utils.get_logger(
        #     "loggings", experiment_name+".global", "loader")
        self.data_path = data_path
        self.format = format
        self.mode = mode

        self.normalize = transforms.Normalize(mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([

            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])

        self.create_dataset()

    def get_meta(self):
        return 0

    def create_pairs(self, lis):
        pairs = list()
        for i in range(len(lis)-2):
            for j in range(i+1, len(lis)):
                pair = [lis[i], lis[j]]
                pairs.append(pair)

        pairs.append([lis[-2], lis[-1]])
        return pairs

    def create_dataset(self):

        phrases = os.listdir(self.data_path)
        self.music_pairs = list()
        for phrase in phrases:
            speed_dict = {"90": [], "100": [], "110": [],
                          "120": [], "130": [], "140": [], "150": []}
            files = glob.glob(os.path.join(
                self.data_path, phrase+"/"+self.format))

            files = (sorted(files))
            for file in files:
                speed = file.split(" ")[-1].split("-")[0]
                speed_dict[speed].append(file)

            for speed in speed_dict.values():
                self.music_pairs += (self.create_pairs(speed))

    def augmentation(self):
        aug = iaa.Sequential([
            # iaa.Scale((224, 224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 2.0))),
            iaa.Fliplr(0.5),

            iaa.Sometimes(0.5, iaa.Affine(

                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees

            )),

            # iaa.Sometimes(0.2,
            #               iaa.OneOf([iaa.Dropout(p=(0, 0.0071)),
            # iaa.CoarseDropout(0.041, size_percent=0.5)])),
            iaa.Sometimes(0.3, iaa.AddToHueAndSaturation(
                value=(-40, 40), per_channel=True)),
            iaa.Sometimes(0.3, iaa.PerspectiveTransform(
                scale=(0.061, 0.071))),
            iaa.Sometimes(0.3, iaa.Add((-10, 10), per_channel=0.5)),
            iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.007*255), per_channel=0.5)),

        ], random_order=True)
        return aug

    def __len__(self):
        return len(self.music_pairs)

    def __getitem__(self, index):
        pair = self.music_pairs[index]
        random.shuffle(pair)
        img1 = cv2.imread(pair[0])
        img2 = cv2.imread(pair[1])
        time_window = np.random.randint(350, 650)
        max_allowed_difference = np.random.randint(0, 10)
        max_x_start = min(img1.shape[1], img2.shape[
                          1]) - time_window - max_allowed_difference
        x_start = np.random.randint(0, max_x_start)
        img1 = img1[:, x_start - max_allowed_difference: x_start -
                    max_allowed_difference + time_window]
        img2 = img2[:, x_start: x_start+time_window]

        assert img1.shape == img2.shape
        images = [img1, img2]
        for i, image in enumerate(images):
            images[i] = self.preprocess(image)

        return images, [1, 1]
