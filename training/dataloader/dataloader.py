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
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logger = logging.getLogger('global')


class MUSIC_LOADER(data.Dataset):

    def __init__(self, data_path, format="*png", mode="Train", experiment_name=""):
        # self.logger = utils.utils.get_logger(
        #     "loggings", experiment_name+".global", "loader")
        self.data_path = data_path
        self.format = format
        self.mode = mode
        self.music_pairs = list()
        self.normalize = transforms.Normalize(mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([

            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])

        self.create_positive_dataset()
        self.create_negative_dataset()
        random.shuffle(self.music_pairs)

    def get_meta(self):
        return 0

    def create_pairs(self, lis):
        pairs = list()
        for i in range(len(lis)-2):
            for j in range(i+1, len(lis)):
                pair = [lis[i], lis[j], 1]
                pairs.append(pair)

        pairs.append([lis[-2], lis[-1], 1])
        return pairs

    def create_positive_dataset(self):

        phrases = os.listdir(self.data_path)
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

    def create_negative_dataset(self):
        total_positive_samples = len(self.music_pairs)
        negative_types = {0: "different_series",
                          2: "Different_speed", 3: "Different_time"}
        sample_per_type = int(total_positive_samples/len(negative_types))
        phrases = os.listdir(self.data_path)

        for key in negative_types.keys():
            max_reached = False
            samples_created = 0
            if key == 0:
                for phrase in phrases:
                    files = glob.glob(os.path.join(
                        self.data_path, phrase+"/"+self.format))
                    for file in files:
                        other_phrases = phrases.copy()
                        other_phrases.remove(phrase)
                        random_phrase = other_phrases[
                            np.random.randint(0, len(other_phrases))]
                        random_file = glob.glob(os.path.join(
                            self.data_path, random_phrase+"/"+self.format))
                        random_file = random_file[
                            np.random.randint(0, len(random_file))]
                        pair = [file, random_file, key]
                        self.music_pairs.append(pair)

                        samples_created += 1
                        if samples_created >= sample_per_type:
                            max_reached = True
                            break
                    if max_reached:
                        break

            elif key == 2:
                for phrase in phrases:
                    files = glob.glob(os.path.join(
                        self.data_path, phrase+"/"+self.format))
                    for file in files:
                        file_speed = int(file.split(" ")[-1].split("-")[0])
                        other_files = files.copy()
                        random.shuffle(other_files)
                        for other_file in other_files:
                            other_file_speed = int(other_file.split(
                                " ")[-1].split("-")[0])
                            if abs(other_file_speed - file_speed) >= 30:
                                pair = [file, other_file, key]

                                self.music_pairs.append(pair)
                                samples_created += 1
                        if samples_created >= sample_per_type:
                            max_reached = True
                            break
                    if max_reached:
                        break

            else:
                for phrase in phrases:
                    files = glob.glob(os.path.join(
                        self.data_path, phrase+"/"+self.format))
                    for file in files:
                        if int(file.split("_")[-1].split(".")[0]) < 800:
                            continue
                        file_speed = int(file.split(" ")[-1].split("-")[0])
                        other_files = files.copy()
                        random.shuffle(other_files)
                        for other_file in other_files:
                            other_file_speed = int(other_file.split(
                                " ")[-1].split("-")[0])
                            if abs(other_file_speed - file_speed) == 0 and \
                                    int(other_file.split("_")[-1].split(".")[0]) >= 800:
                                pair = [file, other_file, key]

                                self.music_pairs.append(pair)
                                samples_created += 1
                        if samples_created >= sample_per_type:
                            max_reached = True
                            break
                    if max_reached:
                        break
        print (len(self.music_pairs))

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
        pair_type = pair[-1]
        in_sync = 1 if pair_type == 1 else 0
        pair = pair[:-1]
        random.shuffle(pair)
        img1 = cv2.imread(pair[0])
        img2 = cv2.imread(pair[1])
        min_width = min(img1.shape[1], img2.shape[
            1])
        time_window = min(min_width, np.random.randint(350, 750))

        max_allowed_difference = np.random.randint(0, 3)
        if pair_type == 3:
            time_window = 500
            max_allowed_difference = np.random.randint(100, 300)
        max_x_start = max(
            max_allowed_difference, min_width - (time_window + max_allowed_difference))
        if time_window != min_width:
            x_start = np.random.randint(0, max_x_start)
            x_start = max(x_start, max_allowed_difference)

        else:
            x_start = 0
            max_allowed_difference = 0

        img1 = img1[:, x_start - max_allowed_difference: x_start -
                    max_allowed_difference + time_window]
        img2 = img2[:, x_start: x_start+time_window]

        assert np.abs(img1.shape[1] - img2.shape[1]) <= max_allowed_difference
        j_label = int(pair[0].split(" ")[0][-1])-1
        i_label = int(pair[1].split(" ")[0][-1])-1
        # print (j_label, i_label, in_sync, pair[
        #     0].split("/")[-1], pair[1].split("/")[-1], pair_type)
        # plt.imshow(cv2.applyColorMap(
        #     np.vstack((img1, img2)), cv2.COLORMAP_JET))
        # plt.show()
        img1 = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
        img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)

        images = [img1, img2]

        for i, image in enumerate(images):
            images[i] = self.preprocess(image)

        return images, [i_label, j_label], in_sync
