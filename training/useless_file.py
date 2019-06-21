import logging
import cv2
import numpy as np
import sys
from torch.utils import data
import matplotlib.pyplot as plt
# logger = logging.getLogger("trendage")
# logger.setLevel(logging.INFO)
from dataloader.dataloader import MUSIC_LOADER
dataset = MUSIC_LOADER("datasets/TrainI")
dl = data.DataLoader(dataset, batch_size=1)

for image, label in dl:
    print(image[1].shape)
    pass
    #     for img, label in zip(image, label):
    #         img = img.data.numpy()
    #         img = np.transpose(img, [1, 2, 0])
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         print (np.max(img), np.min(img), np.mean(img))

    #         print (label)
    #         cv2.imshow("", img)
    #         key = cv2.waitKey()
    #         if key == ord("q"):
    #             break
    #             sys.exit()

    #     if key == ord("q"):
    #         cv2.destroyAllWindows()
    #         sys.exit()
