import numpy as np
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import logging
import os
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def visualize_image(image, text=None, path=False):
    fig, ax = plt.subplots()
    print (path)
    if not path:
        if type(image) is not np.ndarray:
            image = np.asarray(image)
        image = cv2.resize(image, (100, 100))
        im = ax.imshow(image, interpolation='nearest', aspect='auto')
    else:
        assert os.path.exists(image)
        image = cv2.imread(image)
        if image.shape[0] > 1000:
            image = cv2.resize(image, (100, 100))
        im = ax.imshow(image, interpolation='nearest', aspect='auto')
        print ("showing image")
    if text is not None:
        if type(text) == list:
            text = " ".join(text)
        ax.annotate(text,
                    xy=(0.5, 0), xytext=(0, 10),
                    xycoords=('axes fraction', 'figure fraction'),
                    textcoords='offset points',
                    size=10, ha='center', va='bottom')
    plt.show()


def save_img(path, img, title, description):
    plt.title(title)
    plt.ylabel(description.keys(), fontsize=9)
    plt.xlabel(description.values(), fontsize=9)
    plt.imshow(img)
    plt.savefig(path)


def parse_file(path):
    dic = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            line = [i.strip() for i in line]
            dic[line[0]] = line[1:]
        f.close()
    return dic


def preprocess():
    normalize = transforms.Normalize(mean=[
        0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess


def parse_val_file(path):
    assert os.path.exists(path)
    with open(path, "r") as f:
        lines = f.readlines()
    return lines


def get_logger(directory, file_name, name, debug=True):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create directory to store logs
    print ("\n\n\n", directory, file_name, "\n")
    assert (os.path.exists(directory))
    save_dir = os.path.join(directory, file_name.split(".")[0])
    os.makedirs(os.path.join(
        directory, file_name.split(".")[0]), exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    print (file_path)
    # Text Logger
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    # Console Logger
    stream_handler = logging.StreamHandler()
    if debug:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print (unique_labels(y_true, y_pred), classes)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
