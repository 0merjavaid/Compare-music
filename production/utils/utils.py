import logging
import os
import requests
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_logger(directory, file_name, name, debug=True):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create directory to store logs
    directory = os.path.join(directory, file_name.split(".")[0])
    os.makedirs(directory, exist_ok=True)
    assert (os.path.exists(directory))

    # Text Logger
    file_path = os.path.join(directory, file_name)
    print (file_path)
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


def save_img(path, img, title, description):
    plt.title(title)
    plt.ylabel(description.keys(), fontsize=9)
    plt.xlabel(description.values(), fontsize=9)
    plt.imshow(img)
    plt.savefig(path)


def download_img(save_dir, entry, index):
    os.makedirs(save_dir, exist_ok=True)
    file_save_path = os.path.join(
        save_dir, str(index)+"."+entry.split(".")[-1])
    if not os.path.exists(file_save_path):
        r = requests.get(entry, stream=True)
        if r.status_code == 200:
            with open(file_save_path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return file_save_path


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


def parse_attributes(category, attributes):
    """
    returns: a dict with key as category and list of all of its attributes 
                 required
    """
    assert os.path.exists(category)
    assert os.path.exists(attributes)

    return parse_file(category), parse_file(attributes)
