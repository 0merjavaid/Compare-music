import dataloader
from model import backbone
import utils
import argparse
import torch
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser(description='Smart-Cart Experiments')
    parser.add_argument('--val_file',
                        help='Text file containting path to test images')
    parser.add_argument('--model_weight', default=None,
                        help='path to model weight file to be loaded')
    parser.add_argument('--version', default='resnet50',
                        help='version for finetuning')
    parser.add_argument('--attribute',
                        help='Attribute being tested')
    parser.add_argument('--cuda', default=True,
                        help='Use GPU')
    parser.add_argument('--save_dir', default="results",
                        help='Folder to save output')
    parser.add_argument('--save_img', default=True,
                        help='Save images or not')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.cuda = True
    return args


def main():
    args = get_argparser()
    attributes = utils.parse_file(args.attribute)
    attribute = args.model_weight.split("/")[-2]
    attributes = attributes[attribute]
    attribute_model =\
        backbone.Resnet.get_model(args.version, len(attributes),
                                  False,
                                  args.model_weight, 7, train=False
                                  )
    preprocess = utils.preprocess

    total = 0
    correct = 0
    total_per_attribute = {}
    correct_per_attribute = {}
    preds, labels = [], []
    if args.cuda:
        attribute_model.cuda()
    test_set = utils.parse_val_file(args.val_file)
    with torch.no_grad():
        for image in test_set:
            total += 1
            label = image.strip()[-1]
            rgb = Image.open(image.strip()[:-2]).convert("RGB")
            preprocessor = preprocess()
            tensor = preprocessor(rgb).view(1, 3, 224, 224)
            if args.cuda:
                tensor = tensor.cuda()
            output = attribute_model(tensor)
            pred = torch.max(output, 1)[1]
            print (attributes[pred], attributes[int(label)],
                   attributes[pred] == attributes[int(label)])
            if attributes[int(label)] not in total_per_attribute.keys():
                total_per_attribute[attributes[int(label)]] = 1
            else:
                total_per_attribute[attributes[int(label)]] += 1

            if attributes[pred] == attributes[int(label)]:
                if attributes[pred] not in correct_per_attribute.keys():
                    correct_per_attribute[attributes[int(label)]] = 1
                else:
                    correct_per_attribute[attributes[int(label)]] += 1

                correct += 1

            print (float(correct)/total, total, correct)
            preds.append(pred.cpu().numpy()[0])
            labels.append(int(label))

            save_path = os.path.join(
                args.save_dir, attribute)
            # +"/"+image.split("/")[-1][:-2]
            os.makedirs(save_path, exist_ok=True)
            save_img = save_path + \
                "/"+image.split("/")[-1][:-3]

            if args.save_img and not os.path.exists(save_img):
                utils.save_img(save_img, rgb, attribute, {attributes[
                    int(label)]: attributes[int(pred)]})
    save_plt = save_path+"/"+"cmatrix.jpg"
    utils.plot_confusion_matrix(
        labels, preds, np.array(attributes), True, "Confusion Matrix")
    plt.savefig(save_plt)
    print (total_per_attribute.keys(), np.array(
        list(correct_per_attribute.values()))/np.array(list(total_per_attribute.values())))

if __name__ == "__main__":
    main()
