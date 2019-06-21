import dataloader
from model import backbone
import trainer
import utils
import argparse
import logging
import torch
from tqdm import tqdm
import os


def get_argparser():
    parser = argparse.ArgumentParser(description='Smart-Cart Experiments')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--dataset_dir', default='datasets/MVC/frequencies.npy',
                        help='directory that contains images and labels folders')
    parser.add_argument('--images_folder', default='datasets/MVC/resized_squares',
                        help='directory that contains images and labels folders')
    parser.add_argument('--optimizer', default='adam',
                        help='which optimizer to use, adam, gradient decent etc')
    parser.add_argument('--model_weight', default=None,
                        help='path to model weight file to be loaded')
    parser.add_argument('--checkpoint_dir', default='checkpoints', metavar='LR',
                        help='directory path to store checkpoints')
    parser.add_argument('--no_of_attributes', default=264,
                        help='total number of attributes')
    parser.add_argument('--experiment_name', default="attributes",
                        help='name of current experiment')
    parser.add_argument('--testing', default=False,
                        help='true if testing stage')
    parser.add_argument('--version', default='resnet50',
                        help='version for finetuning')
    parser.add_argument('--freeze_layers', default=7,
                        help='number of layers to freeze')
    parser.add_argument('--val_step', default=100,
                        help='iterations to validation')
    args = parser.parse_args()
    logger = utils.utils.get_logger(
        "loggings", args.experiment_name+".global", "global")
    if torch.cuda.is_available():
        args.cuda = True
        logger.info("Using GPU for Training")
    else:
        logger.info("Using CPU for Training")

    return args, logger


def write_val(path, files):
    with open(path, "w") as f:
        for file in files:
            file = [str(i) for i in file]
            file = "_".join(file)
            f.write(file)
            f.write("\n")


def main():
    args, logger = get_argparser()
    logger.info(f'{args}')

    train_loader = dataloader.dataloader.Trendage_HDD(
        args.images_folder, 0, 96,  "*jpg", "Train", args.experiment_name)
    train_iterator = torch.utils.data.DataLoader(train_loader,
                                                 batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4,
                                                 pin_memory=True)

    # val_loader = dataloader.dataloader.Trendage_HDD(
    #     args.images_folder, 96, 100, "*jpg", "Val", args.experiment_name)
    # val_iterator = torch.utils.data.DataLoader(val_loader,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=True, num_workers=4,
    #                                            pin_memory=True)

    # classes, val_set = val_loader.get_meta()
    # write_val("loggings/"+args.experiment_name+"/val.txt", val_set)
    attribute_model =\
        backbone.Resnet.get_model(args.version, len(classes),
                                  True,
                                  args.model_weight,
                                  int(args.freeze_layers)
                                  )
    if args.cuda:
        attribute_model.cuda()

    optimizer = torch.optim.Adam(attribute_model.parameters(), lr=args.lr)
    attribute_trainer = trainer.Classifier(
        attribute_model, optimizer, args.cuda, args.experiment_name, args.val_step)

    for epoch in range(args.epochs):
        prev_val_acc = attribute_trainer.best_val_accuracy
        attribute_trainer.train(epoch, train_iterator,
                                val_iterator, test=False)
        logger.info(
            f'Epoch {epoch}, Best Epoch {attribute_trainer.best_epoch},\
            Best Accuracy {attribute_trainer.best_val_accuracy}')

        save_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        if attribute_trainer.best_val_accuracy > prev_val_acc:
            torch.save(attribute_trainer.best_model.state_dict(), os.path.join(
                save_dir, args.version+"_"+str(attribute_trainer.best_epoch) +
                "_"+str(attribute_trainer.best_val_accuracy)+".pt"))


if __name__ == '__main__':
    main()
