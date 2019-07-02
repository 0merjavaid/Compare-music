import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging
import torch
from utils.utils import *


class MUSIC_ANALYSIS:

    def __init__(self, model, optimizer, cuda=True, experiment_name="", val_step=500):
        self.train_logger = get_logger(
            "loggings", experiment_name+".training", "train_logger", True)
        self.model = model
        self.optimizer = optimizer
        self.cuda = cuda
        self.best_val_accuracy = 0
        self.best_epoch = -1
        self.val_step = val_step
        self.best_model = None

    def predict(self, images, cat, target):
        cat_loss = 0
        if self.cuda:
            images = images.cuda().float()
            labels = labels.cuda().long()
        self.optimizer.zero_grad()
        sync_pred, cat_pred = self.model(images)
        # print ("\n\n", sync_pred, "\n", cat_pred, "\n")
        sync_loss = F.cross_entropy(sync_pred, target)
        # print ("Sync loss", sync_loss)
        for i in range(len(cat)):
            cat_loss += F.cross_entropy(cat_pred[i], cat[i])
            # print ("Cat loss ", cat_loss)

        return sync_pred, cat_pred, sync_loss, cat_loss

    def train(self, epoch, data_iterator, val_iterator, test=False):

        self.model.train()
        sync_losses, cat_losses, sync_accuracies, cat_accuracies = list(), list(), list(), list()
        best_model = None
        best_val_accuracy = 0
        best_epoch = 0
        for i, (images, cat_target, target)in enumerate(tqdm(data_iterator)):
            sync_pred, cat_pred, sync_loss, cat_loss = self.predict(
                images, cat_target, target)

            cat_losses.append(cat_loss.cpu().item())
            sync_losses.append(sync_loss.cpu().item())
            t_accuracy = self.eval_metrics(
                sync_pred, cat_pred, target, cat_target)
            sync_accuracies.append(t_accuracy[0])
            cat_accuracies.append(t_accuracy[1])

            if i % 1 == 0:
                print (sync_accuracies[-10:])
                sync_avg_loss = np.mean(sync_losses[-30:])
                cat_avg_loss = np.mean(cat_losses[-30:])

                sync_avg_acc = np.mean(sync_accuracies[-30:])
                cat_avg_acc = np.mean(cat_accuracies[-30:])

                self.train_logger.info(f'Epoch {epoch}, Iter: {i}, TRAINING__  Closs: {cat_avg_loss}, Sloss{sync_avg_loss}, CAccuracy: {cat_avg_acc}, SAccuracy: {sync_avg_acc}')

            # if i % int(self.val_step) == 0:
            #     self.model.eval()
            #     val_accuracy, val_loss = list(), list()
            #     for j, (v_images, v_labels)in enumerate(tqdm(val_iterator)):
            #         v_output, v_loss, v_labels = self.predict(
            #             v_images, v_labels)

            #         v_accuracy = self.eval_metrics(
            #             v_output, v_labels)
            #         val_accuracy.append(v_accuracy)
            #         val_loss.append(v_loss.cpu().item())
            #     if np.mean(val_accuracy) > self.best_val_accuracy:
            #         self.best_val_accuracy = np.mean(val_accuracy)
            #         self.best_model = self.model
            #         self.best_epoch = epoch

            # self.train_logger.info(f'Epoch" {epoch}, Iter: {i},VALIDATION__
            # loss :{np.mean(val_loss)},Accuracy:{np.mean(val_accuracy)}')

            # self.model.train()
            # cat_loss.backward()
            combined_loss = sync_loss+cat_loss
            combined_loss.backward()
            self.optimizer.step()

    def eval_metrics(self, sync_pred, cat_pred, target, cat_target):
        cat_acc = 0
        sync_acc = float(torch.sum(torch.max(sync_pred, 1)
                                   [1] == target))/sync_pred.shape[0]
        for i in range(len(cat_pred)):
            cat_acc += float(torch.sum(torch.max(cat_pred[i], 1)
                                       [1] == cat_target[i]))/sync_pred.shape[0]
        return sync_acc, cat_acc/2
