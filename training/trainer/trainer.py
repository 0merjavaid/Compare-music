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
        self.best_val_accuracy_sync = 0
        self.best_val_accuracy_cat = 0
        self.best_epoch = -1
        self.val_step = val_step
        self.best_model = None

    def predict(self, images, cat, target):
        cat_loss = 0
        if self.cuda:
            for i in range(len(images)):
                images[i] = images[i].cuda().float()
                cat[i] = cat[i].cuda().long()
            target = target.cuda().float()
        self.optimizer.zero_grad()
        sync_pred, cat_pred = self.model(images)
        # print (sync_pred, target)
        # 0/0
        # print ("\n\n", sync_pred, "\n", cat_pred, "\n")
        # sync_loss = F.cross_entropy(sync_pred, target)
        loss = nn.BCELoss()
        sync_loss = loss(sync_pred, target)
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

            if i % 10 == 0:
                sync_avg_loss = np.mean(sync_losses[-10:])
                cat_avg_loss = np.mean(cat_losses[-10:])

                sync_avg_acc = np.mean(sync_accuracies[-10:])
                cat_avg_acc = np.mean(cat_accuracies[-10:])

                self.train_logger.info(f'Epoch {epoch}, Iter: {i}, TRAINING__  Closs: {cat_avg_loss}, Sloss{sync_avg_loss}, CAccuracy: {cat_avg_acc}, SAccuracy: {sync_avg_acc}')
###
            v_sync_losses, v_cat_losses, v_sync_accuracies, v_cat_accuracies = list(
            ), list(), list(), list()

            if (i+1) % int(self.val_step) == 0:
                self.model.eval()
                val_accuracy, val_loss = list(), list()
                for i, (v_images, v_cat_target, v_target)in enumerate(tqdm(val_iterator)):
                    v_sync_pred, v_cat_pred, v_sync_loss, v_cat_loss = self.predict(
                        v_images, v_cat_target, v_target)
                    v_cat_losses.append(v_cat_loss.cpu().item())
                    v_sync_losses.append(v_sync_loss.cpu().item())
                    v_accuracy = self.eval_metrics(
                        v_sync_pred, v_cat_pred, v_target, v_cat_target)
                    v_sync_accuracies.append(v_accuracy[0])
                    v_cat_accuracies.append(v_accuracy[1])
                v_sync_avg_loss = np.mean(v_sync_losses)
                v_cat_avg_loss = np.mean(v_cat_losses)
                v_sync_avg_acc = np.mean(v_sync_accuracies)
                v_cat_avg_acc = np.mean(v_cat_accuracies)

                self.train_logger.info(f'Epoch {epoch}, Iter: {i}, Validation__  Closs: {v_cat_avg_loss}, SVloss{v_sync_avg_loss}, CVAccuracy: {v_cat_avg_acc}, AVG Accuracy: {v_sync_avg_acc}')
                if v_sync_avg_acc*100 > self.best_val_accuracy_sync*100:
                    self.best_val_accuracy_sync = v_sync_avg_acc
                    self.best_val_accuracy_cat = v_cat_avg_acc
                    self.best_model = self.model
                    self.best_epoch = epoch

            self.model.train()
            # cat_loss.backward()
            combined_loss = 9*(sync_loss) + cat_loss
            # sync_loss.backward()
            combined_loss.backward()
            self.optimizer.step()

    def eval_metrics(self, sync_pred, cat_pred, target, cat_target):
        cat_acc = 0
        # sync_acc = float(torch.sum(torch.max(sync_pred.cuda().long(), 1)
        # [1] == target.cuda().long()))/sync_pred.shape[0]
        bin_acc = (sync_pred > 0.5).float() * 1
        sync_acc = float(torch.sum(
            bin_acc == target.cuda().float()))/sync_pred.shape[0]
        for i in range(len(cat_pred)):
            cat_acc += float(torch.sum(torch.max(cat_pred[i].cuda().long(), 1)
                                       [1] == cat_target[i].cuda().long()))/sync_pred.shape[0]
        return sync_acc, cat_acc/2
