import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging
import torch
from utils.utils import *


class ML_Trainer:

    def __init__(self, model, optimizer, cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.cuda = cuda

    def predict(self, images, labels, att_weight):
        if self.cuda:
            images = images.cuda().float()
            labels = labels.cuda().float()
            att_weight = att_weight.cuda().float()
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = F.binary_cross_entropy(F.sigmoid(output), target=labels)
        weighted_loss = F.binary_cross_entropy(
            F.sigmoid(output), target=labels, weight=att_weight, reduction='sum')
        return output, labels, loss, weighted_loss

    def log(self, output, labels, att_weight):
        l = np.where(labels[0].cpu().data.numpy() == 1)
        p = np.where(torch.round(
            F.sigmoid(output[0])).cpu().data.numpy() == 1)
        w = att_weight[0].cpu().data.numpy()[np.where(
            labels[0].cpu().data.numpy() == 1)[0]]
        return l, p, w

    def train(self, epoch, data_iterator, val_iterator, test=False):
        if not test:
            self.model.train()
            train_loss, train_w_loss, train_precision, train_recall = list(), list(), list(), list()
            for i, (images, labels, att_weight)in enumerate(tqdm(data_iterator)):
                output, labels, loss, weighted_loss = self.predict(
                    images, labels, att_weight)
                if i % 20 == 0:
                    l, p, w = self.log(output, labels, att_weight)
                    t_precision, t_recall = self.eval_metrics(output, labels)
                    train_loss.append(loss.cpu().item())
                    train_w_loss.append(weighted_loss.cpu().item())
                    train_precision.append(t_precision)
                    train_recall.append(t_recall)
                    train_log.info(f'TRAINING__   loss :{loss}, weighted_loss :{weighted_loss} ,Precision :{t_precision}, Recall :{t_recall}')

                if i % 200 == 0:
                    self.model.eval()
                    val_precision, val_recall, val_loss, val_w_loss = list(), list(), list(), list()
                    for j, (v_images, v_labels, v_att_weight)in enumerate(tqdm(val_iterator)):
                        v_output, v_labels, v_loss, v_weighted_loss = self.predict(
                            v_images, v_labels, v_att_weight)

                        v_presision, v_recall = self.eval_metrics(
                            v_output, v_labels)
                        val_precision.append(v_presision)
                        val_recall.append(v_recall)
                        val_loss.append(loss.cpu().item())
                        val_w_loss.append(v_weighted_loss.cpu().item())
                    val_log.info(f'VALIDATION__   loss :{np.mean(val_loss)}, weighted_loss :{np.mean(val_w_loss)}, Precision :{np.mean(val_precision)}, Recall :{np.mean(val_recall)}')

                    self.model.train()

                weighted_loss.backward()
                self.optimizer.step()
        if test:
            self.model.eval()
            test_precision, test_recall, test_loss, test_w_loss = list(), list(), list(), list()
            for i, (images, labels, att_weight)in enumerate(tqdm(data_iterator)):
                output, labels, loss, weighted_loss = self.predict(
                    images, labels, att_weight)
                t_precision, t_recall = self.eval_metrics(output, labels)
                test_loss.append(loss.cpu().item())
                test_w_loss.append(weighted_loss.cpu().item())
                test_precision.append(t_precision)
                test_recall.append(t_recall)
            val_log.info(f'TESTING__   loss :{np.mean(test_loss)}, weighted_loss :{np.mean(test_w_loss)}, Precision :{np.mean(test_precision)},Recall :{np.mean(test_recall)}')

    def eval_metrics(self, pred, target):
        target = target.cpu().data.numpy()
        pred = torch.round(
            F.sigmoid(pred)).cpu().data.numpy()
        precisions = list()
        recalls = list()
        for i in range(target.shape[0]):
            print("Target,  ", np.where(target[i] == 1)[0])
            print ("Pred    ", np.where(pred[i] == 1)[0], "\n\n")
            intersection = np.intersect1d(
                np.where(target[i] == 1)[0], np.where(pred[i] == 1)[0])
            total_preds = len(np.where(pred[i] == 1)[0])
            precision = float(len(intersection))/(total_preds)

            recall = float(len(intersection)/len(np.where(target[i] == 1)[0]))
            precisions.append(precision)
            recalls.append(recall)
        ap, ar = np.mean(precisions), np.mean(recalls)
        return ap*100, ar*100


class Classifier:

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

    def predict(self, images, labels):
        if self.cuda:
            images = images.cuda().float()
            labels = labels.cuda().long()
        self.optimizer.zero_grad()
        output = self.model(images)
        criterion = nn.CrossEntropyLoss()
        labels = torch.argmax(labels, 1)
        loss = criterion(output, labels)

        return output, loss, labels

    def train(self, epoch, data_iterator, val_iterator, test=False):
        self.model.train()
        train_loss, train_accuracy = list(), list()
        best_model = None
        best_val_accuracy = 0
        best_epoch = 0
        for i, (images, labels)in enumerate(tqdm(data_iterator)):
            output, loss, labels = self.predict(
                images, labels)

            train_loss.append(loss.cpu().item())
            t_accuracy = self.eval_metrics(output, labels)
            train_accuracy.append(t_accuracy)
            if i % 20 == 0:
                avg_loss = np.mean(train_loss[-10:])
                avg_acc = np.mean(train_accuracy[-10:])

                self.train_logger.info(f'Epoch" {epoch}, Iter: {i},TRAINING__   loss :{loss},Accuracy :{t_accuracy}, smooth_loss: {avg_loss}, smooth_acc: {avg_acc}')

            if i % int(self.val_step) == 0:
                self.model.eval()
                val_accuracy, val_loss = list(), list()
                for j, (v_images, v_labels)in enumerate(tqdm(val_iterator)):
                    v_output, v_loss, v_labels = self.predict(
                        v_images, v_labels)

                    v_accuracy = self.eval_metrics(
                        v_output, v_labels)
                    val_accuracy.append(v_accuracy)
                    val_loss.append(v_loss.cpu().item())
                if np.mean(val_accuracy) > self.best_val_accuracy:
                    self.best_val_accuracy = np.mean(val_accuracy)
                    self.best_model = self.model
                    self.best_epoch = epoch

                self.train_logger.info(f'Epoch" {epoch}, Iter: {i},VALIDATION__   loss :{np.mean(val_loss)},Accuracy:{np.mean(val_accuracy)}')

                self.model.train()

            loss.backward()
            self.optimizer.step()

    def eval_metrics(self, pred, target):
        return float(torch.sum(torch.max(pred, 1)[1] == target))/pred.shape[0]
