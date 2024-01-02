# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/utils.py

from http.client import UnimplementedFileMode
import numpy as np
import torch
import os


class EarlyStopping:
    def __init__(self, patience=100, store_path="es_checkpoint", gauge='acc'):
        # self.gauge = gauge
        self.patience = patience
        self.counter = 0
        self.best_score = None
        # self.record_val_acc = None

        self.best_epoch = -1
        self.epoch_id = -1

        self.early_stop = False
        self.store_path = os.path.join("cache", "ckpts", store_path)
        if not os.path.exists(os.path.dirname(self.store_path)):
            os.mkdir(os.path.dirname(self.store_path))
        self.history = []

        if gauge == 'acc':
            self.step = self.step_acc
        else:
            self.step = self.step_loss


    def step_acc(self, val_acc, model):
        self.epoch_id += 1
        self.history.append(val_acc)

        if self.best_score is None:
            self.best_score = val_acc
            self.save_checkpoint(model)
            self.best_epoch = self.epoch_id
        elif val_acc <= self.best_score:  
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # stop
        else:
            self.best_score = val_acc
            self.save_checkpoint(model)
            self.counter = 0
            self.best_epoch = self.epoch_id
        return self.early_stop


    def step_loss(self, val_loss, model):
        self.epoch_id += 1
        self.history.append(val_loss)

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.best_epoch = self.epoch_id
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # stop
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.best_epoch = self.epoch_id
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.store_path)