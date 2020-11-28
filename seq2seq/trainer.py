import os
import logging
import json

from tqdm import tqdm
from progress_text import ProgressText
import torch
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, 
                 model,
                 loss_fn,
                 optimizer,
                 train_dataloder,
                 valid_dataloder=None,
                 num_epoches=1,
                 start_epoch=0,
                 early_stop_num=10,
                 save_path=None,
                 save_every_epoch=5,
                 plot_loss_group_by="epoch"):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn 
        self.optimizer = optimizer
        self.num_epoches = num_epoches
        self.start_epoch = start_epoch

        self.train_dataloder = train_dataloder
        self.valid_dataloder = valid_dataloder
        if self.valid_dataloder is None:
            logger.warning("Validation data loder is not provided! Using train data loder instead.")
            self.valid_dataloder = train_dataloder

        self.early_stop_num = early_stop_num
        self.save_path = save_path
        self.save_every_epoch = save_every_epoch
        self.plot_loss_group_by = plot_loss_group_by.lower() # option: "epoch" or "update"
        if self.plot_loss_group_by.lower() not in ("epoch", "update"):
            logger.warning("Argument plot_loss_group_by=\"{}\" not supported!"
                           " Swich to plot_loss_group_by=\"epoch\"."\
                                .format(self.plot_loss_group_by))
            self.plot_loss_group_by = "epoch"

        self.now_epoch = -1

        self.train_loss_record = []
        self.valid_loss_record = []
        self.best_record = {'epoch': -1,
                            'train_loss': float('inf'),
                            'valid_loss': float('inf'),
                            'model': None}
        # if there is no better record in early_stop_num round , stop the training process

    def run(self, grad_clip=5.0, progress_indicator="progress-text"):
        for self.now_epoch in range(self.start_epoch, self.num_epoches+1):
            logger.info('Epoch {} starts.'.format(self.now_epoch))

            train_loss_avg, train_losses = self.train_epoch(grad_clip=grad_clip, progress_indicator=progress_indicator)
            valid_loss_avg, valid_losses = self.eval(progress_indicator=progress_indicator)

            logger.info('Trian loss avg: {:.3f}\tValid loss avg: {:.3f}'.format(train_loss_avg, valid_loss_avg))

            self.train_loss_record += [train_loss_avg] if self.plot_loss_group_by == "epoch" else train_losses
            self.valid_loss_record += [valid_loss_avg] if self.plot_loss_group_by == "epoch" else valid_losses
            
            # Update the best record.
            if valid_loss_avg < self.best_record['valid_loss']:
                self.best_record['valid_loss'] = valid_loss_avg
                self.best_record['epoch'] = self.now_epoch
                self.best_record['model'] = self.model

            # If the loss has not been descending in several epochs, stop training.
            if ((self.early_stop_num != None) 
                and (self.early_stop_num > 0)
                and (self.now_epoch - self.best_record['epoch'] >= self.early_stop_num)
                and (valid_loss_avg > self.best_record['valid_loss'])):
                logger.info('Early stop training in epoch {}.\n'
                            '\tThe best epoch is {}.\n'
                            '\tThe best validation loss is {}.'\
                                .format(self.now_epoch,
                                        self.best_record['epoch'],
                                        self.best_record['valid_loss']))
                break

            if self.save_every_epoch > 0 and self.now_epoch % self.save_every_epoch == 0:
                self.save()

        self.save_best()
        self.save_loss_record()
        self.plot_loss()
    
    def plot_loss(self):
        plt.plot(self.train_loss_record, label='Train loss')
        plt.plot(self.valid_loss_record, label='Valid loss')
        plt.legend()
        plt.xlabel("Num {}(s)".format(self.plot_loss_group_by))
        plt.ylabel("Loss")
        plt.savefig(self.save_path + 'train_val_loss_plot.svg')

    def save_best(self):
        logger.info('The best epoch is {}. The best validation loss is {}.'\
                        .format(self.best_record['epoch'] , self.best_record['valid_loss']))
        save_dict = {'model': self.best_record['model'].state_dict(),
                     'epoch': self.best_record['epoch'],
                     'val_loss': self.best_record['valid_loss']}
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            joint_save_path = os.path.join(self.save_path,
                                'best_epoch_{}.pt'.format(save_dict['epoch']))
            torch.save(save_dict, joint_save_path)

    def save(self):
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            joint_save_path = os.path.join(self.save_path,
                                'epoch_{}.pt'.format(self.now_epoch))
            save_dict = {'model':self.model.state_dict(),
                         'optimizer':self.optimizer.state_dict(),
                         'epoch':self.now_epoch}
            torch.save(save_dict, joint_save_path)

    def save_loss_record(self):
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            joint_save_path = os.path.join(self.save_path,
                                'train_valid_loss_record.json')
            save_dict = {'plot_loss_group_by': self.plot_loss_group_by,
                         'train_loss': self.train_loss_record,
                         'valid_loss': self.valid_loss_record}
            with open(joint_save_path, mode="w") as f:
                json.dump(save_dict, f)

    def eval(self, progress_indicator="progress-text"):
        losses = []
        self.model.eval()
        
        if progress_indicator == "progress-text":
            wrapped_iterable = ProgressText(self.valid_dataloder)
        elif progress_indicator == "tqdm":
            wrapped_iterable = tqdm(self.valid_dataloder)
        else:
            wrapped_iterable = self.valid_dataloder
        for inputs in wrapped_iterable:
            target = inputs['target']
            score = self.model(inputs)
            if isinstance(target, (tuple, list)) and len(target) == 2:
                target, target_lengths = target
            loss = self.loss_fn(score, target)
            losses.append(loss.item())
        loss_avg = sum(losses) / len(losses)
        return loss_avg, losses

    def train_epoch(self, grad_clip=5.0, progress_indicator="progress-text"):
        self.model.train()
        losses = []
        if progress_indicator == "progress-text":
            wrapped_iterable = ProgressText(self.train_dataloder)#, every_percent=1)
        elif progress_indicator == "tqdm":
            wrapped_iterable = tqdm(self.train_dataloder)
        else:
            wrapped_iterable = self.train_dataloder
        for inputs in wrapped_iterable:
            target = inputs['target']
            loss = self.train_batch(inputs, target, grad_clip=grad_clip)
            losses.append(loss.item())
            if progress_indicator == "tqdm":
                wrapped_iterable.set_postfix({'current loss': "{:.3f}".format(loss.item())})
        loss_avg = sum(losses) / len(losses)
        return loss_avg, losses

    def train_batch(self, inputs, targets, grad_clip=None):
        # Forward propagation.
        score = self.model(inputs)
        # Calculate loss.
        if isinstance(targets, (tuple, list)) and len(targets) == 2:
            targets, target_lengths = targets
        loss = self.loss_fn(score, targets)
        if torch.isnan(loss):
            logger.error("nan loss encountered")
            raise ValueError("nan loss encountered")
        # Back propagation.
        self.optimizer.zero_grad()
        loss.backward()
        # Clip the gradient.
        if grad_clip is not None and grad_clip > 0 :
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=grad_clip)  # Clip in place.
        # Update model parameter by gradient descent.
        self.optimizer.step()
        return loss
