import os
import logging
import json
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot

from seq2seq.evaluator.evaluator import Evaluator

logger = logging.getLogger(__name__)


def avg_every_n_elems(l: list, n: int = 1, drop_last: bool = False):
    """
    drop_last:
        If `True`, the last group whose size is less than n will be discarded.
        If `False`, and if the number of the last group of elements is less than n,
        the average of the last group needs to be correctly computed.
    """
    num_remains = len(l) % n
    result = [sum(l[i:i + n]) / n for i in range(0, len(l), n)]
    if num_remains != 0:
        if drop_last:
            result = result[:-1]
        else:
            result[-1] = sum(l[-num_remains:]) / num_remains
    return result


class Trainer(object):
    """docstring for Trainer"""

    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            train_dataloader,
            valid_dataloader=None,
            scheduler=None,
            gradient_accumulation=1,
            num_epochs=1,
            early_stop_num=10,
            save_best_model=True,
            save_path="./checkpoints",
            save_every_epoch=1,
            plot_loss_group_by="epoch",
            plot_loss_group_by_every=1,
            evaluate_before_train=True,
            use_gpu=False,
            config={},
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.gradient_accumulation = gradient_accumulation
        self.num_epochs = num_epochs

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        if self.valid_dataloader is None:
            logger.warning("Validation data loader is not provided! Using train data loader instead.")
            self.valid_dataloader = train_dataloader
        self.evaluator = Evaluator(loss_fn=self.loss_fn, dataloader=self.valid_dataloader)

        if self.gradient_accumulation > 1:
            equiv_batch_size = train_dataloader.batch_size * self.gradient_accumulation
            logger.info("Note! Gradient accumulation setting is greater than 1. "
                        "The equivalent training batch size is the actual batch "
                        "size times the number of gradient accumulation steps. "
                        "Current setting: actual={}, accumulation={}, equivalent={}.".format(
                self.train_dataloader.batch_size,
                self.gradient_accumulation,
                equiv_batch_size))

        self.scheduler = scheduler
        self.early_stop_num = early_stop_num
        self.save_best_model = save_best_model
        self.save_path = save_path
        self.save_every_epoch = save_every_epoch
        self.plot_loss_group_by = plot_loss_group_by.lower()  # option: "epoch" or "update"
        if self.plot_loss_group_by.lower() not in ("epoch", "update"):
            logger.warning("Argument plot_loss_group_by=\"{}\" not supported!"
                           " Switch to plot_loss_group_by=\"epoch\".".format(
                self.plot_loss_group_by))
            self.plot_loss_group_by = "epoch"
        self.plot_loss_group_by_every = plot_loss_group_by_every

        self.evaluate_before_train = evaluate_before_train

        self.use_gpu = use_gpu
        if self.use_gpu and not torch.cuda.is_available():
            raise ImportError("(self.use_gpu == True) but (torch.cuda.is_available() == False)")

        self.now_epoch = -1

        self.config = config
        config_save_path = os.path.join(self.save_path, 'config.json')
        if not os.path.exists(config_save_path):
            with open(config_save_path, mode='w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)

        self.train_loss_record = []
        self.valid_loss_record = []
        self.best_record = {'epoch': -1,
                            'train_loss': float('inf'),
                            'valid_loss': float('inf'),
                            'model': None}

    def run(self, grad_clip=5.0, use_tqdm: bool = True):
        if self.evaluate_before_train and self.now_epoch < 0:
            logger.info("Evaluation before training.")
            # Evaluate on training set.
            valid_dataloader_backup = self.valid_dataloader
            self.evaluator.dataloader = self.train_dataloader
            train_loss_avg, train_losses = self.evaluator.eval(self.model, use_tqdm=use_tqdm)
            self.evaluator.dataloader = valid_dataloader_backup
            # Evaluate on validation set.
            valid_loss_avg, valid_losses = self.evaluator.eval(self.model, use_tqdm=use_tqdm)
            # Record loss. Only record the average over the dataset instead of each batch.
            logger.info('Train loss avg: {:.3f}\tValid loss avg: {:.3f}'.format(train_loss_avg, valid_loss_avg))
            self.train_loss_record += [train_loss_avg]
            self.valid_loss_record += [valid_loss_avg]

        # Note that the epoch counter starts at 1 and end at max number of epochs + 1,
        # which is [1, num_epochs], which is also [1, num_epochs + 1)
        # Start from scratch or continue from a checkpoint?
        start_epoch = 1 if self.now_epoch < 0 else self.now_epoch + 1
        for self.now_epoch in range(start_epoch, self.num_epochs + 1):
            logger.info('Epoch {} starts.'.format(self.now_epoch))

            train_loss_avg, train_losses = self.train_epoch(grad_clip=grad_clip, use_tqdm=use_tqdm)
            valid_loss_avg, valid_losses = self.evaluator.eval(self.model, use_tqdm=use_tqdm)

            if self.scheduler is not None:
                self.scheduler.step()

            logger.info('Train loss avg: {:.3f}\tValid loss avg: {:.3f}'.format(train_loss_avg, valid_loss_avg))

            train_losses_grouped = avg_every_n_elems(train_losses, self.plot_loss_group_by_every)
            valid_losses_grouped = avg_every_n_elems(valid_losses, self.plot_loss_group_by_every)

            self.train_loss_record += [train_loss_avg] if self.plot_loss_group_by == "epoch" else train_losses_grouped
            self.valid_loss_record += [valid_loss_avg] if self.plot_loss_group_by == "epoch" else valid_losses_grouped

            # Update the best record.
            if valid_loss_avg < self.best_record['valid_loss']:
                self.best_record['valid_loss'] = valid_loss_avg
                self.best_record['epoch'] = self.now_epoch
                if self.save_best_model:
                    self.best_record['model'] = self.model

            # If the loss has not been descending in several epochs, stop training.
            if ((self.early_stop_num is not None)
                    and (self.early_stop_num > 0)
                    and (self.now_epoch - self.best_record['epoch'] >= self.early_stop_num)
                    and (valid_loss_avg > self.best_record['valid_loss'])):
                logger.info('Early stop training in epoch {}.\n'
                            '\tThe best epoch is {}.\n'
                            '\tThe best validation loss is {}.'.format(
                    self.now_epoch,
                    self.best_record['epoch'],
                    self.best_record['valid_loss']))
                break

            if self.save_every_epoch > 0 and self.now_epoch % self.save_every_epoch == 0:
                self.save()

            self.save_loss_record()  # Update loss record after each epoch
            self.plot_loss()  # Update loss plot after each epoch

        self.save_loss_record()
        self.plot_loss()
        if self.save_best_model:
            self.save_best()

    def plot_loss(self):
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        pyplot.clf()  # Clear current figures.
        pyplot.plot(self.train_loss_record, label='Train loss')
        pyplot.plot(self.valid_loss_record, label='Valid loss')
        pyplot.legend()  # Show the label of each curve.
        pyplot.xlabel("Num x{} {}(s)".format(self.plot_loss_group_by_every, self.plot_loss_group_by))
        pyplot.ylabel("Loss")
        pyplot.savefig(os.path.join(self.save_path, 'train_val_loss_plot.svg'))

    def save_best(self):
        logger.info('The best epoch is {}. The best validation loss is {}.'.format(
            self.best_record['epoch'],
            self.best_record['valid_loss']))
        save_dict = {'model': self.best_record['model'].state_dict(),
                     'epoch': self.best_record['epoch'],
                     'val_loss': self.best_record['valid_loss']}
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            joint_save_path = os.path.join(self.save_path,
                                           'best_epoch_{}-model_only.pt'.format(save_dict['epoch']))
            torch.save(save_dict, joint_save_path)

    def save(self):
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            joint_save_path = os.path.join(self.save_path,
                                           'epoch_{}-train_checkpoint.pt'.format(self.now_epoch))
            save_dict = {'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                         'epoch': self.now_epoch,
                         'train_loss_record': self.train_loss_record,
                         'valid_loss_record': self.valid_loss_record}
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

    def load(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        logger.info('Loading checkpoint file [{}].'.format(filename))
        checkpoint = torch.load(filename)

        logger.info('Loading model state dictionary.')
        self.model.load_state_dict(checkpoint['model'])
        if self.use_gpu:
            self.model.to(torch.device("cuda"))

        logger.info('Loading model optimizer state dictionary.')
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if (self.scheduler is not None) and (checkpoint['scheduler'] is not None):
            logger.info('Loading model optimizer scheduler state dictionary.')
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.now_epoch = checkpoint['epoch']
        self.train_loss_record = checkpoint['train_loss_record']
        self.valid_loss_record = checkpoint['valid_loss_record']

    def train_epoch(self, grad_clip=5.0, use_tqdm: bool = True):
        self.model.train()
        losses = []

        if use_tqdm:
            wrapped_iterable = tqdm(self.train_dataloader)
            wrapped_iterable.set_description("Train Epoch {}".format(self.now_epoch))
        else:
            wrapped_iterable = self.train_dataloader

        current_grad_accumulation_count = 0
        current_batch, num_batches = 0, len(self.train_dataloader)
        loss_items_accumulation_normalized = []
        for inputs in wrapped_iterable:
            current_batch += 1
            target = inputs['target']
            loss = self.loss_on_batch(inputs, target)
            loss /= self.gradient_accumulation  # Normalize the loss (if averaged)
            loss_items_accumulation_normalized.append(loss.item())
            loss.backward()  # Back propagation.
            current_grad_accumulation_count += 1
            if current_grad_accumulation_count == self.gradient_accumulation or current_batch == num_batches:
                self.update_params(grad_clip=grad_clip)
                current_grad_accumulation_count = 0
                losses.append(sum(loss_items_accumulation_normalized))  # Append the mean.
                loss_items_accumulation_normalized = []
                if use_tqdm:
                    wrapped_iterable.set_postfix({'current loss': "{:.3f}".format(losses[-1])})
        loss_avg = sum(losses) / len(losses)
        return loss_avg, losses

    def loss_on_batch(self, inputs, targets):
        # Forward propagation.
        score = self.model(inputs)
        # Calculate loss.
        if isinstance(targets, (tuple, list)) and len(targets) == 2:
            targets, target_lengths = targets
        loss = self.loss_fn(score, targets)
        if torch.isnan(loss):
            logger.error("NaN loss encountered")
            raise ValueError("NaN loss encountered")
        return loss

    def update_params(self, grad_clip):
        if grad_clip is not None and grad_clip > 0:  # Clip the gradient.
            # Trailing underscore means clip in place.
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=grad_clip)
        self.optimizer.step()  # Update model parameter by gradient descent.
        self.optimizer.zero_grad()  # Clear the gradient for the next update.
