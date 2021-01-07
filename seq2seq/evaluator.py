from tqdm import tqdm
from progress_text import ProgressText
import torch
import logging


class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self,
                 loss_fn,
                 dataloder):
        super(Evaluator, self).__init__()
        self.loss_fn = loss_fn
        self.dataloder = dataloder

    def __repr__(self):
        main_string = "Evaluator(loss_fn={}, dataloder=(batch_size={}, len={}))"\
                        .format(self.loss_fn, dataloder.batch_size, len(dataloder))
        return main_string

    def eval(self, model, progress_indicator="progress-text"):
        if progress_indicator == "progress-text":
            wrapped_iterable = ProgressText(self.dataloder, task_name="Evaluation")
        elif progress_indicator == "tqdm":
            wrapped_iterable = tqdm(self.dataloder)
            wrapped_iterable.set_description("Evaluation")
        else:
            wrapped_iterable = self.dataloder

        losses = []
        model.eval()
        with torch.no_grad():
            for inputs in wrapped_iterable:
                target = inputs['target']
                score = model(inputs)
                if isinstance(target, (tuple, list)) and len(target) == 2:
                    target, target_lengths = target
                loss = self.loss_fn(score, target)
                losses.append(loss.item())
        loss_avg = sum(losses) / len(losses)
        return loss_avg, losses
