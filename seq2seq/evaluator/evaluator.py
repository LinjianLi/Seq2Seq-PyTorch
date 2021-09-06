import torch
from tqdm import tqdm


class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self,
                 loss_fn,
                 dataloader):
        super(Evaluator, self).__init__()
        self.loss_fn = loss_fn
        self.dataloader = dataloader

    def __repr__(self):
        main_string = "Evaluator(loss_fn={}, dataloader=(batch_size={}, len={}))"\
                        .format(self.loss_fn, self.dataloader.batch_size, len(self.dataloader))
        return main_string

    def eval(self, model, use_tqdm: bool = True):
        if use_tqdm:
            wrapped_iterable = tqdm(self.dataloader)
            wrapped_iterable.set_description("Evaluation")
        else:
            wrapped_iterable = self.dataloader

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
