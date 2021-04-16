import torch
from torch.utils.data import DataLoader

from seq2seq.utility.utilities import list2tensor


class Pack(dict):
    """
    Pack
    """
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        """
        add
        """
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        """
        flatten
        """
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.to(device) for x in v)
            else:
                pack[k] = v.to(device)
        return pack


class Dataset(torch.utils.data.Dataset):
    """
    Dataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=None):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if torch.cuda.is_available() and device is not None:
                batch = batch.cuda(device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=None, **kwargs):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False,
                            **kwargs)
        return loader
