"""
utility classes for adaptations

- CombinedLoader: combine source & target dataloader into one (e.g. batch = {'source_domain': input_src, 'target_domain': input_tgt}).
- DomainAdaptationDataModule: data modules for trainer, warps train, val, test datasets into source, target, val, test dataloaders.

"""
from itertools import cycle
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader

from configs.PROBLEM_CONFIG import target_shorter

class CombinedLoader:
    def __init__(self, source_loader, target_loader, cycle=True):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.cycle = cycle

    def __iter__(self):
        self.source_iter = iter(self.source_loader)
        if self.cycle:
            self.target_iter = cycle(self.target_loader)  # ensures continuous looping
        else:
            self.target_iter = iter(self.target_loader)
        return self

    def __next__(self):
        try:
            source_batch = next(self.source_iter)
        except StopIteration:
            raise StopIteration
        target_batch = next(self.target_iter)

        if "climate" in source_batch and source_batch["climate"].size(0) > target_batch["climate"].size(0):
            b = target_batch["climate"].size(0)
            for k in source_batch.keys():
                if k == "target":
                    for t in source_batch["target"].keys():
                        source_batch["target"][t] = source_batch["target"][t][:b]
                else:
                    source_batch[k] = source_batch[k][:b]

        return {'source_domain': source_batch, 'target_domain': target_batch}

    def __len__(self):
        if self.cycle:
            return len(self.source_loader)
        else:
            return len(self.target_loader)
    

class PairedDataset(Dataset):
    def __init__(self, dataset_orig, dataset_aug):
        assert len(dataset_orig) == len(dataset_aug), "Datasets must be the same length"
        self.dataset_orig = dataset_orig
        self.dataset_aug = dataset_aug

    def __len__(self):
        return len(self.dataset_orig)

    def __getitem__(self, idx):
        sample_orig = self.dataset_orig[idx]
        sample_aug = self.dataset_aug[idx]

        # original & aug share the same target, year..., only climate data is augmented.
        data = {"original": {"climate": sample_orig["climate"], "doys": sample_orig["doys"]},
                "aug": {"climate": sample_aug["climate"], "doys": sample_aug["doys"]}}
        for k in sample_orig.keys():
            if k == "climate": 
                continue
            data[k] = sample_orig[k]

        return data
    

class DomainAdaptationDataModule(pl.LightningDataModule):
    def __init__(self, source_dataset, target_dataset, 
                 train_dataset, val_dataset, test_dataset, batch_size,
                 train_loader_residual=None,
                 ):
        super().__init__()
        self.source_dataset = source_dataset  # train
        self.target_dataset = target_dataset  # val + test during training
        self.train_dataset = train_dataset # no aug
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_loader_residual = train_loader_residual

    def setup(self, stage=None):  
        self.source_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.target_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.train_loader =  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # for eval
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def train_dataloader(self):
        if self.train_loader_residual:
            # use residual setting
            return CombinedLoader(self.train_loader_residual, self.target_loader)
        else:
            return CombinedLoader(self.source_loader, self.target_loader)

    def val_dataloader(self):
        if self.train_loader_residual:
            # use residual setting
            return CombinedLoader(self.train_loader, self.val_loader, cycle=False)
        else:
            return self.val_loader

    def test_dataloader(self):
        if self.train_loader_residual:
            # use residual setting
            return CombinedLoader(self.train_loader, self.test_loader, cycle=False)
        else:
            return self.test_loader