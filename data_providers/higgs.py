# data preprocessing is adopted from https://github.com/Yura52/tabular-dl-revisiting-models

from sklearn.preprocessing import QuantileTransformer
import os
import torch


from torch.utils import data
from utils import worker_init_fn, MySubsetRandomSampler
import numpy as np
from collections import Counter


class Higgs(data.Dataset):

    def __init__(self, root, split="train", **kwargs):
        assert split in ["train", "val", "test"]
        path_x = os.path.join(root, 'N_%s.npy' % split)
        path_y = os.path.join(root, 'y_%s.npy' % split)

        self.x_n = np.load(path_x)
        self.targets = np.load(path_y)

    def fill_nans(self, values=0.0):
        for k in range(self.x_n.shape[1]):
            num_nan_indices = np.where(np.isnan(self.x_n[:, k]))
            self.x_n[num_nan_indices, k] = values[k]

    def to_torch(self):
        self.x_n = torch.from_numpy(self.x_n).float()
        self.targets = torch.from_numpy(self.targets).long()

    def __len__(self):
        return len(self.x_n)

    def __getitem__(self, index):
        return self.x_n[index], self.targets[index]


class HiggsDataProvider:
    @staticmethod
    def name():
        return 'HiggsDataProvider'

    @property
    def n_classes(self):
        return 2

    def n_samples(self):
        return len(self.train_set)

    def train_dataset(self, _transforms):
        dataset = Higgs(root=self.save_path, split='train')
        return dataset

    def val_dataset(self, _transforms):
        dataset = Higgs(root=self.save_path, split='val')
        return dataset

    def test_dataset(self, _transforms):
        dataset = Higgs(root=self.save_path, split='test')
        return dataset

    def __init__(self, dataset_path, train_batch_size, test_batch_size, n_workers, train_size, val_size,
                 policy_type, resolution=32, **kwargs):

        self._save_path = dataset_path
        self.n_workers = n_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.train_set = self.train_dataset(None)
        self.val_set = self.val_dataset(None)
        self.test_set = self.test_dataset(None)

        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(self.train_set.x_n.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=42,
        )

        # fill nans
        num_new_values = np.nanmean(self.train_set.x_n, axis=0)
        self.train_set.fill_nans(num_new_values)
        self.val_set.fill_nans(num_new_values)
        self.test_set.fill_nans(num_new_values)

        # normalization
        self.train_set.x_n = normalizer.fit_transform(self.train_set.x_n)
        self.val_set.x_n = normalizer.transform(self.val_set.x_n)
        self.test_set.x_n = normalizer.transform(self.test_set.x_n)

        self.d_num = self.train_set.x_n.shape[1]
        self.cat = []

        self.train_set.to_torch()
        self.val_set.to_torch()
        self.test_set.to_torch()

        self.train_targets = self.train_set.targets
        self.val_targets = self.val_set.targets
        self.test_targets = self.test_set.targets

        print('train Y:', Counter(list(self.train_targets.flatten().numpy())))
        print('val Y:', Counter(list(self.val_targets.flatten().numpy())))
        print('test Y:', Counter(list(self.test_targets.flatten().numpy())))

        self.train_sampler = MySubsetRandomSampler(
            np.arange(len(self.train_set.x_n)))

    def create_dataloaders(self, hyperparameters, transforms):

        data_loaders = {'val': torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=False
        ), 'test': torch.utils.data.DataLoader(
            self.test_set,  # without augmentations
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            drop_last=False),
            'train': torch.utils.data.DataLoader(
            self.train_set,  # with_augmentations
            batch_size=self.train_batch_size,
            # shuffle=True,
                sampler=self.train_sampler,
                num_workers=self.n_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                drop_last=True
        )}

        return data_loaders

    @property
    def save_path(self):
        return self._save_path