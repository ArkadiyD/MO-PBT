from collections import Counter

import os

import torchvision
import torch.utils.data

from sklearn.model_selection import StratifiedShuffleSplit

from utils import worker_init_fn
import pickle
import numpy as np



class CIFARBaseDataProvider:

    def __init__(self, dataset_path, train_batch_size, test_batch_size, n_workers, train_size, val_size,
                 policy_type, **kwargs):

        self._save_path = dataset_path
        self.n_workers = n_workers

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.policy_type = policy_type

        self.train_set = self.train_dataset(None)
        self.val_set = self.val_dataset(None)
        self.test_set = self.test_dataset(None)

        # If using for the first time
        # train/val stratified split, val size is 10000
        stratified_targets_split = StratifiedShuffleSplit(
            n_splits=1, test_size=50000 - train_size, random_state=0)
        stratified_targets_split2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=0)
        for train_indices, extended_val_indices in stratified_targets_split.split(self.train_set.targets, self.train_set.targets):
            if len(extended_val_indices) > val_size:
                for _, val_indices in stratified_targets_split2.split(np.array(self.train_set.targets)[np.array(extended_val_indices)], np.array(self.train_set.targets)[np.array(extended_val_indices)]):
                    val_indices = extended_val_indices[val_indices]
            else:
                val_indices = extended_val_indices
        print("TRAIN ind:", len(train_indices))
        print("VAL ind:", len(val_indices))

        assert len(set(train_indices).intersection(set(val_indices))) == 0

        indices = {'train': train_indices, 'val': val_indices}
        pickle.dump(indices, open(os.path.join(
            self._save_path, 'data_split.pkl'), 'wb'))

        self.train_set.data = self.train_set.data[train_indices]
        self.train_set.targets = np.array(
            self.train_set.targets)[train_indices]
        self.val_set.data = self.val_set.data[val_indices]
        self.val_set.targets = np.array(self.val_set.targets)[val_indices]

        self.train_targets = np.array(
            self.train_set.targets)  # [train_indices]
        self.val_targets = np.array(self.val_set.targets)  # [val_indices]
        self.test_targets = np.array(self.test_set.targets)

        # Making sure that class proportions in all subsets are equal
        print('Number of samples per class in train:',
              Counter(self.train_targets))
        print('Number of samples per class in val:', Counter(self.val_targets))
        print('Number of samples per class in test:', Counter(self.test_targets))

    def create_dataloaders(self, hyperparameters, transforms):

        self.train_set.transform = transforms['train']
        self.val_set.transform = transforms['test']
        self.test_set.transform = transforms['test']

        data_loaders = {'val': torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            # sampler=self.val_sampler,
            num_workers=self.n_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn, drop_last=False
        ), 'test': torch.utils.data.DataLoader(
            self.test_set,  # without augmentations
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True, drop_last=False),
            'train': torch.utils.data.DataLoader(
            self.train_set,  # with_augmentations
            batch_size=self.train_batch_size,
            # sampler=self.train_sampler,
            shuffle=True,
            num_workers=self.n_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            drop_last=True
        )}
        return data_loaders

    @staticmethod
    def name():
        raise NotImplementedError

    @property
    def n_classes(self):
        raise NotImplementedError

    def train_dataset(self, _transforms):
        raise NotImplementedError

    def val_dataset(self, _transforms):
        raise NotImplementedError

    def test_dataset(self, _transforms):
        raise NotImplementedError

    @property
    def save_path(self):
        return self._save_path

    def n_samples(self):
        return len(self.train_set.targets)


class CIFAR10DataProvider(CIFARBaseDataProvider):
    @staticmethod
    def name():
        return 'cifar10'

    @property
    def n_classes(self):
        return 10

    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(root=self.save_path, train=True,
                                               download=True, transform=_transforms)
        return dataset

    def val_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(root=self.save_path, train=True,
                                               download=True, transform=_transforms)
        return dataset

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(root=self.save_path, train=False,
                                               download=True, transform=_transforms)
        return dataset


class CIFAR100DataProvider(CIFARBaseDataProvider):
    @staticmethod
    def name():
        return 'cifar100'

    @property
    def n_classes(self):
        return 100

    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(root=self.save_path, train=True,
                                                download=True, transform=_transforms)
        return dataset

    def val_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(root=self.save_path, train=True,
                                                download=True, transform=_transforms)
        return dataset

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(root=self.save_path, train=False,
                                                download=True, transform=_transforms)
        return dataset
