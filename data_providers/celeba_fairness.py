import PIL
from collections import Counter

import os

import torchvision
import torch.utils.data


from utils import worker_init_fn
import numpy as np

from typing import Callable, List, Optional, Union


class MyCelebaFairness(torchvision.datasets.CelebA):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, split, target_type, transform, target_transform, download)
        target_index = self.attr_names.index('Attractive')
        sensitive_attribute_index = self.attr_names.index('Male')

        self.sensitive_attributes = np.array(
            self.attr[:, sensitive_attribute_index])
        self.targets = np.array(self.attr[:, target_index])
        self.filename = np.array(self.filename)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = PIL.Image.open(os.path.join(
            self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target = self.targets[index]
        sensitive_attribute = self.sensitive_attributes[index]

        if self.transform is not None:
            img = self.transform(img)
        return {'data': img, 'labels': target, 'sensitive_attribute': sensitive_attribute}

    def __len__(self) -> int:
        return len(self.targets)


class CelebaFairnessDataProvider:

    def __init__(self, dataset_path, train_batch_size, test_batch_size, n_workers, 
                 policy_type, train_size=None, val_size=None, **kwargs):

        self._save_path = dataset_path
        self.n_workers = n_workers

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.policy_type = policy_type

        self.train_set = self.train_dataset(None)
        self.val_set = self.val_dataset(None)
        self.test_set = self.test_dataset(None)

        self.train_targets = np.array(self.train_set.targets)
        self.val_targets = np.array(self.val_set.targets)
        self.test_targets = np.array(self.test_set.targets)

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
        raise 'celebadataprovider'

    @property
    def n_classes(self):
        return 2

    def train_dataset(self, _transforms):
        dataset = MyCelebaFairness(
            root=self.save_path, split='train', download=True, transform=_transforms)
        return dataset

    def val_dataset(self, _transforms):
        dataset = MyCelebaFairness(
            root=self.save_path, split='valid', download=True, transform=_transforms)
        return dataset

    def test_dataset(self, _transforms):
        dataset = MyCelebaFairness(
            root=self.save_path, split='test', download=True, transform=_transforms)
        return dataset

    @property
    def save_path(self):
        return self._save_path

    def n_samples(self):
        return len(self.train_set.targets)
