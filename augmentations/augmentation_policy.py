import numpy as np
import torch
import augmentations.pil_augmentations as pil_augs
import random

def get_prob_from_tree(probs):
    #obtaining the number of augmentations in a probabilstic way
    cur_value = 0
    for prob in probs:
        if np.random.uniform(0,1) >= prob:
            return cur_value
        cur_value += 1
    return cur_value

class PBASearchSpace:
    def __init__(self, prob_bins=10, mag_bins=10):
        self.prob_bins = prob_bins
        self.mag_bins = mag_bins

        pil_augs.set_augmentation_space('wide', mag_bins)

        self.search_space = {
            'auto_contrast': [np.linspace(0, 1.0, prob_bins)],
            'equalize': [np.linspace(0, 1.0, prob_bins)],
            'invert': [np.linspace(0, 1.0, prob_bins)],
            'rotate': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'posterize': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'solarize': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'color': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'contrast': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'brightness': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'sharpness': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'shear_x': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'shear_y': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'translate_x': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'translate_y': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],

            'special_Cutout': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],


            'op_1': [np.linspace(0, 1.0, prob_bins)],
            'op_2': [np.linspace(0, 1.0, prob_bins)],
            'op_3': [np.linspace(0, 1.0, prob_bins)],

        }

    def apply_op(self, img, op, mag_bin):
        real_mag = 0
        if len(self.search_space[op]) == 2:
            real_mag = mag_bin
        return getattr(pil_augs, op)(img, real_mag)


class RandAugmentSearchSpace:
    def __init__(self, prob_bins=10, mag_bins=10):
        self.prob_bins = prob_bins
        self.mag_bins = mag_bins

        pil_augs.set_augmentation_space('wide', mag_bins)

        self.search_space = {
            'n': [np.array([0, 1, 2, 3, 4])],
            'm': [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
            'special_Cutout': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)]}

        self.augs_list = {
            'auto_contrast': [np.linspace(0, 1.0, prob_bins)],
            'equalize': [np.linspace(0, 1.0, prob_bins)],
            'invert': [np.linspace(0, 1.0, prob_bins)],
            'rotate': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'posterize': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'solarize': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'color': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'contrast': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'brightness': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'sharpness': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'shear_x': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'shear_y': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'translate_x': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
            'translate_y': [np.linspace(0, 1.0, prob_bins), np.linspace(0, 9, mag_bins)],
        }

    def apply_op(self, img, op, mag_bin):
        real_mag = 0
        if len(self.search_space[op]) == 2:
            real_mag = mag_bin
        return getattr(pil_augs, op)(img, real_mag)


class AugmentationPolicyRandAugment(torch.nn.Module):
    def __init__(self, augmentations, n, m):
        super(AugmentationPolicyRandAugment, self).__init__()

        self.mag_bins = 10
        self.prob_bins = 10
        self.n = n
        self.m = m
        self.augmentations = augmentations
        self.search_space = PBASearchSpace(
            mag_bins=self.mag_bins, prob_bins=self.prob_bins)

    def __call__(self, img):
        ops = np.random.permutation(list(self.augmentations))[:self.n]
        for op in ops:
            img = self.search_space.apply_op(img, op, self.m)

        return img

class AugmentationPolicyPBA(torch.nn.Module):
    def __init__(self, augmentations, n_ops_probs):
        super(AugmentationPolicyPBA, self).__init__()
        
        self.mag_bins = 10
        self.prob_bins = 10
        self.n_ops_probs = n_ops_probs
        self.augmentations = augmentations
        self.search_space = PBASearchSpace(mag_bins=self.mag_bins, prob_bins=self.prob_bins)
        
    def __call__(self, img):
        N = get_prob_from_tree(self.n_ops_probs)
        ops = random.choices(self.augmentations, k=N)
        for op, prob, mag in ops:
            if np.random.uniform(0,1) <= prob:
                img = self.search_space.apply_op(img, op, mag)

        return img

class MyPILTransformsCompose(torch.nn.Module):
    def __init__(self, augmentations_list):
        super(MyPILTransformsCompose, self).__init__()
        self.augmentations_list = augmentations_list

    def __call__(self, img):
        for i, t in enumerate(self.augmentations_list):
            img = t(img)
        return img
