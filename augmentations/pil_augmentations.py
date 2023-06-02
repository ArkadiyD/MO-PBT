# this code is adpoted from https://github.com/automl/trivialaugment

from torchvision.transforms import RandomResizedCrop
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import random
from dataclasses import dataclass
from typing import Union


@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    shear: MinMax = MinMax(.0, .3)
    translate: MinMax = MinMax(0, 10)  # different from uniaug: MinMax(0,14.4)
    rotate: MinMax = MinMax(0, 30)
    solarize: MinMax = MinMax(0, 256)
    posterize: MinMax = MinMax(0, 4)  # different from uniaug: MinMax(4,8)
    enhancer: MinMax = MinMax(.1, 1.9)
    cutout: MinMax = MinMax(.0, .2)
    randomresizedcrop: MinMax = MinMax(0.08, 1.0)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

Returns:
    A float that results from scaling `maxval` according to `level`.
"""
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

Returns:
    An int that results from scaling `maxval` according to `level`.
"""
    return int(level * maxval / PARAMETER_MAX)


################## Transform Functions ##################
def identity(pil_img, level):
    return pil_img


def auto_contrast(pil_img, level):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, level):
    return ImageOps.equalize(pil_img)


def invert(pil_img, level):
    return ImageOps.invert(pil_img)


def blur(pil_img, level):
    pil_img = pil_img.filter(ImageFilter.BLUR)
    return pil_img


def smooth(pil_img, level):
    pil_img = pil_img.filter(ImageFilter.SMOOTH)
    return pil_img


def rotate(pil_img, level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, min_max_vals.rotate.max)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


def posterize(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(
        level, min_max_vals.posterize.max - min_max_vals.posterize.min)
    return ImageOps.posterize(pil_img, min_max_vals.posterize.max - level)


def shear_x(pil_img, level):
    """Applies PIL ShearX to `pil_img`.

The ShearX operation shears the image along the horizontal axis with `level`
magnitude.

Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

Returns:
    A PIL Image that has had ShearX applied to it.
"""
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def shear_y(pil_img, level):
    """Applies PIL ShearY to `pil_img`.

The ShearY operation shears the image along the vertical axis with `level`
magnitude.

Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

Returns:
    A PIL Image that has had ShearX applied to it.
"""
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def translate_x(pil_img, level):
    """Applies PIL TranslateX to `pil_img`.

Translate the image in the horizontal direction by `level`
number of pixels.

Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

Returns:
    A PIL Image that has had TranslateX applied to it.
"""
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def translate_y(pil_img, level):
    """Applies PIL TranslateY to `pil_img`.

Translate the image in the vertical direction by `level`
number of pixels.

Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

Returns:
    A PIL Image that has had TranslateY applied to it.
"""
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def solarize(pil_img, level):
    """Applies PIL Solarize to `pil_img`.

Translate the image in the vertical direction by `level`
number of pixels.

Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

Returns:
    A PIL Image that has had Solarize applied to it.
"""
    level = int_parameter(level, min_max_vals.solarize.max)
    return ImageOps.solarize(pil_img, 256 - level)


def color(pil_img, level):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
    enhancer = ImageEnhance.Color
    mini = min_max_vals.enhancer.min
    maxi = min_max_vals.enhancer.max
    v = float_parameter(level, maxi - mini) + \
        mini  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)


def contrast(pil_img, level):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
    enhancer = ImageEnhance.Contrast
    mini = min_max_vals.enhancer.min
    maxi = min_max_vals.enhancer.max
    v = float_parameter(level, maxi - mini) + \
        mini  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)


def brightness(pil_img, level):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
    enhancer = ImageEnhance.Brightness
    mini = min_max_vals.enhancer.min
    maxi = min_max_vals.enhancer.max
    v = float_parameter(level, maxi - mini) + \
        mini  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)


def sharpness(pil_img, level):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
    enhancer = ImageEnhance.Sharpness
    mini = min_max_vals.enhancer.min
    maxi = min_max_vals.enhancer.max
    v = float_parameter(level, maxi - mini) + \
        mini  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)


def gaussian(pil_img, level):
    pil_img = pil_img.filter(ImageFilter.GaussianBlur)
    return pil_img


def hflip(pil_img, level):
    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    return pil_img


def vflip(pil_img, level):
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    return pil_img

class TensorCutout(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, prob, level):
        self.min_max_vals = min_max_vals
        self.prob = prob
        self.level = level

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.prob:
            return img

        value = int_parameter(self.level, 20)

        if value == 0:
            return img

        h, w = img.size(1), img.size(2)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - value // 2, 0, h)
        y2 = np.clip(y + value // 2, 0, h)
        x1 = np.clip(x - value // 2, 0, w)
        x2 = np.clip(x + value // 2, 0, w)

        img[:, y1: y2, x1: x2] = 0.
        return img


def blend(img1, v):
    if blend_images is None:
        print("please set google_transformations.blend_images before using the enlarged_randaug search space.")
    i = np.random.choice(len(blend_images))
    img2 = blend_images[i]
    m = float_parameter(v, .4)
    return Image.blend(img1, img2, m)


def randomresizedcrop(img, level):
    lb = float_parameter(level, min_max_vals.randomresizedcrop.max -
                         min_max_vals.randomresizedcrop.min)+min_max_vals.randomresizedcrop.min
    ub = 1.0
    tr = RandomResizedCrop(size=img.size, scale=(lb, ub))
    return tr(img)


def set_augmentation_space(augmentation_space, num_strengths, custom_augmentation_space_augs=None):
    global ALL_TRANSFORMS, min_max_vals, PARAMETER_MAX
    assert num_strengths > 0
    PARAMETER_MAX = num_strengths - 1
    if augmentation_space == 'wide':
        min_max_vals = MinMaxVals(
            shear=MinMax(.0, .99),
            translate=MinMax(0, 32),
            rotate=MinMax(0, 135),
            solarize=MinMax(0, 256),
            posterize=MinMax(2, 8),
            enhancer=MinMax(.01, 2.),
            cutout=MinMax(.0, .6),
        )
    elif ('uniaug' in augmentation_space) or ('randaug' in augmentation_space):
        min_max_vals = MinMaxVals(
            posterize=MinMax(4, 8),
            translate=MinMax(0, 14.4)
        )
    elif augmentation_space == 'extra_wide':
        min_max_vals = MinMaxVals(
            shear=MinMax(.0, .99),
            translate=MinMax(0, 32),
            rotate=MinMax(0, 180),
            solarize=MinMax(0, 256),
            posterize=MinMax(2, 8),
            enhancer=MinMax(.01, 2.),
            cutout=MinMax(.0, .8),
            randomresizedcrop=MinMax(0.08, 1.0)
        )
    else:
        min_max_vals = MinMaxVals()

    ALL_TRANSFORMS = [
        shear_x,
        shear_y,
        translate_x,
        translate_y,
        rotate,
        auto_contrast,
        invert,
        equalize,
        solarize,
        posterize,
        contrast,
        color,
        brightness,
        sharpness
    ]


def apply_augmentation(aug_idx, m, img):
    return ALL_TRANSFORMS[aug_idx].pil_transformer(1., m)(img)


def num_augmentations():
    return len(ALL_TRANSFORMS)
