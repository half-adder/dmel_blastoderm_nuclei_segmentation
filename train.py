from __future__ import print_function, unicode_literals, absolute_import, division

import sys
import numpy as np
import albumentations as A

from tqdm import tqdm
from glob import glob

from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import calculate_extents, fill_label_holes
from stardist.models import Config2D, StarDist2D

from datetime import datetime


# Read Training Data

X = sorted(glob('data/img/*.tif'))
Y = sorted(glob('data/labels/*.tif'))

X = np.concatenate(list(map(imread,X)))
Y = np.concatenate(list(map(imread,Y)))

axis_norm = (0, 1)

X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

# Select training and validation data at random

assert len(X) > 1, "not enough training data"

rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]

X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 

print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))


# Set up model

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# This is only about generating training data, the training will use the GPU if run on the gpu node
use_gpu = False

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (4,4)

conf = Config2D(
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = 1,
    train_patch_size = (96,96),
    train_epochs=600,
    train_steps_per_epoch=400,
)

# Name of the model corresponds to the folder that the model is stored in

now_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

model = StarDist2D(conf, name=now_time, basedir='models')

# %%
# median_size = calculate_extents(list(Y), np.median)
# fov = np.array(model._axes_tile_overlap('YX'))
# print(f"median object size:      {median_size}")
# print(f"network field of view :  {fov}")
# if any(median_size > fov):
    # print("WARNING: median object size larger than field of view of the neural network.")


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.GaussNoise(
            var_limit=(0, 0.05),  # ScaleFloatType
            mean=0.0,  # float
            per_channel=True,  # bool
            noise_scale_factor=1,  # float
            always_apply=None,  # bool | None
            p=1.0,  # float
        ),
        A.Rotate(
            limit=(-90, 90),  # ScaleFloatType
            interpolation=2,  # <class 'int'>
            border_mode=4,  # int
            value=None,  # ColorType | None
            mask_value=None,  # ColorType | None
            rotate_method="largest_box",  # Literal['largest_box', 'ellipse']
            crop_border=False,  # bool
            always_apply=None,  # bool | None
            p=1.0,  # float
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    ])
    
    augmented = transform(image=x, mask=y)
    x_aug = augmented['image']
    y_aug = augmented['mask']
    
    return x_aug, y_aug


# Train and optimize model

model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

model.optimize_thresholds(X_val, Y_val)

# model.export_TF("2024-09-24_stardist-model_blastoderm-nuclei.zip")
