from __future__ import division
import numpy as np
from numpy import random
from scipy import ndimage, signal
import cv2
import pyfastnoisesimd as fns
import imgaug.augmenters as iaa

from .transform import change_transform_origin


def read_image_bgr(path):
    image = cv2.imread(path, -1)
    # image = np.asarray(Image.open(path).convert('RGB'))
    #return image[:, :, ::-1].copy()
    return image.copy()


def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def rgb_augmentation(image):
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.2))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.4, 2.3))),
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.3)),
        iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
        iaa.Invert(0.5, per_channel=0.3)], random_order=True)
    return seq.augment_image(image)


def depth_augmentation(image):
    # assumes meters
    image1 = image[:, :, 0]
    mask1 = image[:, :, 2]
    image1 = image1.astype('float32')
    blurK = np.random.choice([3, 5, 7], 1).astype(int)
    blurS = random.uniform(0.0, 1.5)
    shadowClK = np.random.choice([3, 5, 7], 1).astype(int)
    shadowMK = np.random.choice([3, 5, 7], 1).astype(int)

    partmask = np.where(mask1 > 0, 255.0, 0.0)
    kernel = np.ones((shadowClK[0], shadowClK[0]))
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=shadowMK[0])
    partmask = partmask.astype(np.uint8)
    mask = partmask > 0
    image1 = np.where(mask, image1, 0.0)

    image1 = cv2.resize(image1, None, fx=1 / 2, fy=1 / 2)
    res = (((image1 / 1000.0) * 1.41421356) ** 2)
    image1 = cv2.GaussianBlur(image1, (blurK, blurK), blurS, blurS)
    # quantify to depth resolution and apply gaussian
    dNonVar = np.divide(image1, res, out=np.zeros_like(image1), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)
    noise = np.multiply(dNonVar, random.uniform(0.002, 0.004))  # empirically determined
    image1 = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)
    image = cv2.resize(image1, (image.shape[1], image.shape[0]))

    # fast perlin noise
    seed = np.random.randint(2 ** 31)
    N_threads = 4
    perlin = fns.Noise(seed=seed, numWorkers=N_threads)
    drawFreq = random.uniform(0.05, 0.2)  # 0.05 - 0.2
    # drawFreq = 0.5
    perlin.frequency = drawFreq
    perlin.noiseType = fns.NoiseType.SimplexFractal
    perlin.fractal.fractalType = fns.FractalType.FBM
    drawOct = [4, 8]
    freqOct = np.bincount(drawOct)
    rndOct = np.random.choice(np.arange(len(freqOct)), 1, p=freqOct / len(drawOct), replace=False)
    # rndOct = 8
    perlin.fractal.octaves = rndOct
    perlin.fractal.lacunarity = 2.1
    perlin.fractal.gain = 0.45
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb

    noiseX = np.random.uniform(0.001, 0.01, image.shape[1] * image.shape[0])  # 0.0001 - 0.1
    noiseY = np.random.uniform(0.001, 0.01, image.shape[1] * image.shape[0])  # 0.0001 - 0.1
    noiseZ = np.random.uniform(0.01, 0.1, image.shape[1] * image.shape[0])  # 0.01 - 0.1
    Wxy = np.random.randint(1, 5)  # 1 - 5
    Wz = np.random.uniform(0.0001, 0.004)  # 0.0001 - 0.004

    X, Y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    coords0 = fns.empty_coords(image.shape[1] * image.shape[0])
    coords1 = fns.empty_coords(image.shape[1] * image.shape[0])
    coords2 = fns.empty_coords(image.shape[1] * image.shape[0])

    coords0[0, :] = noiseX.ravel()
    coords0[1, :] = Y.ravel()
    coords0[2, :] = X.ravel()
    VecF0 = perlin.genFromCoords(coords0)
    VecF0 = VecF0.reshape((image.shape[0], image.shape[1]))

    coords1[0, :] = noiseY.ravel()
    coords1[1, :] = Y.ravel()
    coords1[2, :] = X.ravel()
    VecF1 = perlin.genFromCoords(coords1)
    VecF1 = VecF1.reshape((image.shape[0], image.shape[1]))

    coords2[0, :] = noiseZ.ravel()
    coords2[1, :] = Y.ravel()
    coords2[2, :] = X.ravel()
    VecF2 = perlin.genFromCoords(coords2)
    VecF2 = VecF2.reshape((image.shape[0], image.shape[1]))

    x = np.arange(image.shape[1], dtype=np.uint16)
    x = x[np.newaxis, :].repeat(image.shape[0], axis=0)
    y = np.arange(image.shape[0], dtype=np.uint16)
    y = y[:, np.newaxis].repeat(image.shape[1], axis=1)

    Wxy_scaled = image * 0.001 * Wxy
    Wz_scaled = image * 0.001 * Wz
    # scale with depth
    fx = x + Wxy_scaled * VecF0
    fy = y + Wxy_scaled * VecF1
    fx = np.where(fx < 0, 0, fx)
    fx = np.where(fx >= image.shape[1], image.shape[1] - 1, fx)
    fy = np.where(fy < 0, 0, fy)
    fy = np.where(fy >= image.shape[0], image.shape[0] - 1, fy)
    fx = fx.astype(dtype=np.uint16)
    fy = fy.astype(dtype=np.uint16)
    image = image[fy, fx] + Wz_scaled * VecF2
    image = np.where(image > 0, image, 0.0)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    image = np.multiply(image, 255.0 / np.nanmax(image))

    return image


def adjust_transform_for_image(transform, image, relative_translation):
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:

    def __init__(
            self,
            fill_mode='nearest',
            interpolation='linear',
            cval=0,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):

    #image = depth_augmentation(image)
    #image = rgb_augmentation(image)
    image = image.astype('float32')
    image = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )
    return image


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    scale = min_side / smallest_side

    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
