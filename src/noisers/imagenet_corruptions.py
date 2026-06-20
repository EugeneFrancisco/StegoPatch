"""
Self-contained ImageNet-C corruptions, adapted from Hendrycks & Dietterich
(https://github.com/hendrycks/robustness) -- the same corruptions RoSteALS
(Bui et al.) trains on.

The original `imagenet_c` package pulls in Wand / ImageMagick at the system
level, which is painful to install (especially on a fresh Modal image). We only
need the corruptions RoSteALS actually trains with, none of which require
ImageMagick, so we reimplement them here on top of numpy / scipy / opencv /
scikit-image / Pillow. The ImageMagick-backed corruptions (motion/zoom/glass
blur, snow) and the asset-dependent one (frost) are intentionally omitted.

Public API mirrors `imagenet_c.corrupt`:

    corrupt(img_uint8_hwc_rgb, severity=1, corruption_number=0) -> uint8 HxWx3
"""
# pylint: skip-file
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import color as sk_color
from skimage.util import random_noise


# -- helpers -----------------------------------------------------------------
def _disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= aliased_disk.sum()
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _plasma_fractal(mapsize=256, wibbledecay=3.0):
    """Generate a heightmap (in [0, 1]) using the diamond-square algorithm."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


# -- corruptions -------------------------------------------------------------
# Each takes an HxWx3 uint8 RGB array + severity in [1, 5] and returns a float
# array in [0, 255]; `corrupt` below clips and casts back to uint8.
def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = x / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = x / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = random_noise(x / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = x / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian_filter(x / 255., sigma=(c, c, 0))
    return np.clip(x, 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = x / 255.
    kernel = _disk(radius=c[0], alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    x = x / 255.
    h, w = x.shape[:2]
    max_val = x.max()
    fractal = _plasma_fractal(mapsize=_next_pow2(max(h, w)), wibbledecay=c[1])
    x += c[0] * fractal[:h, :w][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = sk_color.rgb2hsv(x / 255.)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk_color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = x / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = sk_color.rgb2hsv(x / 255.)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk_color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    out = BytesIO()
    Image.fromarray(x).save(out, 'JPEG', quality=c)
    return np.asarray(Image.open(out)).astype(np.float32)


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    h, w = x.shape[:2]
    img = Image.fromarray(x)
    img = img.resize((max(int(w * c), 1), max(int(h * c), 1)), Image.BOX)
    img = img.resize((w, h), Image.BOX)
    return np.asarray(img).astype(np.float32)


def elastic_transform(x, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
    image = np.array(x, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine warp
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2) 
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # random elastic displacement field
    dx = (gaussian_filter(np.random.uniform(-1, 1, size=shape[:2]),
                          c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian_filter(np.random.uniform(-1, 1, size=shape[:2]),
                          c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    xx, yy, zz = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (np.reshape(yy + dy, (-1, 1)),
               np.reshape(xx + dx, (-1, 1)),
               np.reshape(zz, (-1, 1)))
    return np.clip(
        map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1
    ) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian_filter(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:  # water droplets
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor((liquid_layer * dist).astype(np.float32), cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:  # mud splatter
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian_filter(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)
        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])
        return np.clip(x + color, 0, 1) * 255


# Corruption ids follow the original imagenet_c numbering so existing configs
# keep working. Omitted ids: 4 glass_blur, 5 motion_blur, 6 zoom_blur, 7 snow
# (ImageMagick) and 8 frost (needs bundled frost images).
_CORRUPTIONS = {
    0: gaussian_noise,
    1: shot_noise,
    2: impulse_noise,
    3: defocus_blur,
    9: fog,
    10: brightness,
    11: contrast,
    12: elastic_transform,
    13: pixelate,
    14: jpeg_compression,
    15: speckle_noise,
    16: gaussian_blur,
    17: spatter,
    18: saturate,
}

# Default training set: everything we support here.
DEFAULT_CORRUPTION_IDS = sorted(_CORRUPTIONS)


def corrupt(x, severity=1, corruption_number=0):
    """Apply one corruption to an HxWx3 uint8 RGB image; returns uint8 HxWx3."""
    if corruption_number not in _CORRUPTIONS:
        raise ValueError(
            f"Unsupported corruption_number {corruption_number}; "
            f"available: {DEFAULT_CORRUPTION_IDS}"
        )
    out = _CORRUPTIONS[corruption_number](np.asarray(x), severity)
    return np.clip(np.asarray(out), 0, 255).astype(np.uint8)
