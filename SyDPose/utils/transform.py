
import numpy as np

DEFAULT_PRNG = np.random


def colvec(*args):

    return np.array([args]).T


def transform_aabb(transform, aabb):

    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def transform_box3d(transform, box3d):

    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8 = box3d
    points = transform.dot([
        [x1, x2, x3, x4, x5, x6, x7, x8],
        [y1, y2, y3, y4, y5, y6, y7, y8],
        [1,  1,  1,  1,  1,  1,  1,  1 ],
    ])
    return [points[0, 0], points[1, 0], points[0, 1], points[1, 1],points[0, 2], points[1, 2], points[0, 3], points[1, 3], points[0, 4], points[1, 4], points[0, 5], points[1, 5], points[0, 6], points[1, 6], points[0, 7], points[1, 7]]


def _random_vector(min, max, prng=DEFAULT_PRNG):

    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def rotation(angle):

    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(min, max, prng=DEFAULT_PRNG):

    return rotation(prng.uniform(min, max))


def translation(translation):

    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def random_translation(min, max, prng=DEFAULT_PRNG):

    return translation(_random_vector(min, max, prng))


def scaling(factor):

    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng=DEFAULT_PRNG):

    return scaling(_random_vector(min, max, prng))


def change_transform_origin(transform, center):

    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


def random_transform(
    min_rotation=-0.1,
    max_rotation=0.1,
    min_translation=(-0.2, -0.2),
    max_translation=(0.2, 0.2),
    min_scaling=(0.9, 0.9),
    max_scaling=(1.1, 1.1),
    prng=DEFAULT_PRNG
):

    return np.linalg.multi_dot([
        random_translation(min_translation, max_translation, prng),
        random_scaling(min_scaling, max_scaling, prng),
    ])


def random_transform_generator(prng=None, **kwargs):

    if prng is None:
        # RandomState automatically seeds using the best available method.
        prng = np.random.RandomState()

    while True:
        yield random_transform(prng=prng, **kwargs)
