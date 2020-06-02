
import numpy as np
import random
import warnings
import copy
import cv2

import keras

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from ..utils.config import parse_anchor_parameters
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb, transform_box3d


class Generator(keras.utils.Sequence):

    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config=None
    ):

        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.config                 = config

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):

        raise NotImplementedError('size method not implemented')

    def num_classes(self):

        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):

        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):

        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):

        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):

        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):

        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):

        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):

        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):

        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('poses' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('segmentations' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):

        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
                #(annotations['segmentations'][:, 0] < 0) |
                #(annotations['segmentations'][:, 0] > image.shape[1]) |
                #(annotations['segmentations'][:, 1] < 0) |
                #(annotations['segmentations'][:, 1] > image.shape[0]) |
                #(annotations['segmentations'][:, 2] < 0) |
                #(annotations['segmentations'][:, 2] > image.shape[1]) |
                #(annotations['segmentations'][:, 3] < 0) |
                #(annotations['segmentations'][:, 3] > image.shape[0]) |
                #(annotations['segmentations'][:, 4] < 0) |
                #(annotations['segmentations'][:, 4] > image.shape[1]) |
                #(annotations['segmentations'][:, 5] < 0) |
                #(annotations['segmentations'][:, 5] > image.shape[0]) |
                #(annotations['segmentations'][:, 6] < 0) |
                #(annotations['segmentations'][:, 6] > image.shape[1]) |
                #(annotations['segmentations'][:, 7] < 0) |
                #(annotations['segmentations'][:, 7] > image.shape[0]) |
                #(annotations['segmentations'][:, 8] < 0) |
                #(annotations['segmentations'][:, 8] > image.shape[1]) |
                #(annotations['segmentations'][:, 9] < 0) |
                #(annotations['segmentations'][:, 9] > image.shape[0]) |
                #(annotations['segmentations'][:, 10] < 0) |
                #(annotations['segmentations'][:, 10] > image.shape[1]) |
                #(annotations['segmentations'][:, 11] < 0) |
                #(annotations['segmentations'][:, 11] > image.shape[0]) |
                #(annotations['segmentations'][:, 12] < 0) |
                #(annotations['segmentations'][:, 12] > image.shape[1]) |
                #(annotations['segmentations'][:, 13] < 0) |
                #(annotations['segmentations'][:, 13] > image.shape[0]) |
                #(annotations['segmentations'][:, 14] < 0) |
                #(annotations['segmentations'][:, 14] > image.shape[1]) |
                #(annotations['segmentations'][:, 15] < 0) |
                #(annotations['segmentations'][:, 15] > image.shape[0])
            )[0]

            if len(invalid_indices):
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):

        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):

        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            annotations['segmentations'] = annotations['segmentations'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])
                annotations['segmentations'][index, :] = transform_box3d(transform, annotations['segmentations'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):

        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):

        image = self.preprocess_image(image)

        image, image_scale = self.resize_image(image)

        annotations['bboxes'] *= image_scale
        annotations['segmentations'] *= image_scale
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def group_images(self):

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):

        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):

        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def compute_input_output(self, group):

        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        inputs = self.compute_inputs(image_group)

        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):

        return len(self.groups)

    def __getitem__(self, index):

        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets
