#!/usr/bin/env python

import argparse
import os
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import RetNetPose.bin  # noqa: F401
    __package__ = "RetNetPose.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters


def get_session():

    config = tf.compat.v1.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return tf.compat.v1.Session(config=config)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Set modified tf session to avoid using the GPUs
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    anchor_parameters = None
    if args.config:
        args.config = read_config_file(args.config)
        if 'anchor_parameters' in args.config:
            anchor_parameters = parse_anchor_parameters(args.config)

    # load the model
    model = models.load_model(args.model_in, backbone_name=args.backbone)

    # check if this is indeed a training model
    models.check_training_model(model)

    # convert the model
    model = models.convert_model(model, nms=args.nms, class_specific_filter=args.class_specific_filter, anchor_params=anchor_parameters)

    # save model
    model.save(args.model_out)


if __name__ == '__main__':
    main()
