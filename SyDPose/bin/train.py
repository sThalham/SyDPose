#!/usr/bin/env python

import argparse
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import SyDPose.bin  # noqa: F401
    __package__ = "SyDPose.bin"

from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..models.retinanet import retinanet_bbox
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator


def makedirs(path):

    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):

    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):

    modifier = freeze_model if freeze_backbone else None

    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    training_model.compile(
        loss={
            'bbox'            : losses.smooth_l1(),
            '3Dbox'          : losses.orthogonal_l1(),
            'cls'            : losses.focal(),
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):

    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        elif args.dataset_type == 'linemod':
            from ..callbacks.linemod import LinemodEval
            evaluation = LinemodEval(validation_generator, tensorboard=tensorboard_callback)

        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

        # save the model
    if args.snapshots:
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):

    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.2, -0.2),
            max_translation=(0.2, 0.2),
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
        )

    if args.dataset_type == 'linemod':
        from ..preprocessing.linemod import LinemodGenerator

        train_generator = LinemodGenerator(
            args.linemod_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = LinemodGenerator(
            args.linemod_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )

    elif args.dataset_type == 'occlusion':
        from ..preprocessing.occlusion import OcclusionGenerator

        train_generator = OcclusionGenerator(
            args.occlusion_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = OcclusionGenerator(
            args.linemod_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )

    elif args.dataset_type == 'tless':
        from ..preprocessing.tless import TlessGenerator

        train_generator = TlessGenerator(
            args.tless_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = TlessGenerator(
            args.tless_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )
        train_iterations = len(os.listdir(os.path.join(args.tless_path, 'images/train')))

    elif args.dataset_type == 'ycbv':
        from ..preprocessing.ycbv import YCBVGenerator

        train_generator = YCBVGenerator(
            args.ycbv_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = YCBVGenerator(
            args.ycbv_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )

    elif args.dataset_type == 'fronius':
        from ..preprocessing.fronius import FroniusGenerator

        train_generator = FroniusGenerator(
            args.fronius_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = FroniusGenerator(
            args.fronius_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )

    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network with object pose estimation.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help='Path to dataset directory (ie. /tmp/linemod).')

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help='Path to dataset directory (ie. /tmp/occlusion.')

    tless_parser = subparsers.add_parser('tless')
    tless_parser.add_argument('tless_path', help='Path to dataset directory (ie. /tmp/tless).')

    ycbv_parser = subparsers.add_parser('ycbv')
    ycbv_parser.add_argument('ycbv_path', help='Path to dataset directory (ie. /tmp/ycbv).')

    fronius_parser = subparsers.add_parser('fronius')
    fronius_parser.add_argument('fronius_path', help='Path to dataset directory (ie. /tmp/fronius).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./models\')', default='./models')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_true')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=6400)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=3)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

    return parser.parse_args(args)


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    backbone = models.backbone(args.backbone)

    check_keras_version()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #tf.config.experimental.set_memory_growth

    if args.config:
        args.config = read_config_file(args.config)

    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=0,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config
        )

    print(model.summary())

    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size
    )


if __name__ == '__main__':
    main()
