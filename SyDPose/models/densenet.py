
import keras
from keras.applications import densenet
from keras.utils import get_file

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


allowed_backbones = {
    'densenet121': ([6, 12, 24, 16], densenet.DenseNet121),
    'densenet169': ([6, 12, 32, 32], densenet.DenseNet169),
    'densenet201': ([6, 12, 48, 32], densenet.DenseNet201),
}


class DenseNetBackbone(Backbone):

    def retinanet(self, *args, **kwargs):

        return densenet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):

        origin    = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/'
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'

        # load weights
        if keras.backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_first" format are not available.')

        weights_url = origin + file_name.format(self.backbone)
        return get_file(file_name.format(self.backbone), weights_url, cache_subdir='models')

    def validate(self):

        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones.keys()))

    def preprocess_image(self, inputs):

        return preprocess_image(inputs, mode='tf')


def densenet_retinanet(num_classes, backbone='densenet121', inputs=None, modifier=None, **kwargs):

    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    blocks, creator = allowed_backbones[backbone]
    model = creator(input_tensor=inputs, include_top=False, pooling=None, weights=None)

    layer_outputs = [model.get_layer(name='conv{}_block{}_concat'.format(idx + 2, block_num)).output for idx, block_num in enumerate(blocks)]

    model = keras.models.Model(inputs=inputs, outputs=layer_outputs[1:], name=model.name)

    if modifier:
        model = modifier(model)

    model = retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=model.outputs, **kwargs)

    return model
