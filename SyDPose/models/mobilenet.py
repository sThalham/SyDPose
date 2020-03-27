
import keras
from keras.applications import mobilenet
from keras.utils import get_file
from ..utils.image import preprocess_image

from . import retinanet
from . import Backbone


class MobileNetBackbone(Backbone):

    allowed_backbones = ['mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224']

    def retinanet(self, *args, **kwargs):

        return mobilenet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):

        alpha = float(self.backbone.split('_')[1])
        rows = int(self.backbone.split('_')[0].replace('mobilenet', ''))

        # load weights
        if keras.backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        model_name = 'mobilenet_{}_{}_tf_no_top.h5'.format(alpha_text, rows)
        weights_url = mobilenet.mobilenet.BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weights_url, cache_subdir='models')

        return weights_path

    def validate(self):

        backbone = self.backbone.split('_')[0]

        if backbone not in MobileNetBackbone.allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetBackbone.allowed_backbones))

    def preprocess_image(self, inputs):

        return preprocess_image(inputs, mode='tf')


def mobilenet_retinanet(num_classes, backbone='mobilenet224_1.0', inputs=None, modifier=None, **kwargs):

    alpha = float(backbone.split('_')[1])

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    backbone = mobilenet.MobileNet(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights=None)

    # create the full model
    layer_names = ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)
