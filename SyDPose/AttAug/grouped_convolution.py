import keras
import tensorflow as tf
from .utils import shape_list, split_heads_2d, combine_heads_2d, rel_to_abs, relative_logits_1d


def create_conv_2d(input, group_filters, kernel, activation):

    return keras.layers.Conv2D(group_filters, kernel_size=kernel, activation=activation, strides=1, padding='same')(input)


def grouped_convolution_2d(input, filters=256, kernel=3, activation='linear', groups=8, name=None, residual=True):

    skip_input = keras.layers.Conv2D(filters=filters, kernel_size=1, activation=activation)(input)

    grouped_convolutions = []
    group_filters = filters // groups

    #sub_groups = [group_filters] * groups

    #sub_groups = tf.split(output, groups, 3)
    #for sub_inp in sub_groups:
    for i in range(groups):
        # Slicing the ith channel:
        sub_inp = keras.layers.Lambda(lambda x: x[:, :, :, int(i*group_filters):int((i+1)*group_filters)])(skip_input)

        sub_conv = create_conv_2d(sub_inp, group_filters, kernel, activation)
        grouped_convolutions.append(sub_conv)

    output = keras.layers.concatenate(grouped_convolutions, axis=3)

    output = keras.layers.Conv2D(filters=filters, kernel_size=1, activation=activation, name=name)(output)

    if residual:
        output = keras.layers.Add()([output, skip_input])

    if name:
        output._name = name

    return output
