import keras
import tensorflow as tf
from .utils import shape_list, split_heads_2d, combine_heads_2d, rel_to_abs, relative_logits_1d
import math


class sa_conv2d(keras.layers.Layer):

    def __init__(self, filters, kernel_size,
                 stride=1, padding=0, groups=1, bias=True):
        super(sa_conv2d, self).__init__()
        #if out_channels % groups != 0:
        #    raise ValueError('out_channels must be divisible by groups')
        self.out_channels = filters
        self.kernel_size = kernel_size
        self.strides = stride
        self.padding = (kernel_size - 1) // 2
        self.groups = groups # multi-head count
        self.out_filters = filters // self.groups

        self.rel_size = (filters // groups) // 2
        self.relative_x = self.add_weight(name='rel_x',
                                      shape=(self.rel_size, self.kernel_size),
                                      initializer='normal',
                                      trainable=True)
        self.relative_y = self.add_weight(name='rel_y',
                                          shape=((filters // groups) - self.rel_size, self.kernel_size),
                                          initializer='normal',
                                          trainable=True)

        if bias:
            maxval = 1 /math.sqrt(self.out_channels)
            self.bias = self.add_weight(shape=(1, 1, 1, filters), initializer=tf.random_normal_initializer(mean=0.0, stddev=maxval), trainable=True)
        else:
            self.bias = None

    def call(self, inputs):

        #in_shape = tf.shape(input=inputs)
        in_shape = inputs.get_shape().as_list()
        sa_convolutions = []
        g_filters = in_shape[3] // self.groups
        kh, kw = self.kernel_size, self.kernel_size
        #ph, pw = in_shape[1] + self.padding * 2, in_shape[2] + self.padding * 2
        #print(in_shape[1], in_shape[2])
        ph, pw = in_shape[1], in_shape[2]
        #wh = (ph - kh) // self.stride + 1
        #ww = (pw - kw) // self.stride + 1
        wh = ph // self.strides
        ww = pw // self.strides
        #inputs = tf.pad(inputs, tf.constant([[0, 0], [self.padding, self.padding], [self.padding, self.padding], [0, 0]]))

        vanDamme = keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=self.groups, axis=3))(inputs)
        for sub_inp in vanDamme:
            # Slicing the ith channel:
            #sub_inp = tf.keras.layers.Lambda(lambda x: x[:, :, :, int(i * g_filters):int((i + 1) * g_filters)])(inputs)

            queries = keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, use_bias=False)(sub_inp)
            keys = keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, use_bias=False)(sub_inp)
            vals = keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, use_bias=False)(sub_inp)

            #que = queries[:, (kh - 1) // 2:ph - (kh // 2):self.stride, (kw - 1) // 2:pw - (kw // 2):self.stride, :]
            #que_b = tf.reshape(queries, [wh, ww, self.out_filters])
            que_b = keras.layers.Reshape((wh, ww, -1))(queries)

            #que_x, que_y = tf.split(que_b, num_or_size_splits=[self.rel_size, self.rel_size], axis=3)
            que_x_y = keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=3))(que_b)
            que_x, que_y = que_x_y

            #que_x = tf.einsum('bhwx,xk->bhwk', que_x, self.relative_x)
            #que_y = tf.einsum('bhwy,yk->bhwk', que_y, self.relative_y)
            que_x = keras.layers.Lambda(lambda x: tf.einsum('bhwx,xk->bhwk', x[0], x[1]))([que_x, self.relative_x])
            que_y = keras.layers.Lambda(lambda x: tf.einsum('bhwy,yk->bhwk', x[0], x[1]))([que_y, self.relative_y])

            #ext_keys = tf.image.extract_patches(keys, sizes=[1, kh, kh, 1], strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1],padding='SAME')
            ext_keys = keras.layers.Lambda(lambda x: tf.image.extract_patches(x, sizes=[1, kh, kh, 1], strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1], padding='SAME'))(keys)
            #ext_keys = tf.reshape(ext_keys, [wh, ww, self.out_filters, kh, kw])
            #ext_keys = tf.keras.layers.Reshape((wh, ww, self.out_filters, kh, kw))(ext_keys)
            ext_keys = keras.layers.Reshape((wh, ww, self.out_filters, kh, kw))(ext_keys)

            ext_keys = keras.backend.sum(keras.backend.expand_dims(keras.backend.expand_dims(que_b, axis=4), axis=4) * ext_keys, axis=3)  # b, fh, fw, kh, kw

            ext_keys = ext_keys + keras.backend.expand_dims(que_x, axis=3) + keras.backend.expand_dims(que_y, axis=4)
            #ext_keys = tf.reshape(ext_keys, [wh, ww, -1])
            ext_keys = keras.layers.Reshape((wh, ww, -1))(ext_keys)
            ext_keys = keras.layers.Softmax(axis=-1)(ext_keys)
            #ext_keys = tf.reshape(ext_keys, [wh, ww, 1, kh, kw])
            ext_keys = keras.layers.Reshape((wh, ww, 1, kh, kw))(ext_keys)
            #ext_vals = tf.image.extract_patches(vals, sizes=[1, kh, kh, 1], strides=[1, self.stride, self.stride, 1], rates=[1, 1, 1, 1], padding='SAME')
            ext_vals = keras.layers.Lambda(lambda x: tf.image.extract_patches(x, sizes=[1, kh, kh, 1], strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1], padding='SAME'))(vals)

            #ext_vals = tf.reshape(ext_vals, [wh, ww, self.out_filters, kh, kw])
            ext_vals = keras.layers.Reshape((wh, ww, self.out_filters, kh, kw))(ext_vals)

            #attn_out = tf.einsum('bhwckl->bhwc', ext_keys * ext_vals,)
            attn_out = keras.layers.Lambda(lambda x: tf.einsum('bhwckl->bhwc', x[0] * x[1],))([ext_keys, ext_vals])

            sa_convolutions.append(attn_out)

        sa_conv = keras.layers.concatenate(sa_convolutions, axis=3)
        if self.bias is not None:
            sa_conv += self.bias

        return sa_conv

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.out_channels)


