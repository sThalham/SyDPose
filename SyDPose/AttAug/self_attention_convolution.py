import keras
import tensorflow as tf
from .utils import shape_list, split_heads_2d, combine_heads_2d, rel_to_abs, relative_logits_1d


def relative_logits(q, H, W, Nh, dkh):
    """Compute relative logits."""
    # Relative logits in width dimension first.
    init_w = tf.random_normal_initializer(stddev=dkh ** -0.5)
    print(init_w)
    rel_embeddings_w = tf.Variable(lambda: init_w(shape=(2 * W - 1, dkh)))
    #rel_embeddings_w = tf.compat.v1.get_variable('r_width', shape=(2 * W - 1, dkh), initializer=tf.compat.v1.random_normal_initializer(dkh**-0.5))
    # [B, Nh, HW, HW]
    rel_logits_w = relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
    # Relative logits in height dimension next.
    # For ease, we 1) transpose height and width,
    # 2) repeat the above steps and
    # 3) transpose to eventually put the logits
    # in their right positions.
    #rel_embeddings_h = tf.constant(tf.compat.v1.random_normal_initializer(dkh ** -0.5), dtype=tf.float64, shape=(2 * H - 1, dkh))
    rel_embeddings_h = tf.Variable(keras.initializers.RandomNormal(dkh**-0.5), shape=(2 * H - 1, dkh), validate_shape=False)
    #rel_embeddings_h = tf.compat.v1.get_variable('r_height', shape=(2 * H - 1, dkh), initializer=tf.compat.v1.random_normal_initializer(dkh ** -0.5))
    # [B, Nh, HW, HW]
    rel_logits_h = relative_logits_1d(tf.transpose(a=q, perm=[0, 1, 3, 2, 4]),
    rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
    return rel_logits_h, rel_logits_w


def sa_conv2d(inputs, filters, kernel, activation='linear', groups=8, relative=True):
    """2d relative self−attention."""
    _, H, W, _ = shape_list(inputs)
    sa_convolutions = []
    g_filters = filters // groups

    for i in range(groups):
        # Slicing the ith channel:
        sub_inp = keras.layers.Lambda(lambda x: x[:, :, :, int(i * g_filters):int((i+1) * g_filters)])(inputs)

        queries = keras.layers.Conv2D(filters=g_filters, kernel_size=1)(sub_inp)
        keys = keras.layers.Conv2D(filters=g_filters, kernel_size=kernel, activation=activation)(sub_inp)
        flat_q = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, H * W, g_filters]))(queries)
        flat_k = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, H * W, g_filters]))(keys)
        logits = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([flat_q, flat_k])
        #if relative:
        #    rel_logits_h, rel_logits_w = relative_logits(queries, H, W, g_filters)
        #    logits += rel_logits_h
        #    logits += rel_logits_w
        weights = keras.layers.Lambda(lambda x: tf.nn.softmax(x))(logits)
        values = keras.layers.Conv2D(filters=g_filters, kernel_size=kernel, activation=activation)(sub_inp)
        flat_v = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, H * W, g_filters]))(values)
        attn_out = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=False))([weights, flat_v])
        attn_out = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, H, W, g_filters]))(attn_out)

        sa_convolutions.append(attn_out)

    output = keras.layers.concatenate(sa_convolutions, axis=3)

    return output

    #flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])
    # Compute q, k, v
    #kqv = keras.layers.Conv2D(filters=2 * dk + dv, kernel_size=1)(inputs)
    #k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
    #q *= dkh ** -0.5 # scaled dot−product
    # After splitting, shape is [B, Nh, H, W, dkh or dvh]
    #q = split_heads_2d(q, Nh)
    #k = split_heads_2d(k, Nh)
    #v = split_heads_2d(v, Nh)
    # [B, Nh, HW, HW]
    #logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)
    #if relative:
    #    rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh, dkh)
    #    logits += rel_logits_h
    #    logits += rel_logits_w
    #weights = tf.nn.softmax(logits)
    #attn_out = tf.matmul(weights, flatten_hw(v, dvh))
    #attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
    #attn_out = combine_heads_2d(attn_out)
    # Project heads and mix contribution
    #attn_out = keras.layers.Conv2D(filters=dv, kernel_size=1)(attn_out)
    #return attn_out
