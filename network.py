import tensorflow as tf
import numpy as np


def relu(x):
    return tf.nn.relu(x)


def lrelu(x, a=0.01):
    return a * (x-tf.abs(x))/2 + (tf.abs(x)+x)/2


def elu(x, a=1.0):
    return (tf.abs(x) + x)/2.0 + a*(tf.exp((x-tf.abs(x))/2) - 1)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def network(name, x, layers, is_training = True):
    tensors = {}
    with tf.variable_scope(name):
        tensors['input'] = x
        for (layer_name, prop) in layers:
            with tf.variable_scope(layer_name):
                shape = prop.get('reshape', None)
                input = prop.get('input', None)
                pad = prop.get('pad', 'SAME')
                batch = prop.get('batch', None)
                act = prop.get('act', None)
                concat = prop.get('concat', None)
                add = prop.get('add', None)
                if input is not None:
                    x = tensors[input]
                if shape is not None:
                    x = tf.reshape(x, x.get_shape().as_list()[:1] + shape)
                if layer_name[:2] == 'fc':
                    if batch is not None:
                        x = tf.contrib.layers.fully_connected(x, num_outputs=prop['n_out'], activation_fn=act,
                                                              normalizer_fn=tf.contrib.layers.batch_norm,
                                                              normalizer_params={'decay': 0.9,   # this works better than 0.999
                                                              'scale': True, 'is_training': is_training}
                                                              )
                    else:
                        x = tf.contrib.layers.fully_connected(x, num_outputs=prop['n_out'], activation_fn=act)
                elif layer_name[:4] == 'conv':
                    if batch is not None:
                        x = tf.contrib.layers.conv2d(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=act,
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params={'decay': 0.9,
                                                                        'scale': True, 'is_training': is_training}
                                                     )
                    else:
                        x = tf.contrib.layers.conv2d(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=act
                                                     )
                elif layer_name[:4] == 'cnvT':
                    if batch is not None:
                        x = tf.contrib.layers.conv2d_transpose(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=act,
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params={'decay': 0.9,
                                                                        'scale': True, 'is_training': is_training}
                                                     )
                    else:
                        x = tf.contrib.layers.conv2d_transpose(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=act
                                                     )
                tensors[layer_name] = x
                if concat is not None:
                    x = tf.concat([x, tensors[prop['concat']]], axis=(len(x.shape)-1))
                if add is not None:
                    x = x+tensors[prop['add']]

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
    moving_avgs = tf.get_collection(tf.GraphKeys.UPDATE_OPS,name)  # these are assigin nodes of moving_average -non trainable variables
    
    return x, tensors, vars, moving_avgs


if __name__ == "__main__":
    print("start checking networks")

    def residual(iter, input):
        residual_layer = []
        for i in range(iter):
            it = i+1
            residual_layer += [('res%d_conv1'%it, {'n_out': 64, 'k': 3, 's': 1, 'act': relu}),
                ('res%d_conv2'%it, {'input': input, 'n_out': 64, 'k': 1, 's': 1, 'act': relu}),
                ('res%d'%it, {'n_out': 64, 'k': 1, 's': 1, 'act': relu, 'concat': 'res%d_conv1'%it})]
        return residual_layer

    layers = [('fc1', {'n_out': 100, 'act': None}),                                                    # Nx100
                ('fc2', {'n_out': 16*16*3, 'act': relu, 'batch': True}),                              # Nx16x16x3
                ('conv3', {'reshape': [16, 16, 3], 'n_out': 64, 'k': 4, 's': 2, 'act': lrelu}),      # Nx8x8x64
                ('conv4', {'n_out': 64, 'k': 4, 's': 2, 'act': elu, 'batch': True}),                 # Nx4x4x64
                ('cnvT5', {'n_out': 64, 'k': 4, 's': 2, 'act': elu, 'concat': 'conv3'}),            # Nx8x8x(64+64)
                ('cnvT6', {'n_out': 3, 'k': 4, 's': 2, 'act': lrelu, 'batch': True}),                # Nx16x16x3
                ('fc7', {'reshape': [-1], 'n_out': 100, 'act': relu, 'batch': True, 'add': 'fc1'}), # Nx100
                ('fc8', {'n_out': 1, 'act': sigmoid})                                                  # Nx1
                ]


    z = tf.placeholder(shape = [64,64], dtype = tf.float32)
    x, layers_d, d_vars, d_moving_avg = network('discriminator', z, layers)

    n = 0
    print("\n discriminator output:", x.shape)
    print("\n layers")
    for i ,j in layers_d.items():
        print(i ,j)
        n += np.prod(j.get_shape().as_list())

    print("\n moving average assign nodes")
    print(d_moving_avg)

    print("\n total data size: ", n)

    n = 0
    for v in d_vars:
        n += np.prod(v.get_shape().as_list())
    print(" total variable size: ", n)

    with tf.Session() as sess:
        writer = tf. summary.FileWriter(r"C:\Users\LG\Documents\ex", sess.graph)

