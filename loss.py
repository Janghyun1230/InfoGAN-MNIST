import tensorflow as tf


def loss(type, real, fake):
    if type =='GAN':
        d_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(real))) + tf.reduce_mean(-tf.log(1-tf.nn.sigmoid(fake)))
        g_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(fake)))
    elif type =='lsGAN':
        d_loss = tf.reduce_mean((real-1)**2)/2.0 + tf.reduce_mean(fake**2)/2.0
        g_loss = tf.reduce_mean((fake-1)**2)/2.0
    elif type[:4]=='wGAN':
        d_loss = - tf.reduce_mean(real) + tf.reduce_mean(fake)
        g_loss = - tf.reduce_mean(fake)
    else:
        raise Exception("you should enter type GAN, lsGAN, wGAN, wGAN_GP")
    return d_loss, g_loss