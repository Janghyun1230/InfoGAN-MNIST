import tensorflow as tf
from network import *
import train

# command center
batch = 64
latent_dim = 96
cat = 10
conti = 2

layers_d = [('conv1', {'n_out': 64, 'k': 4, 's': 2, 'act': lrelu}),
            ('conv2', {'n_out': 128, 'k': 4, 's': 2, 'act': lrelu, 'batch': True}),
            ('fc3', {'reshape': [-1], 'n_out': 1024, 'act': lrelu, 'batch': True}),
            ('fc4', {'n_out': 1, 'act': None})
            ]

layers_q = [('fc1', {'n_out': 128, 'act': lrelu, 'batch': True}),
            ('fc2', {'n_out': cat+conti*2})]

layers_g = [('fc1', {'n_out': 1024, 'act': relu, 'batch': True}),
            ('fc2', {'n_out': 7 * 7 * 128, 'act': relu, 'batch': True}),
            ('cnvT3', {'reshape': [7, 7, 128], 'n_out': 64, 'k': 4, 's': 2, 'act': relu, 'batch': True}),
            ('cnvT4', {'n_out': 1, 'k': 4, 's': 2, 'act': tanh})
            ]

gan_type = 'GAN'
loss_lambda = 10.0  # for wGAN_GP

optim_d = {'type': 'adam', 'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999}
optim_g = {'type': 'adam', 'lr': 1e-3, 'beta1': 0.5, 'beta2': 0.999}

train_config = {'d_per_g': 5, 'num_epoch': 5, 'show_every': 500, 'print_every': 250}
save_file_dir = "GAN_MNIST"

tf.reset_default_graph()
# train and save model (configuration is also saved after training)
train.train_mnist(layers_d, layers_g, layers_q, gan_type, optim_d, optim_g, batch, latent_dim, train_config, save_file_dir,
                  loss_lambda = loss_lambda, device = '/cpu:0')