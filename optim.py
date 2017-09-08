import tensorflow as tf


# beta1 means momentum scale. (larger means consider previous update more)
# beta2 works similar so commonly do not change
def optimizer(type, lr=1e-3, beta1=0.9, beta2=0.999):
    if type == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=beta1)
    elif type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=lr, decay=beta2)
    elif type == "adam":
        return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)