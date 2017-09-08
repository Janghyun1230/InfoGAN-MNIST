import numpy as np
import matplotlib.pyplot as plt


def preprocess(x):
    return 2 * x - 1.0


def deprocess(x):
    return (x + 1.0) / 2.0


def noise(batch_size, dim):
    return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, dim))


def sample_c1(batch_size, cat=10):
    return np.random.multinomial(1,[1.0/cat]*cat,batch_size).astype(float)


def sample_c2(batch_size, conti=2):
    return np.random.uniform(size= (batch_size,2), low = -1, high = 1)


def image_show(images, c1=0, c2=0, generated=True):
    num = images.shape[0]
    plt.figure(figsize = (10,3))
    for i in range(num):
        plt.subplot(1,num,i+1)
        plt.imshow(deprocess(images[i]).reshape(28,28))
        plt.axis('off')
        if generated:
            plt.title("cat:%d"%np.argmax(c1[i]))
    plt.show()
