import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib
import init
import os

#Filesystem organization
videogame = 'image1'
real = 'image1'
created = 'created'
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

#Initialize constants

#EXPERIMENT WITH THESE
beta = 5
alpha = 100
ITERATIONS = 100
#mean of VGG model
means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
style_weights = [('11', 0.5), ('21', 1.0), ('31', 1.5),('41', 3.0),('51', 4.0)]

#Content loss
def L_content(sess, model):
    #i is content layer
    i = '42'
    #x is generated image
    #F is feature representation of x in given layer Li
    F = sess.run(model[i])
    #p is original image
    #P is feature representation of p in given layer Li
    P = model[i]
    loss = 0.5 * tf.reduce_sum(tf.pow(F - P, 2))
    return loss

#Style loss
def L_style(sess, model, weights):
    
    def layer_loss(out1, out2):
        G = tf.reshape(out1, (out1.shape[2] * out1.shape[1], out1.shape[3]))
        G = tf.matmul(tf.transpose(G), G)
        A = tf.reshape(out2, (out1.shape[2] * out1.shape[1], out1.shape[3]))
        A = tf.matmul(tf.transpose(A), A)
        squared_sumGA = tf.reduce_sum(tf.pow(G - A, 2))
        return (1.0 / (4 * (out1.shape[2] * out1.shape[1])**2 * out1.shape[3]**2)) * squared_sumGA
    
    total_loss = 0.0
    for layer, weight in weights:
        temp_loss = layer_loss(sess.run(model[layer]), model[layer])
        total_loss += weight * temp_loss

    return total_loss

def init_image(path):
    img = scipy.misc.imread(path) * 1.0
    reshape = np.reshape(img, ((1,) + img.shape))
    return np.reshape(img, ((1,) + img.shape)) - means

def create_canvas(img, noise):
    noise_image = np.random.uniform(-20, 20,(1, img.shape[1], img.shape[2], img.shape[3])).astype('float32')
    canvas = noise_image * noise + img * (1 - noise)
    return canvas

def save_image(path, image):
    image = image + means
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


if __name__ == '__main__':
    if not os.path.exists('canvas'):
        os.mkdir('canvas')
    for it in range(11):
        content = init_image('content/test.jpg')
        style = init_image('style/test.jpg')
        noise = it * 1.0 / 10
        canvas = create_canvas(content, noise)
        filename = 'canvas/%d.png' % (it)
        save_image(filename, canvas)



