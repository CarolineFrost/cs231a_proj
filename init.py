import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib

#Import VGG exisitng model
vgg_file = 'imagenet-vgg-verydeep-19.mat'
vgg_model = scipy.io.loadmat(vgg_file)
vgg_layers = vgg_model['layers']
#print(vgg_layers)

def import_model():
    return vgg_model

def get_weights(layer_num):
    return vgg_layers[0][layer_num][0][0][2][0][0]

def get_bias(layer_num):
    return vgg_layers[0][layer_num][0][0][2][0][1]

#Activation function from https://www.tensorflow.org/api_guides/python/nn#Activation_Functions
def activ_func(feat):
    return tf.nn.relu(feat)

#Convolution function from https://www.tensorflow.org/api_guides/python/nn#Convolution
#Takes in 1 layer from model to inform filter, 1 layer to act as input
def conv_func(input, fil_num):
    w = tf.constant(get_weights(fil_num))
    b = get_bias(fil_num)
    b = tf.constant(np.reshape(b, (b.size)))
    return tf.nn.conv2d(input, filter=w, strides=[1, 1, 1, 1], padding='SAME') + b

#Pooling function from https://www.tensorflow.org/api_guides/python/nn#Pooling
def pool(prev):
    return tf.nn.avg_pool(prev, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Network
#Layer configuration information here: https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt
#Given layer -> conv2d(network layer w/ vgg layer) -> relu -> save in network
#             [ repeat with each necessary vgg layer ]
#            -> pool -> save in network
#5 pools total
def create_network(height, width, color_num):
    network = {}
    network['init'] = tf.Variable(np.zeros((1, height, width, color_num)), dtype = 'float32')
    #tf.cast(network['init'], tf.int32)

    #1
    network['11'] = activ_func(conv_func(network['init'], 0))
    network['12'] = activ_func(conv_func(network['11'], 2))
    network['p1'] = pool(network['12'])

    #2
    network['21'] = activ_func(conv_func(network['p1'], 5))
    network['22'] = activ_func(conv_func(network['21'], 7))
    network['p2'] = pool(network['22'])

    #3
    network['31'] = activ_func(conv_func(network['p2'], 10))
    network['32'] = activ_func(conv_func(network['31'], 12))
    network['33'] = activ_func(conv_func(network['32'], 14))
    network['34'] = activ_func(conv_func(network['33'], 16))
    network['p3'] = pool(network['34'])
    
    #4
    network['41'] = activ_func(conv_func(network['p3'], 19))
    network['42'] = activ_func(conv_func(network['41'], 21))
    network['43'] = activ_func(conv_func(network['42'], 23))
    network['44'] = activ_func(conv_func(network['43'], 25))
    network['p4'] = pool(network['44'])
    
    #5
    network['51'] = activ_func(conv_func(network['p4'], 28))
    network['52'] = activ_func(conv_func(network['51'], 30))
    network['53'] = activ_func(conv_func(network['52'], 32))
    network['54'] = activ_func(conv_func(network['53'], 34))
    network['p5'] = pool(network['54'])
    
    return network







