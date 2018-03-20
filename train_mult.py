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
noise = .6 #ratio
beta = .5
alpha = .5
ITERATIONS = 100
#mean of VGG model
means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
style_weights = [('11', 0.2), ('21', .2), ('31', .2),('41', .2),('51', .2)]

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

def create_canvas(img):
    noise_image = np.random.uniform(-20, 20,(1, img.shape[1], img.shape[2], img.shape[3])).astype('float32')
    canvas = noise_image * noise + img * (1 - noise)
    return canvas

def save_image(path, image):
    image = image + means
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


if __name__ == '__main__':
    #initialize session, imagesx3
    session = tf.Session()
    content = init_image('content/test.jpg')
    style = init_image('style/test.jpg')
    vgg = init.import_model()
    
    model = init.create_network(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    canvas = create_canvas(content)
    session.run(tf.initialize_all_variables())

    
    #content loss from content image
    session.run(model['init'].assign(content))
    L_content = L_content(session, model)

    #style loss from style image
    session.run(model['init'].assign(style))
    L_style = L_style(session, model, style_weights)


    #there is a tradeoff between mainting content and introducing stlye, therefore
    #the loss function to minimize: Ltotal(~p,~a, ~x) = alpha * Lcontent(p, x) + beta * Lstyle(a, x)
    total_loss = alpha * L_content + beta * L_style

    #train
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)

    session.run(tf.initialize_all_variables())
    session.run(model['init'].assign(canvas))

    for it in range(ITERATIONS+1):
        session.run(train_step)
        if it%10 == 0:
            # Print every 100 iteration.
            mixed_image = session.run(model['init'])
            print('Iteration %d' % (it))
            print('sum : ', session.run(tf.reduce_sum(mixed_image)))
            print('cost: ', session.run(total_loss))
            if not os.path.exists('output'):
                os.mkdir('output')
            filename = 'output/mixed1_%d.png' % (it)
            save_image(filename, mixed_image)


    session = tf.Session()
    content = init_image('output/mixed1_30.png')
    style = init_image('style/temp1.jpg')
    vgg = init.import_model()

    model = init.create_network(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    canvas = create_canvas(content)
    session.run(tf.initialize_all_variables())


    #content loss from content image
    session.run(model['init'].assign(content))
    L_content = L_content(session, model)

    #style loss from style image
    session.run(model['init'].assign(style))
    L_style = L_style(session, model, style_weights)


    #there is a tradeoff between mainting content and introducing stlye, therefore
    #the loss function to minimize: Ltotal(~p,~a, ~x) = alpha * Lcontent(p, x) + beta * Lstyle(a, x)
    total_loss = alpha * L_content + beta * L_style

    #train
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)

    session.run(tf.initialize_all_variables())
    session.run(model['init'].assign(canvas))

    for it in range(ITERATIONS+1):
        session.run(train_step)
        if it%10 == 0:
            # Print every 100 iteration.
            mixed_image = session.run(model['init'])
            print('Iteration %d' % (it))
            print('sum : ', session.run(tf.reduce_sum(mixed_image)))
            print('cost: ', session.run(total_loss))
            if not os.path.exists('output'):
                os.mkdir('output')
            filename = 'output/mixed2_%d.png' % (it)
            save_image(filename, mixed_image)



    session = tf.Session()
    content = init_image('output/mixed2_30.png')
    style = init_image('style/temp2.jpg')
    vgg = init.import_model()

    model = init.create_network(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    canvas = create_canvas(content)
    session.run(tf.initialize_all_variables())


    #content loss from content image
    session.run(model['init'].assign(content))
    L_content = L_content(session, model)
    
    #style loss from style image
    session.run(model['init'].assign(style))
    L_style = L_style(session, model, style_weights)


    #there is a tradeoff between mainting content and introducing stlye, therefore
    #the loss function to minimize: Ltotal(~p,~a, ~x) = alpha * Lcontent(p, x) + beta * Lstyle(a, x)
    total_loss = alpha * L_content + beta * L_style
    
    #train
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)

    session.run(tf.initialize_all_variables())
    session.run(model['init'].assign(canvas))
    
    for it in range(ITERATIONS+1):
        session.run(train_step)
        if it%10 == 0:
            # Print every 100 iteration.
            mixed_image = session.run(model['init'])
            print('Iteration %d' % (it))
            print('sum : ', session.run(tf.reduce_sum(mixed_image)))
            print('cost: ', session.run(total_loss))
            if not os.path.exists('output'):
                os.mkdir('output')
            filename = 'output/mixed3_%d.png' % (it)
            save_image(filename, mixed_image)

