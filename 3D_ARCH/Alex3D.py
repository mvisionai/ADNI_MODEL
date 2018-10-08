from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.keras.api.keras import backend as K
import ops as op_linker

shape_to_return=None
d_W_fc0=None
d_b_fc0=None
def input_placeholder(image_size, image_channel, label_cnt):
    with tf.name_scope('inputlayer'):
        inputs = tf.placeholder("float", [None, image_size, image_size, image_size, image_channel], 'inputs')
        labels = tf.placeholder("float", [None, label_cnt], 'labels')
        dropout_keep_prob = tf.placeholder("float",None, name='keep_prob')
        learning_rate = tf.placeholder("float", None, name='learning_rate')
        domain_label = tf.placeholder("float", [None,2], name='domain')
        flip_gra=tf.placeholder(tf.float32, [])

    return inputs, labels, dropout_keep_prob, learning_rate,domain_label,flip_gra

def train_inputs(image_size, image_channel, label_cnt) :
    with tf.name_scope('coninputlayer'):
        inputs = tf.placeholder("float", [None, image_size, image_size, image_size, image_channel], 'input')
        labels = tf.placeholder("float", [None, label_cnt], 'label')
        dropout_keep_prob = tf.placeholder("float", None, name='keep_prob')
        learning_rate = tf.placeholder("float", None, name='learning_rate')
        domain_label = tf.placeholder("float", [None, 2], name='domain')
        flip_gra = tf.placeholder(tf.float32, [], name='flip')

    return inputs, labels, dropout_keep_prob, learning_rate, domain_label, flip_gra




def inference(inputs, dropout_keep_prob, label_cnt):
    # todo: change lrn parameters
    with tf.variable_scope("convolution"):
            with tf.variable_scope('conv1layer'):
                conv1 = op_linker.conv(inputs, 2, 32,1)   # (input data, kernel size, #output channels, stride_size)
                conv1 = tf.layers.batch_normalization(conv1)
                conv1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

            # conv layer 2
            with tf.variable_scope('conv2layer'):
                conv2 = op_linker.conv(conv1,2,64, 1, 0.1)
                conv2 = tf.layers.batch_normalization(conv2)
                conv2 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

            # conv layer 3
            with tf.variable_scope('conv3layer'):
                conv3 = op_linker.conv(conv2, 2,128, 1, 0.1)
                conv3 = tf.layers.batch_normalization(conv3)
                conv3 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
                #no bias

            # conv layer 4
            with tf.variable_scope('conv4layer'):
                conv4 = op_linker.conv(conv3,2, 256, 1, 0.1)
                conv4 = tf.layers.batch_normalization(conv4)
                conv4 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

            # conv layer 5
            with tf.variable_scope('conv5layer'):
                conv5 = op_linker.conv(conv4, 2, 512, 1, 0.1)
                conv5 = tf.layers.batch_normalization(conv5)
                conv5 = tf.nn.max_pool3d(conv5, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

            # fc layer 1
            with tf.variable_scope('fc1layer'):
                global  shape_to_return
                global  d_b_fc0
                global  d_W_fc0
                fc1,shape_to_return,d_W_fc0,d_b_fc0 = op_linker.fc(conv5,512, 0.1, use_weight=True)
                fc1 = tf.nn.dropout(fc1, dropout_keep_prob)
                #bias changed

            # fc layer 2
            with tf.variable_scope('fc2layer'):
                fc2 = op_linker.fc(fc1,512, 0.1)
                fc2 = tf.nn.dropout(fc2, dropout_keep_prob)
                #bias changed

            # fc layer 3 - output
            with tf.variable_scope('fc3layer'):
                print(" ",end="\n")
                print("Graph Built")
                final_layer=op_linker.fc(fc2, label_cnt, 0.1, activation_func=tf.nn.softmax)
                return final_layer
                #bias changed



def autoencoder(inputs,batch):

  with tf.variable_scope('autoencoder') as scope:

     #scope.reuse_variables()
        # encoder
     #print("shape 1",inputs.shape)
     with tf.variable_scope('conv1layer'):
        #print("Main", inputs.shape)
        net = op_linker.conv(inputs, 2,32)  #3
        net=tf.layers.batch_normalization(net)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        #print("check ", net.shap # e)
        #print("shape 1",net.shape)

     with tf.variable_scope('conv2layer'):
        net = op_linker.conv(net, 2,64)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        #print("shape 2", net.shape)

     with tf.variable_scope('conv3layer'):
        net = op_linker.conv(net, 2,128)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #tuned
        #print("shape 3", net.shape)

     with tf.variable_scope('conv4layer'):
        net = op_linker.conv(net, 2,256)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #tuned
        #print("shape 4", net.shape)

     with tf.variable_scope('conv5layer'):
        net = op_linker.conv(net, 2,512)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #tuned
        #print("shape 5", net.shape)

        # decoder
     with tf.variable_scope('decon1layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 256,batch_size=batch)
        #print("check ", net.shape)


     with tf.variable_scope('decon2layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 128,batch_size=batch)
        #print("check ", net.shape)

     with tf.variable_scope('decon3layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 64,batch_size=batch)  # for max pooling , conv_padding='VALID'
        #net = op.deconv(net,5, 256, 1,batch_size)
        #print("check ", net.shape)

     with tf.variable_scope('decon4layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 32,batch_size=batch)
        #print("check ", net.shape)

       #net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='VALID')

     with tf.variable_scope('decon5layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = tf.layers.dense(inputs=net, units=1)
        # activation_fn = tf.nn.tanh
        #print("check ",net.shape)
        return net



def domain_parameters(flip_value):

    with tf.variable_scope('domain_predictor'):
        # Flip the gradient when backpropagating through this operation
        global  shape_to_return
        global  d_W_fc0
        global  d_b_fc0
        l = flip_value
        feature = shape_to_return
        feat = feature   #flip_gradient(feature, l)
        d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

        d_W_fc1 = weight_variable([512, 2])
        d_b_fc1 = bias_variable([2])
        final_la =  tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1
        domain_pred = tf.nn.softmax(final_la)
        return domain_pred



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def accuracy(logits, labels):
    # accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


def domain_accuracy(logits, labels):
    # accuracy
    with tf.name_scope('domain_accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('domain_accuracy', accuracy)
    return accuracy


def loss(logits, labels):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
    return loss



def loss_autoencoder(d_inputs, ae_output):
    with tf.name_scope('autoencoder_loss'):
        loss = tf.reduce_mean(tf.square(ae_output- d_inputs))
        tf.summary.scalar('autoencoder_loss', loss)
    return loss




def domain_loss(logits, labels):
    with tf.name_scope('domain_loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        tf.summary.scalar('domain_loss', loss)
    return loss


def train_rms_prop(loss, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp'):
    return tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon, use_locking, name).minimize(loss)




def new_encoder(inputs_,padding="SAME",stride=1):
    conv1 = tf.layers.conv3d(inputs=inputs_, filters=16, kernel_size=(3, 3, 3), padding=padding, strides=stride,activation=tf.nn.relu)
    maxpool1 = tf.layers.max_pooling3d(conv1, pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding)
    conv2 = tf.layers.conv3d(inputs=maxpool1, filters=32, kernel_size=(3, 3, 3), padding=padding, strides=stride,
                             activation=tf.nn.relu)
    maxpool2 = tf.layers.max_pooling3d(conv2, pool_size=(3, 3, 3), strides=(3, 3, 3), padding=padding)
    conv3 = tf.layers.conv3d(inputs=maxpool2, filters=96, kernel_size=(2, 2, 2), padding=padding, strides=stride,
                             activation=tf.nn.relu)
    maxpool3 = tf.layers.max_pooling3d(conv3, pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding)
    # latent internal representation

    # decoder
    unpool1 = K.resize_volumes(maxpool3, 2, 2, 2, "channels_last")
    deconv1 = tf.layers.conv3d_transpose(inputs=unpool1, filters=96, kernel_size=(2, 2, 2), padding=padding,
                                         strides=stride, activation=tf.nn.relu)
    unpool2 = K.resize_volumes(deconv1, 3, 3, 3, "channels_last")
    deconv2 = tf.layers.conv3d_transpose(inputs=unpool2, filters=32, kernel_size=(3, 3, 3), padding=padding,
                                         strides=stride, activation=tf.nn.relu)
    unpool3 = K.resize_volumes(deconv2, 2, 2, 2, "channels_last")
    deconv3 = tf.layers.conv3d_transpose(inputs=unpool3, filters=16, kernel_size=(3, 3, 3), padding=padding,
                                         strides=stride, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=deconv3, units=1)
    #print("main shape ",output.shape)

    return  output