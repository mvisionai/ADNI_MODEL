import tensorflow as tf
import  numpy as np
import  AD_Constants as constants


def conv(inputs, kernel_size, output_num, stride_size=1, init_bias=0.0, conv_padding='SAME', stddev=0.01, activation_func=tf.nn.relu):

    input_size = inputs.get_shape().as_list()[-1]

    #init_weight_var=tf.random_normal_initializer(stddev=0.02)
    #conv_weights =tf.get_variable(name='weights', shape=[kernel_size, kernel_size, kernel_size, input_size, output_num],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    #tf.summary.histogram("weight_value",conv_weights)
    #conv_biases = tf.get_variable(name='net_biases',
                                   #shape=[output_num],initializer= tf.constant_initializer(init_bias),trainable=True)
    #tf.summary.histogram("bias_value",conv_biases )
    conv_layer = tf.layers.Conv3D(inputs=inputs, filters=conv_weights,stride_size= [stride_size, stride_size, stride_size], padding=conv_padding,kernel_initializer=tf.contrib.layers.xavier_initializer(),
    bias_initializer=tf.constant_initializer(init_bias))

    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
    if activation_func:
        conv_layer = activation_func(conv_layer)
    return conv_layer


def deconv(inputs, kernel_size, output_num, stride_size=1,conv_padding='SAME', stddev=0.01,batch_size=None):
    input_size = inputs.get_shape().as_list()[-1]

    depth=get_deconv_dim(inputs.get_shape().as_list()[1],stride_size,kernel_size,conv_padding)
    height=get_deconv_dim(inputs.get_shape().as_list()[2],stride_size,kernel_size,conv_padding)
    width=get_deconv_dim(inputs.get_shape().as_list()[3],stride_size,kernel_size,conv_padding)
    dconv_weights = tf.Variable(
        tf.truncated_normal([kernel_size, kernel_size, kernel_size,output_num, input_size], dtype=tf.float32,
                            stddev=stddev),
        name='de_weights')
    dconv_layer = tf.nn.conv3d_transpose(inputs, dconv_weights,output_shape=[batch_size,depth,height,width,output_num], strides=[1, stride_size, stride_size, stride_size, 1], padding=conv_padding)
    #dconv_layer = tf.layers.conv3d_transpose(inputs,output_num,kernel_size,stride_size,padding=conv_padding)

    #print("dec ", dconv_layer.get_shape())


    return dconv_layer




def fc(inputs, output_size, init_bias=0.0, activation_func=tf.nn.relu, stddev=0.01,use_weight=False):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 5:
        fc_weights = tf.Variable(
            tf.truncated_normal([input_shape[1] * input_shape[2] * input_shape[3]* input_shape[4], output_size], dtype=tf.float32,
                             stddev=stddev),
            name='weights')
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.truncated_normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),
                                 name='weights')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32), name='net_biases')
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    if activation_func:
        fc_layer = activation_func(fc_layer)

    if use_weight==True:
     return fc_layer,inputs,fc_weights,fc_biases
    else:
     return  fc_layer

def weight_bias_pass(inputs, output_size, init_bias=0.0,stddev=0.01):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 5:
        fc_weights = tf.Variable(
            tf.truncated_normal([input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], output_size],
                                dtype=tf.float32,
                                stddev=stddev),
            name='weights')
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.truncated_normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),
                                 name='weights')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32), name='sh_biases')
    return fc_weights,fc_biases

def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)


def load_initial_weights(session,trained_var,use_pretrain=False):

  # Load and assign pretrained weights and biases

  if use_pretrain :

      for key in trained_var:

           if len(key.split('/')) > 2:
                  if key.split('/')[1] in constants.pre_trained:


                    with tf.variable_scope("convolution"):

                        parent_scope = key.split('/')[1]
                        with tf.variable_scope(parent_scope, reuse = True):

                          train_w="/".join([key.split('/')[0],parent_scope])
                          trained_weight="/".join([train_w,"weights"])
                          trained_bias = "/".join([train_w, "net_biases"])
                          if  trained_weight==key:
                            weight_var=tf.get_variable('weights', trainable=False)
                            #print("checker ", weight_var)
                            weight_format = ":".join([trained_weight, "0"])
                            weight_r = session.run(weight_format)
                            session.run(weight_var.assign(weight_r))
                          if trained_bias ==key:

                             bias_var=tf.get_variable('net_biases', trainable=False)
                             #print("checker ", bias_var)
                             biase_format = ":".join([trained_bias, "0"])
                             bias = session.run(biase_format)
                             session.run(bias_var.assign(bias))


def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
    dim_size *= stride_size

    if padding == 'VALID' and dim_size is not None:
        dim_size += max(kernel_size - stride_size, 0)
    return dim_size