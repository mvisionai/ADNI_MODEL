import numpy as np
import tensorflow as tf
from AD_Dataset import Dataset_Import

# Path to 3d tensor. Tensor.shape is (111,111,111)
data_feed=Dataset_Import()

# Graph
batch_size = 10
num_labels = 3


predict = tf.Variable(False)
# Input data.
tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 10, 10, 10, 1))
tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

# Variables.
layer1_weights = tf.Variable(tf.truncated_normal(
    [7, 7, 7, 1, 32], stddev=0.1))

layer1_biases = tf.Variable(tf.zeros([32]))

layer2_weights = tf.Variable(tf.truncated_normal(
    [5, 5, 5, 32, 64], stddev=0.1))

layer2_biases = tf.Variable(tf.constant(1.0, shape=[64]))

layer3_weights = tf.Variable(tf.truncated_normal(
    [3, 3, 3, 64, 128], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[128]))

layer4_weights = tf.Variable(tf.truncated_normal(
    [3, 3, 3, 128, 256], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[256]))

layer5_weights = tf.Variable(tf.truncated_normal(
    [3, 3, 3, 256, 256], stddev=0.1))
layer5_biases = tf.Variable(tf.constant(1.0, shape=[256]))

layer6_weights = tf.Variable(tf.truncated_normal(
    [256, 4096], stddev=0.1))

#49 * 49 * 256
layer6_biases = tf.Variable(tf.constant(1.0, shape=[4096]))

layer7_weights = tf.Variable(tf.truncated_normal(
    [4096, 4096], stddev=0.1))
layer7_biases = tf.Variable(tf.constant(1.0, shape=[4096]))

layer8_weights = tf.Variable(tf.truncated_normal(
    [4096, num_labels], stddev=0.1))
layer8_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


# MODEL
def model(data):
    # Conv1
    conv1 = tf.nn.conv3d(data, layer1_weights, [1, 4, 4, 4, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)

    # Pool1
    pool1 = tf.nn.max_pool3d(hidden1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    # Conv2
    conv2 = tf.nn.conv3d(pool1, layer2_weights, [1, 1, 1, 1, 1])
    hidden2 = tf.nn.relu(conv2 + layer2_biases)

    # Conv3
    conv3 = tf.nn.conv3d(hidden2, layer3_weights, [1, 1, 1, 1, 1])

    # Conv4
    conv4 = tf.nn.conv3d(conv3, layer4_weights, [1, 1, 1, 1, 1])

    # Conv5
    conv5 = tf.nn.conv3d(conv4, layer5_weights, [1, 1, 1, 1, 1], padding='SAME')

    # Pool2
    pool2 = tf.nn.max_pool3d(conv5, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    input_shape = pool2.get_shape().as_list()

    normalize3_flat = tf.reshape(pool2, [-1,input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]])
    #[49 * 49 * 256,-1]

    # FC1
    fc1 = tf.tanh(tf.add(tf.matmul(normalize3_flat, layer6_weights), layer6_biases))
    dropout1 = tf.nn.dropout(fc1, 0.5)

    # FC2
    fc2 = tf.tanh(tf.add(tf.matmul(dropout1, layer7_weights), layer7_biases))
    dropout2 = tf.nn.dropout(fc2, 0.5)

    # FC3
    res = tf.nn.softmax(tf.add(tf.matmul(dropout2, layer8_weights), layer8_biases))

    return res


# Training computation
local_res = model(tf_train_dataset)
#print(local_res.get_shape().as_list())
valid_prediction =local_res

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(local_res), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=valid_prediction))
# Optimizer
train_step = tf.train.MomentumOptimizer(0.0014, 0.9).minimize(cross_entropy)

print('Graph was built')

# Session
epochs = 100


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print('EPOCH %d' % epochs)
    #for epch in range(epochs):
       # print('EPOCH %d' % epch)

    for step in range(epochs):
    #     offset = (step * batch_size) % (Y.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data,batch_labels = data_feed.next_batch(batch_size)
        session.run(train_step,feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
        session.run(train_step,
                                     feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels},
                                     )

        accuracy = 100 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(local_res, 1), tf.argmax(tf_train_labels, 1)), tf.float32))

        train_accuracy = accuracy.eval(feed_dict={
            tf_train_dataset: batch_data, tf_train_labels: batch_labels})

        print("Step %d" % 1)
        print("Minibatch accuracy: %.1f%%" % train_accuracy)