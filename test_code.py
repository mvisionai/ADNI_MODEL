import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])

W = tf.get_variable("weights1", shape=[784, 10],
                    initializer=tf.glorot_uniform_initializer())

b = tf.get_variable("bias1", shape=[10],
                    initializer=tf.constant_initializer(0.1))

y = tf.nn.relu(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
with tf.Session() as sess:


        time_string = datetime.datetime.now().isoformat()
        train_writer = tf.summary.FileWriter('np_data/train/{}'.format("vs"), sess.graph)
        test_writer  = tf.summary.FileWriter('np_data/test/{}'.format("vs"), sess.graph)

        tf.global_variables_initializer().run()

        for step in range(1000):
            print("training step:{}".format(step))
            if step % 10 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images,
                                                                       y_: mnist.test.labels})
                test_writer.add_summary(summary, step)
                print("Step ",step ,"Accuracy ",accuracy)
            else:
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
                summary = sess.run(merged, feed_dict={x: batch_xs,
                                                      y_: batch_ys})
                train_writer.add_summary(summary, step)