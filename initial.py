import tensorflow as tf


with tf.variable_scope("ch"):
  v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1), dtype=tf.float32,trainable=True)
  cg=tf.Variable(2,name="vb", dtype=tf.float32)
  fin=v+cg

with tf.Session() as ses:
 ses.run(tf.global_variables_initializer())
 with tf.variable_scope("ch", reuse=True):
    new_v=tf.get_variable('v', trainable=False)
    print(new_v)
    ses.run(new_v.assign([3]))
    df=ses.run(fin,feed_dict={cg:5})

 print(df)