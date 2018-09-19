import tensorflow as tf


def generator(sequence_type):
    if sequence_type == 1:
        for i in range(5):
            yield 10 + i
    elif sequence_type == 2:
        for i in range(5):
            yield (30 + 3 * i, 60 + 2 * i)
    elif sequence_type == 3:
        for i in range(1, 4):
            yield (i, ['Hi'] * i)



sec=("ml,dp","love,like")
lists=(i.split(",") for i in sec)

for j in lists:
    print(j)

value=generator(2)
for i in value:
  print(i)


# dataset=tf.data.Dataset.from_generator(generator, (tf.float32), args = ([1]))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(dataset))
