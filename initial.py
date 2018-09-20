import tensorflow as tf


def function_a():
    for i in range(10):
        yield i+10



gen=function_a()

for j in gen:
    print(j)