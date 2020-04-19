import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 实现一个加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)
print(a)
print(b)

sum1 = tf.add(a,b)
# 获取默认图,相当于给程序分配内存
print(tf.get_default_graph())
print(sum1)
with tf.Session() as sess:
    print(sess.run(sum1))
    print(a.graph)
    print(b.graph)
    print(sess.graph)