import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 变量Op
a = tf.constant([1,2,3,4,5])
# name 参数：在tensorboard使用时候显示名字，可以让相同op进行区分
b = tf.constant(4.5,name='c')
c = tf.constant(5.0,name='d')
d = tf.add(b,c,name='add')
var = tf.Variable(tf.random_normal([2,3],mean=0.0,stddev=1.0))
print(a,var)

# Variable需要先进行初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 必须初始化op
    sess.run(init)
    # 把程序的图结构写入events文件
    file_writer = tf.summary.FileWriter('./temp/summary/test/',graph=sess.graph)
    print(sess.run([a,var,d]))
    # print(sess.run(d))