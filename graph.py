import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
a = tf.constant(5)
b = tf.constant(6)
sum = tf.add(a,b)
# 定义一张新图包含op,tensor,上下文环境中
# op:一个操作对象（Operation）是TensorFlow图中的一个节点,通过tensorflow定义的函数或类都是op
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11)
    print(c.graph)

# tensorflow 重载机制,默认会给运算符重载为op类型
d = 3
e = b + d

# placeholder用于实时提供数据训练场景
m = tf.placeholder(tf.float32,[2,3])

# 会话默认使用一张图
with tf.Session() as sess:
    print(sess.run(sum))
    # 获取默认图,相当于给程序分配内存
    print(tf.get_default_graph())
    # 只要有会话的上下文环境就可以使用eval()获取值
    print(a.eval())
    print(a.graph)
    print(b.graph)
    print(sess.graph)
    print(sess.run(e))  # 9
    print(sess.run(m,feed_dict={m:[[1,2,3],[4,5,6]]}))


