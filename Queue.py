import tensorflow as tf

# 模拟同步先处理数据，然后才能取数据训练
# 1.定义队列
# tf.FIFOQueue(队列中元素个数，dtype类型形状，name=)
Q = tf.FIFOQueue(3,tf.float32)
# 放数据
# Q.enqueue(vals,name=None),vals列表或元组，返回一个进队列操作
# [[0.1,0.2,0.3],]是为了将tensor转变为列表
enq_many = Q.enqueue_many([[0.1,0.2,0.3],])

# 2.定义一些读取数据，取数据的过程  -取数据 +1 入队列
out_q = Q.dequeue()
data = out_q +1
en_q = Q.enqueue(data)

with tf.Session() as sess:
    # 初始化队列
    sess.run(enq_many)
    # 处理数据
    for i in range(100):
        # tensorflow运行操作有依赖性
        sess.run(en_q)
    # 训练数据
    for i in range(Q.size().eval()):
        print(sess.run(Q.dequeue()))
