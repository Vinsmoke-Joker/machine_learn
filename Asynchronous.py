import tensorflow as tf
# 模拟异步子线程存入样本、读取样本
# 1.定义一个队列，1000
Q = tf.FIFOQueue(1000,tf.float32)
# 2.定义子线程要做的事情，循环 值 +1，放入队列当中
var = tf.Variable(0.0)
# 实现一个自增 tf.assig_add()
data = tf.assign_add(var,tf.constant(1.0))
en_q = Q.enqueue(data)
# 3.定义队列管理器op,指定子线程该做的事情
qr = tf.train.QueueRunner(Q,enqueue_ops=[en_q]*2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 开启线程管理器
    coord = tf.train.Coordinator()
    # 真正开启子线程  start=True启动线程
    threads = qr.create_threads(sess,start=True)
    # 主线程，不断读取数据训练
    for i in range(300):
        print(sess.run(Q.dequeue()))

    # 主线程结束意味着Session（）关闭,资源释放
    # 回收子线程
    coord.request_stop()
    coord.join(threads)