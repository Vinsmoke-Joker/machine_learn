import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('is_train',1,'指定模型是预测或训练')
def full_connected():
    # 1.获取数据-建立数据的占位符，x[None,784],y_true[None,10]
    mnist = input_data.read_data_sets('./data/mnist/input_data',one_hot=True)
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])
    # 2.建立一个全连接层的神经网络 w[784,10],b[10]
    with tf.variable_scope('fc_model'):
        # 随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784,10],mean=0,stddev=1.0),name='weight')
        bias = tf.Variable(tf.constant(0.0,shape=[10]),name='bias')
        # 预测None结果的输出结果[None,784] * [784,10]+[10] =[None,10]
        y_predict = tf.matmul(x,weight)+bias
    # 3.计算交叉熵损失
    with tf.variable_scope('soft_loss'):
        # 平均交叉熵损失
        # softmax_cross_entropy_with_logits 返回一个列表
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
    # 4.梯度下降求出损失
    with tf.variable_scope('optimizer'):
        optimizer_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    # 5.计算准确率
    with tf.variable_scope('accuracy'):
        # argmax求出y_true中最大的一个值的下标
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
    # 初始化变量
    init = tf.global_variables_initializer()
    # 收集变量
    tf.summary.scalar('loss',loss) # scalar单个数字值收集
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.histogram('weight',weight)  # 高维度变量收集
    tf.summary.histogram('bias',bias)
    # 合并变量
    merged = tf.summary.merge_all()
    # 创建一个saver保存模型
    saver = tf.train.Saver()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 建立events
        filewriter = tf.summary.FileWriter('./temp/mnist',graph=sess.graph)
        if FLAGS.is_train == 1:
            # 迭代训练，更新参数训练
            for i in range(2000):
                # 取出真实存在的特征值和目标值
                mnist_x,mnist_y = mnist.train.next_batch(100)
                # 运行optimizer_op
                sess.run(optimizer_op,feed_dict={
                    x:mnist_x,
                    y_true:mnist_y
                })
                # 写入每步训练的值
                summary = sess.run(merged,feed_dict={
                    x:mnist_x,
                    y_true:mnist_y
                })
                filewriter.add_summary(summary,i)
                print('第%d次训练,准确率为%lf' % (i,sess.run(accuracy,feed_dict={
                    x:mnist_x,
                    y_true:mnist_y
                })))
            # 保存模型
            saver.save(sess,'./temp/ckpt/mnist/mnist_model')
        # 如果is_train 不为1，预测
        else:
            # 加载模型
            saver.restore(sess,'./temp/ckpt/mnist/mnist_model')
            for i in range(100):
                # 每次测试一张图片
                x_test,y_test = mnist.test.next_batch(1)
                # 预测
                print("第%d张图片，手写数字图片目标是:%d, 预测结果是:%d" % (
                    i,
                    tf.argmax(y_test, 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                ))
    return None
if __name__ =='__main__':
    full_connected()
