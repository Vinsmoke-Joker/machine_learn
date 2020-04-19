import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('is_train',1,'指定模型是预测或训练')
def full_connected():
    # 1.获取数据-建立数据占位符
    mnist = input_data.read_data_sets('./data/mnist/input_data',one_hot=True)
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])
    # 2.建立一个全连接层神经网络w[784,10],b[10]
    with tf.variable_scope('fc_model'):
        # 随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784,10],mean=0,stddev=1.0),name='weight')
        bias = tf.Variable(tf.constant(0.0,shape=[10]),name='bias')
        y_predict = tf.matmul(x,weight) + bias

    # 3.计算交叉熵损失
    with tf.variable_scope('soft_loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
    # 4.梯度下降优化loss
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # 5.计算准确率
    with tf.variable_scope('accuracy'):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()
    # 收集变量
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.histogram('w',weight)
    tf.summary.histogram('b',bias)
    # 合并变量
    merged = tf.summary.merge_all()
    # 创建一个saver保存模型
    saver = tf.train.Saver()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        file_writer = tf.summary.FileWriter('./temp/mnist_CNN',graph=sess.graph)
        if FLAGS.is_train ==1:
            # 迭代训练
            for i in range(2000):
                # 取出真实存在的目标值和特征值
                mnist_x,mnist_y = mnist.train.next_batch(100)
                # 运行optimizer
                sess.run(optimizer,feed_dict={
                    x:mnist_x,
                    y_true:mnist_y
                })
                # 写入每步训练的值
                summary = sess.run(merged,feed_dict={
                    x: mnist_x,
                    y_true: mnist_y
                })
                file_writer.add_summary(summary,i)
                print('第%d次训练，准确率为%lf' %(i,accuracy))
            # 保存模型
            saver.save(sess,'./temp/ckpt/mnist_CNN/mnist_CNN_model')
        # 如果is_train !=1,开始预测
        else:
            # 加载模型
            saver.restore(sess,'./temp/ckpt/mnist_CNN/mnist_CNN_model')
            for i in range(100):
                # 每测试一张图片
                x_test,y_test = mnist.test.next_batch(1)
                # 预测
                print('第%d张图片,手写图片目标是%d,预测结果是%d'%(
                    i,
                    tf.argmax(y_test,1).eval(),
                    tf.argmax(sess.run(y_predict,feed_dict={x:x_test,y_true:y_test}),1).eavl()
                ))
    return None

def weight_variables(shape):
    """
    定义一个初始化权重的函数
    :return:
    """
    w = tf.Variable(tf.random_normal(shape,mean=0,stddev=1.0))
    return w
def bias_variables(shape):
    """
    随机初始化bias
    :return:
    """
    bias = tf.Variable(tf.constant(0.0,shape=shape))
    return bias
def model():
    """
    自定义的卷积模型
    :return:
    """
    # 1.准备数据的占位符x[None,784] ,y[None,10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])
    # 2.一卷积层-卷积:5*5*1,32个filter,strides=1-激活-池化
    with tf.variable_scope('conv1'):
        # 初始化权重
        w_conv1 = weight_variables([5,5,1,32])
        # 初始化偏置 [32]
        b_conv1 = bias_variables([32])
        # 对X进行形状改变 [None,784]->[None,28,28,1],并激活
        x_reshape = tf.reshape(x,[-1,28,28,1]) # 这里-1代表None
        # [None,28,28,1] -->[None,28,28,32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape,w_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1)
        # 池化 2*2,strides2 [None,28,28,32] -->[None,14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    # 3.二卷积层-卷积：5*5*32,64个filter,strides=1-激活-池化
    with tf.variable_scope('conv2'):
        # 初始化权重[5,5,32,64] 偏置[64]
        w_conv2 = weight_variables([5,5,32,64])
        b_conv2 = bias_variables([64])
        # 卷积、激活、池化
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1,w_conv2,strides=[1,1,1,1],padding='SAME') + b_conv2)
        # 池化  2*2,strides 2,[None,14,14,64]--->[None,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # 全连接层 [None,7,7,64] ->[None,7*7*64]*[7*7*64,10]+[10] = [None,10]
    with tf.variable_scope('fc_model'):
        # 随机初始化权重和偏置
        w_fc = weight_variables([7*7*64,10])
        b_fc = bias_variables([10])
        # 修改x_pool2形状 [None,7,7,64]-->[None,7*7*64]
        x_fc_reshape = tf.reshape(x_pool2,[-1,7*7*64])
        # 进行矩阵运算得出结果
        y_predict = tf.matmul(x_fc_reshape,w_fc)+b_fc
    return x,y_true,y_predict

def conv_fc():
    # 获取真实数据
    mnist = input_data.read_data_sets('./data/mnist/input_data',one_hot=True)
    # 定义模型，得出输出
    x,y_true,y_predict = model()
    # 进行交叉熵损失计算
    with tf.variable_scope('soft_loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
    # 梯度下降优化
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    # 计算准确率
    with tf.variable_scope('accuracy'):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    init = tf.global_variables_initializer()
    # 收集变量
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    merged = tf.summary.merge_all()
    # 保存模型
    saver = tf.train.Saver()
    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init)
        # 实例化events
        file_writer = tf.summary.FileWriter('./temp/mnist_CNN', graph=sess.graph)
        # 循环训练
        for i in range(1000):
            # 取出真实的特征值和目标值
            mnist_x,mnist_y = mnist.test.next_batch(50)
            # 训练
            sess.run(optimizer,feed_dict={x:mnist_x,y_true:mnist_y})
            # 写入events
            summary = sess.run(merged,feed_dict={x:mnist_x,y_true:mnist_y})
            file_writer.add_summary(summary,i)
            print('第%d次训练,准确率为%lf' % (i, sess.run(accuracy, feed_dict={x: mnist_x,y_true: mnist_y})))
        # 保存模型
        saver.save(sess, './temp/ckpt/mnist_CNN/mnist_CNN_model')
    return None

if __name__=='__main__':
    # full_connected()
    conv_fc()