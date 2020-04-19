import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

def linear_regression():
    # 实现线性回归
    # 1.创建数据，x特征值[1,100], y目标值[100]
    x = tf.random_normal(shape=(100,1),mean=1.75,stddev=0.5,name='x_data')
    y_true = tf.matmul(x,[[0.7]])+ 0.8
    # 2 建立线性模型 --y=kx+b
    # x [100,1] * [1,1] , y[100]
    # y = wx +b,w[1,1],b[1,1]
    # 训练模型参数必须使用tf.Variable
    # 随机初始化W1和b1,然后计算损失，在当前y_true下进行优化
    weight = tf.Variable(initial_value=tf.random_normal([1,1]),name='w') # 标准正太分布初始化权重
    bias = tf.Variable(initial_value=tf.random_normal([1,1]),name='b')
    y_predict = tf.matmul(x,weight)+bias
    # 3 确定损失函数（预测值与真实值之间的误差）-均方误差
    loss = tf.reduce_mean(tf.square(y_true-y_predict))
    # 4.梯度下降优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1,).minimize(loss)
    # 初始化变量
    init = tf.global_variables_initializer()
    # 实例化保存模型加载模型对象
    saver = tf.train.Saver()
    # 收集需要观察的变量
    tf.summary.scalar('losses',loss)  # scalar收集标量信息
    tf.summary.histogram('weight',weight)
    tf.summary.histogram('bias',bias) # 收集直方图信息
    # 合并tensor
    merge = tf.summary.merge_all()
    # 通过会话运行程序
    with tf.Session() as sess:
        sess.run(init)
        # 打印初始化的变量值
        print('初始化权重为%lf,偏置为%lf' % (weight.eval(),bias.eval()))
        print('训练前损失为',loss)
        # 将整张图写入文件
        file_writer = tf.summary.FileWriter('./temp/summary/linearmodel_2',graph=sess.graph)
        # 保存模型
        checkpoint = tf.train.latest_checkpoint('./temp/ckpt/test_2/mymodel.ckpt')
        if checkpoint is not None:
            saver.restore(sess,checkpoint)
        # if os.path.exists('./temp/ckpt/test_2/mymodel.ckpt'):
        #     saver.restore(sess, './temp/ckpt/test_2/mymodel.ckpt')
        # 训练模型
        for i in range(1,101):
            sess.run(optimizer)
            print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, loss.eval(), weight.eval(), bias.eval()))
            # 运行观察tensor结果
            summary = sess.run(merge)
            file_writer.add_summary(summary,i) # 观察第i步结果
            saver.save(sess,'./temp/ckpt/test_2/mymodel.ckpt')
if __name__ =='__main__':
    linear_regression()