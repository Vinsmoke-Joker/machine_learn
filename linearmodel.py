import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 命令行参数:第一个参数，名字或默认值或说明
# 指定模型训练步数
# tf.app.flags.DEFINE_integer = ('max_step',0,'线性回归模型训练步数')
# tf.app.flags.DEFINE_string('model_pate', '', './temp/ckpt/test/myregression.ckpt')
# FLAGS = tf.app.flags.FLAGS
def linear_regression():
    with tf.variable_scope('init_data'):
        # 1 准备好数据集：y = 0.8x + 0.7 100个样本
        X = tf.random_normal(shape=(100,1),mean=2,stddev=1.0,name='feature') # 正太分布获取
        # 目标值：y =( 0.8 *x + 0.7)这个模型参数的值，是不知道的
        # [100,1] * [1,1]
        y_true = tf.matmul(X,[[0.8]],name='original_matmul')+0.7  # matmul矩阵乘法

    # 2 建立线性模型
    with tf.variable_scope('linear_model'):
        weight = tf.Variable(initial_value=tf.random_normal([1,1]),name='w') # 标准正太分布生成一个初始随机参数
        bias = tf.Variable(initial_value=tf.random_normal([1, 1]), name='bias')  # 随机初始化偏置
        # y = W·X + b，目标：求出权重W和偏置b
        y_predict = tf.matmul(X,weight,name='matmul_model') + bias  # y=wx+b

    # 3 确定损失函数（预测值与真实值之间的误差）-均方误差
    with tf.variable_scope('loss_model'):
        loss = tf.reduce_mean(tf.square(y_true-y_predict))  # reduce_mean 求和再求均值
    # 4 梯度下降优化损失：需要指定学习率（超参数）
    with tf.variable_scope('gd_optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01,name='optimizer').minimize(loss) # minimize 传入需要优化的变量

    # 收集要观察的tensor
    tf.summary.scalar('losses',loss)  # scalar用来显示标量信息
    tf.summary.histogram('w',weight) # histogram 用来显示直方图信息
    tf.summary.histogram('b',bias)
    # 合并tensor
    merge = tf.summary.merge_all()

    # 模型的保存
    saver = tf.train.Saver()
    # 手动初始化Variable
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # 训练线性回归模型-运行整个图
        print("训练前,损失为:",sess.run(loss))
        print('随机初始化weight=%lf,bias=%lf'%(weight.eval(),bias.eval()))
        # 将整张图写入文件中
        file_writer = tf.summary.FileWriter("./temp/summary/linearmodel",graph=sess.graph)

        # 加载历史模型，基于历史模型训练
        checkpoint = tf.train.latest_checkpoint("./temp/ckpt/test/myregression.ckpt")
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        # for i in range(FLAGS.max_step):
        for i in range(1,101):
            sess.run(optimizer)
            print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, loss.eval(), weight.eval(), bias.eval()))
            # 运行观察tensor结果
            summary = sess.run(merge)
            file_writer.add_summary(summary,i) # 观察第N步的损失值
            # 训练每一步进行模型保存 ckeckpoint类型，而filewrite 是events类型
            # 指定目录+模型名字  save默认保存最近5个模型
            saver.save(sess,"./temp/ckpt/test/myregression.ckpt")
    # 总结：学习率和步长是会决定你的训练最后时间
if __name__ == '__main__':
    linear_regression()