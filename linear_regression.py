import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def linear_regression():
    """
     根据数据实现一个线性回归问题
    """
    # 训练数据：
    # 特征值：100个点，只有一个特征， 100个样本[100, 1]
    # 1 准备好数据集：y = 0.8x + 0.7 100个样本
    X = tf.random_normal(shape=(100,1),mean=2,stddev=1.0,name='feature') # 正太分布获取
    # 目标值：y =( 0.8 *x + 0.7)这个模型参数的值，是不知道的
    # [100,1] * [1,1]
    y_true = tf.matmul(X,[[0.8]])+0.7  # matmul矩阵乘法

    # 2 建立线性模型
    # x [100,1] * [1,1] , y[100]
    # y = wx +b,w[1,1],b[1,1]
    # 训练模型参数必须使用tf.Variable
    # 随机初始化W1和b1
    weight = tf.Variable(initial_value=tf.random_normal([1,1]),name='w') # 标准正太分布生成一个初始随机参数
    # trainable 改变量是否在训练过程时改变 ,默认为True
    # bias = tf.Variable(initial_value=tf.random_normal([1,1]),name='bias',trainable=False)
    bias = tf.Variable(initial_value=tf.random_normal([1, 1]), name='bias')  # 随机初始化偏置
    # y = W·X + b，目标：求出权重W和偏置b
    y_predict = tf.matmul(X,weight) + bias  # y=wx+b

    # 3 确定损失函数（预测值与真实值之间的误差）-均方误差
    loss = tf.reduce_mean(tf.square(y_true-y_predict))  # reduce_mean 求和再求均值
    # 4 梯度下降优化损失：需要指定学习率（超参数）
    # W2 = W1 - 学习率*(方向)
    # b2 = b1 - 学习率*(方向)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss) # minimize 传入需要优化的变量
    # 手动初始化Variable
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # 训练线性回归模型-运行整个图
        print("训练前,损失为:",sess.run(loss))
        print('随机初始化weight=%lf,bias=%lf'%(weight.eval(),bias.eval()))
        # 将整张图写入文件中
        file_writer = tf.summary.FileWriter('./temp/linear',graph=sess.graph)
        for i in range(1,101):
            sess.run(optimizer)
            print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, loss.eval(), weight.eval(), bias.eval()))
    # 总结：学习率和步长是会决定你的训练最后时间
if __name__ == '__main__':
    linear_regression()