import tensorflow as tf
FEATURE_COLUMNS = ['channel_id', 'vector', 'user_weights', 'article_weights']

class LrWithFtrl():
    """
    点击样本数据在线优化FTRL训练
    """
    def __init__(self):
        pass
    @staticmethod
    def get_tfrecords_data():
        """获取TFrecords数据"""
        # 解析TFrecords格式
        def deal_with_tfrecords(example_proto):
            # parse_single_example(解析数据，解析规则dict)
            # 按照当时保存的键名，类型
            features = {
                'label':tf.FixedLenFeature([],tf.int64),
                'feature':tf.FixedLenFeature([],tf.string)
            }
            label_feature = tf.parse_single_example(example_proto,features)
            # 进行目标值和特征值处理
            # label_feature字典，如果是str类型需要tf.decode_raw()解码,其他类型不需要
            feature = tf.decode_raw(label_feature['feature'],tf.float64)
            # 对特征值处理
            # 1.确定形状121
            # 2.求出文章向量，用户权重，文章权重的平均值
            feature_reshape = tf.reshape(tf.cast(feature,tf.float32),[1,121])
            # tf.slice(feature, 起始位置, 步长)
            channel_id = tf.cast(tf.slice(feature_reshape,[0,0],[1,1]),tf.int32)
            # tf.reduce_sum(求平均值，axis=1行/0列)
            vector = tf.reduce_sum(tf.slice(feature_reshape, [0, 1], [1, 100]),axis=1)
            user_weights = tf.reduce_sum(tf.slice(feature_reshape, [0, 101], [1, 10]),axis=1)
            article_weight = tf.reduce_sum(tf.slice(feature_reshape, [0, 111], [1, 10]),axis=1)
            # 返回固定的特征字典和目标值
            data = [channel_id,vector,user_weights,article_weight]
            feature_dict = dict(zip(FEATURE_COLUMNS,data))
            label = tf.cast(label_feature['label'],tf.float32)
            return feature_dict,label

        # 读取文件
        dataset = tf.data.TFRecordDataset(['./data/train_ctr_20190605.tfrecords'])
        # 一个样本一个样本处理，传入处理函数一个example,example_proto
        dataset = dataset.map(deal_with_tfrecords)
        dataset = dataset.batch(32)
        dataset = dataset.repeat(10)
        return dataset
    # 2.指定特征列
    def train_eval(self):
        # 离散型
        channel_id = tf.feature_column.categorical_column_with_identity('channel_id',num_buckets=25)
        # 连续型
        vector = tf.feature_column.numeric_column('vector')
        user_weights = tf.feature_column.numeric_column('user_weights')
        article_weight = tf.feature_column.numeric_column('article_weights')
        # 合并到一个列表当中
        column = [channel_id,vector,user_weights,article_weight]
        # 3.训练模型
        # 前提LR（逻辑回归）才可以用FTRL，做了特征交叉FTRL效果会更好
        classifier = tf.estimator.LinearClassifier(feature_columns=column,
                                                   optimizer=tf.train.FtrlOptimizer(
                                                       learning_rate=0.1,
                                                       l1_regularization_strength=10,
                                                       l2_regularization_strength=10
                                                   ))
        # 训练和评估
        classifier.train(input_fn=LrWithFtrl.get_tfrecords_data)
        res = classifier.evaluate(input_fn=LrWithFtrl.get_tfrecords_data)
        print(res)

if __name__=="__main__":
    # 1.构建TFrecords的输入数据读取函数，输入estimator，输入函数不能有参数，因此采用静态方法1
    LWF = LrWithFtrl()
    LWF.train_eval()
