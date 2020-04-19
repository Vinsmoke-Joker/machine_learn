import tensorflow as tf
FEATURE_COLUMNS = ['channel_id', 'vector', 'user_weights', 'article_weights']
class wdl():
    def __init__(self):
        pass
    @staticmethod
    def get_tfrecords_data():
        def deal_with_tfrecords(example_proto):
            feature = {
                'label':tf.FixedLenFeature([],tf.int64),
                'feature':tf.FixedLenFeature([],tf.string)
            }
            # parse_single_example(解析数据，解析规则dict)
            label_feature = tf.parse_single_example(example_proto,feature)
            # label_feature字典，如果是str类型需要tf.decode_raw()解码,其他类型不需要
            feature = tf.decode_raw(label_feature['feature'],tf.float64)
            # 对特征值处理
            # 1.确定形状121
            # 2.求出文章向量，用户权重，文章权重的平均值
            feature_reshape = tf.reshape(tf.cast(feature,tf.float32),[1,121])
            channel_id = tf.cast(tf.slice(feature_reshape,[0,0],[1,1]),tf.int32)
            vector = tf.reduce_sum(tf.slice(feature_reshape, [0, 1], [1, 100]),axis=1)
            user_weights = tf.reduce_sum(tf.slice(feature_reshape, [0, 101], [1, 10]), axis=1)
            article_weight = tf.reduce_sum(tf.slice(feature_reshape, [0, 111], [1, 10]), axis=1)
            # 返回固定的特征字典和目标值
            data = [channel_id,vector,user_weights,article_weight]
            feature_dict = dict(zip(FEATURE_COLUMNS,data))
            label = tf.cast(label_feature['label'],tf.float32)
            return feature_dict,label
        dataset = tf.data.TFRecordDataset(['./data/train_ctr_20190605.tfrecords'])
        dataset = dataset.map(deal_with_tfrecords)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(64)
        return dataset
    def build_estimator(self):
        """
        构建特征列
        :return:
        """
        # 离散型
        channel_id = tf.feature_column.categorical_column_with_identity('channel_id', num_buckets=25)
        # 连续型
        vector = tf.feature_column.numeric_column('vector')
        user_weights = tf.feature_column.numeric_column('user_weights')
        article_weight = tf.feature_column.numeric_column('article_weights')
        # wide侧
        wide_column = [channel_id]
        # deep侧
        deep_column = [
            tf.feature_column.embedding_column(channel_id,dimension=25),
            vector,
            user_weights,
            article_weight
        ]
        estimator = tf.estimator.DNNLinearCombinedClassifier(
            model_dir='./temp/ckpt/wide_deep',
            linear_feature_columns=wide_column,
            dnn_feature_columns=deep_column,
            dnn_hidden_units=[256,128,64]
        )
        return estimator

if __name__ == '__main__':
    # 构建模型训练
    wdl = wdl()
    estimator = wdl.build_estimator()
    estimator.train(input_fn=wdl.get_tfrecords_data)
    res = estimator.evaluate(input_fn=wdl.get_tfrecords_data)
    for key,value in res.items():
        print('%s:%s'%(key,value))
    # 进行模型的savedmodel导出
    # 1.指定特征列
    wide_column = [tf.feature_column.categorical_column_with_identity('channel_id',num_buckets=25)]
    deep_column = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity('channel_id',num_buckets=25),dimension=25),
        tf.feature_column.numeric_column('vector'),
        tf.feature_column.numeric_column('user_weights'),
        tf.feature_column.numeric_column('article_weights')
    ]
    columns = wide_column + deep_column
    # 2.定义一个模型服务接收数据，解析数据的函数
    # 使用example协议解析
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model('./save_model/wdl',serving_input_receiver_fn)