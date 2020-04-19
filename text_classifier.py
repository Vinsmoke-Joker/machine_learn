import tensorflow as tf
from tensorflow import keras

# 指定总共多少不同的词，每个样本的序列长度最大多少
vocab_size = 5000
sentence_size = 200

def text_class():
    """
    训练电影文本-二分类模型
    :return:
    """
    # 1.电影评论数据获取
    def get_train():
        imdb = keras.datasets.imdb
        # 训练集的特征值目标值和测试集的特征值和目标值
        # 填充序列pad_sequences
        # sequences：浮点数或整数构成的两层嵌套列表
        # maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
        # padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
        # value：浮点数，此值将在填充时代替默认的填充值0
        (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=5000)
        x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_sentence,
                                                             padding='post',value=0)
        x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_sentence,
                                                            padding='post',value=0)

        return (x_train,y_train),(x_test,y_test)
    # 2.模型输入特征序列
    # x_train:200个特征值
    def train_input_fn():
        # 文本分类输入函数
        # 定义解析函数
        def parse(x,y):
            feature={'feature':x}
            return feature,y
        dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        dataset = dataset.map(parse)
        dataset.shuffle(buffer_size=25000)
        dataset.batch(32)
        dataset.repeat()
        return dataset
    # 3.模型保存和训练
    (x_train,y_train),(x_test,y_test) = get_train()
    columns = tf.feature_column.categorical_column_with_identity('feature',vocab_size=vocab_size)
    # 不能直接用分类API获取特征列
    # estimator = tf.estimator.DNNClassifier(feature_columns=columen, hidden_units=[512, 256])
    em_columns = tf.feature_column.embedding_column(columns,dimension=50)
    # 模型特征和训练
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[100],
        feature_columns=em_columns,
        model_dir='./temp/ckpt/text_class' # 模型保存路径
    )
    # 训练模型
    estimator(train_input_fn)

if __name__=='__main__':
    text_class()




