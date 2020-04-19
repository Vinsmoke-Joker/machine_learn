import tensorflow as tf
import functools
# 美国普查数据神经网络分类
# 普查数据列名
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
# 读取数据时候进行默认值处理，int类型为0，str类型为空
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
#  指定训练和测试数据目录
train_file = './data/adult.data'
test_file = './data/adult.test'
# 1.读取数据函数 input_func（文件名，循环训练次数，批处理大小）
def get_func(file,epoches,batch_size):
    # 数据处理函数
    def decode_train_data(row):
        # 缺失值处理
        data = tf.decode_csv(row,record_defaults=_CSV_COLUMN_DEFAULTS)
        # 构建特征字典
        feature_dict = dict(zip(_CSV_COLUMNS,data))
        # 构建目标列
        label = feature_dict.pop('income_bracket')
        # 特征处理
        labels = tf.equal(label,'>50K')
        return feature_dict,labels
    dataset = tf.data.TextLineDataset(file)
    dataset = dataset.map(decode_train_data)
    dataset = dataset.repeat(epoches)
    dataset = dataset.batch(batch_size)
    return dataset
# 2.模型选择和特征处理，数据必须每列指定类别：连续列和类别列即连续型和离散型
def get_feature_column_v1():
    # 连续类
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    numeric_columns = [age,education_num,capital_gain,capital_loss,hours_per_week]
    # 类别类
    workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass',['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov','Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=1000)
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list \
        ('marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    education = tf.feature_column.categorical_column_with_vocabulary_list \
        ('education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list \
            (
            'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
    # 离散型特征列需要进行embedding_column或者indicator column处理，不能直接输入DNN模型中以categorical类别
    # dimension：大于0的整型数值，用来指定embedding的维度。
    categorical_columns = [tf.feature_column.embedding_column(relationship,dimension=6),
                           tf.feature_column.embedding_column(occupation,dimension=1000),
                           tf.feature_column.embedding_column(education, dimension=16),
                           tf.feature_column.embedding_column(marital_status, dimension=7),
                           tf.feature_column.embedding_column(workclass, dimension=9)]
    return numeric_columns+categorical_columns

if __name__ =='__main__':
    # 模型选择特征并进行特征工程处理
    columen = get_feature_column_v1()
    # 训练模型-神经网络二分类
    # hidden_units 指定有多少个隐藏层，及每个隐藏层中神经元个数
    estimator = tf.estimator.DNNClassifier(feature_columns=columen,hidden_units=[512,256])
    # 训练数据输入函数不能有参数
    train_input = functools.partial(get_func,train_file,epoches=1, batch_size=32)
    test_input = functools.partial(get_func,test_file,epoches=1,batch_size=32)
    # 训练
    estimator.train(train_input)
    # 评估
    res = estimator.evaluate(test_input)
    for key,values in sorted(res.items()):
        print('%s:%s'%(key,values))
