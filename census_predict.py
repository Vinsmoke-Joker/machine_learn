import tensorflow as tf
import os
# 普查数据列名
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
] # 指定特征值列名
# 解码数据，默认值
# 读取数据时候进行默认值处理，int类型为0，str类型为空
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
# 指定训练和测试数据目录
train_file = './data/adult.data'
test_file = './data/adult.test'
# 1.读取数据函数 input_func（文件名，循环训练次数，批处理大小）
def input_func(file,epoches,batch_size):
    """
    tf.data读取数据，并处理数据格式，返回dataset迭代器
    :return:
    """
    # 定义数据处理函数
    def decode_train_data(row):
        # 解析文本数据 tf.decode_csv()
        data = tf.decode_csv(row,record_defaults=_CSV_COLUMN_DEFAULTS)
        # 进行字典映射，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
        feature_dict = dict(zip(_CSV_COLUMNS,data))
        # 提取最后一列目标列
        label = feature_dict.pop('income_bracket')
        # tf.equal（a,b） 若str_a == str_b ,返回1，否则0
        labels = tf.equal(label,">50K")
        # 返回：特征数据组成的字典，目标值
        return feature_dict,labels


    dataset = tf.data.TextLineDataset(file)
    # map,repeat,batch tf.data
    dataset = dataset.map(decode_train_data)
    # repeat 重复所有样本次数即循环训练次数，batch_size:1000,16/32/64/128个样本一组，分N组 Batch Size定义：分批训练，一次训练所选取的样本数。
    dataset = dataset.repeat(epoches)
    dataset = dataset.batch(batch_size)
    return dataset
# 2.模型选择和特征处理，数据必须每列指定类别：连续列和类别列即连续型和离散型
def get_feature_column():
    # 对于普查数据进行特征列指定
    # 连续型特征 --连续列
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    numeric_column = [age,education_num,capital_gain,capital_loss,hours_per_week]
    # 离散型--类别列
    # tf.feature_column.categorical_column_with_vocabulary_list： 指定所有类别字符串，对应到具体类别
    # tf.feature_column.categorical_column_with_hash_bucket：类别数量过多不确定到底有多少类别，指定一个hash_bucket_size作为上限
    workclass = tf.feature_column.categorical_column_with_vocabulary_list\
        ('workclass',[
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=1000)
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list\
        ('marital_status',['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    education = tf.feature_column.categorical_column_with_vocabulary_list\
            ('education',[
                'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list\
            (
            'relationship',[
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
    categorical_columns = [relationship, occupation, education, marital_status, workclass]
    return numeric_column + categorical_columns

if __name__ == '__main__':
    # 1、读取美国普查收入数据
    # 打印出来的是一批次的数据形状（32，）
    # print(input_func(train_file,3,32))
    # tran_file文件中的样本，循环训练3次即共3000个样本，每组32个样本
    # 2、模型选择特征并进行特征工程处理
    columns = get_feature_column()
    # 3、模型训练与评估
    estimator = tf.estimator.LinearClassifier(feature_columns=columns)
    # estimator.train或者evaluate
    # 训练数据输入函数不能有参数，采用functools解决
    import functools
    # input_func三个参数，通过partial(函数名，该函数的参数) 指定默认值，调用时候不需要再传递参数
    train_input = functools.partial(input_func,train_file,epoches=3,batch_size=32) # 训练集
    test_input = functools.partial(input_func, test_file, epoches=1, batch_size=32)  # 测试集
    # 训练评估
    estimator.train(train_input)
    res = estimator.evaluate(test_input)
    print(res)
    for key,value in sorted(res.items()):
        print('%s,%s,'%(key,value))

