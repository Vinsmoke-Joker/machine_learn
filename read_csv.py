import tensorflow as tf
import os
# 1.找到文件，构造一个列表
# 2.构造文件队列
# 3.构造阅读器，读取队列内容（一行）
# 4.解码内容
# 5.批处理
def csvread(filelist):
    """
    读取csv文件
    :param filelist：文件路径+名字的列表
    :return:读取的内容
    """
    # 2.构造文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 3.构造阅读器读取数据
    reader = tf.TextLineReader()
    key,value = reader.read(file_queue)  # 返回key为文件名，value为文件内容
    # 4.解码内容
    # record_defaults指定每一个样本每一列类型，指定默认值
    records = [['None'],['None']]  # 默认该列为字符串类型，缺失值采用None替换
    example,labels = tf.decode_csv(value,field_delim=',',record_defaults=records) # 几列用几个值接受返回值
    # 5.批处理-读取多个数据
    # batch(tensor,batch_size=从队列中读取批处理大小,num_thread=进入队列线程数量，capacity=队列中元素最大数量)
    # 批处理大小和队列数据数量没有影响，只决定这一批次取多少数据
    example_batch,lable_batch = tf.train.batch([example,labels],batch_size=9,num_threads=1,capacity=9)
    return example_batch,lable_batch

if __name__ =='__main__':
    # 1.找到文件，构造一个列表  路径+名字
    file_name = os.listdir('./data/csvdata')
    filelist = [os.path.join('./data/csvdata',file) for file in file_name]
    example,labels = csvread(filelist=filelist)
    # 开启会话
    with tf.Session() as sess:
        # 开启读取文件线程
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord=coord)
        # 打印读取内容
        print(sess.run([example,labels]))
        # 回收线程
        coord.request_stop()
        coord.join(threads)