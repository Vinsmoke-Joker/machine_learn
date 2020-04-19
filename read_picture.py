import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def read_picture(filelist):
    """
    读取狗图片，并转换为tensor
    :param filelist: 文件名+路径的列表
    :return:每张图片张量
    """
    # 1.构造文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 2.构造读取器-默认读取一张图片
    reader = tf.WholeFileReader()
    # 读取内容-返回key为文件名，value为文件内容
    key,value = reader.read(file_queue)
    # 3.对读取数据图片进行解码
    image = tf.image.decode_jpeg(value)
    # 4.进行批处理-统一图片大小
    image_resize = tf.image.resize_images(image,[200,200])
    # 注意：一定要把样本形状固定[200,200,3],在批处理时候，要求所有数据形状必须需定义
    image_resize.set_shape([200,200,3])
    # 进行批处理
    image_batch = tf.train.batch([image_resize],batch_size=20,num_threads=1,capacity=20)
    return image_batch
if __name__=='__main__':
    # 1.找到文件，放入列表
    file_name = os.listdir('./data/summary/images') # 返回文件名字列表
    # 将文件名与路径拼接为列表
    filelist = [os.path.join('./data/summary/images',file) for file in file_name]
    image_batch = read_picture(filelist)
    # 开启会话
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启文件读取线程
        threads = tf.train.start_queue_runners(sess,coord=coord)
        # 打印读取内容
        print(sess.run([image_batch]))
        # 关闭线程
        coord.request_stop()
        coord.join()