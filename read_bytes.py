import tensorflow as tf
import os
# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cifar_dir','./data/summary/cifar','文件的目录')
tf.app.flags.DEFINE_string('cifar_tfrecords','./temp/summary/tfrecords/cifar.tfrecords','存入tfrecords文件的目录')

class CifarRead(object):
    """
    完成读取二进制文件，写进tfrecords，读取tfrecords
    """
    def __init__(self,filelist):
        # 文件列表
        self.file_list = filelist
        # 读取图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_bytes = 1  # 存储的每个二进制图片标签字节
        self.image_bytes = self.height*self.width*self.channel # 存储的每个二进制图片的图像字节
        self.bytes = self.label_bytes + self.image_bytes  # 存储的每个二进制图片的总字节数
    def read_and_decode(self):
        # 1.构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)
        # 2.构造二进制文件读取器-需传入每个样本的字节数
        reader = tf.FixedLengthRecordReader(self.bytes)
        key,value= reader.read(file_queue)
        # 3.解码内容-二进制文件内容解码
        label_image = tf.decode_raw(value,tf.uint8)
        # 4.分割出图片和标签数据，切出特征值和目标值
        # tf.slice(目标, [start], [stop])
        # 由于label_image 存储时候为uint8类型，计算时候需要int32类型，因此进行cast转换
        labels = tf.cast(tf.slice(label_image,[0],[self.label_bytes]),tf.int32)
        image = tf.slice(label_image,[self.label_bytes],[self.image_bytes])
        # 5.可以对图片的特征数据进行形状改变，[3072]-->[32,32,3]
        image_reshape = tf.reshape(image,[self.height,self.width,self.channel])
        # 6.批处理
        image_batch,label_batch = tf.train.batch([image_reshape,labels],num_threads=1,batch_size=10,capacity=10)
        return image_batch,label_batch
    def write_to_tfrecords(self,image_batch,label_batch):
        """
        将图片的特征值和目标值存入tfrecords
        :param image_batch:张图片的特征值
        :param label_batch:图片目标值
        :return:None
        """
        # 1.构造一个tfrecords存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)
        # 2.循环将所有样本写入文件,每张图片构造example协议
        for i in range(10):
            # 取出第i个图片的特征值和目标值
            image = image_batch[i].eval().tostring()
            label = label_batch[i].eval()[0]
            # 构造样本的example协议
            example = tf.train.Example(features = tf.train.Features(feature={
                'image':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            # 写入单独样本
            writer.write(example.SerializeToString())
        # 关闭
        writer.close()
    def read_from_tfrecords(self):
        """
        :return:
        """
        # 1.构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])
        # 2.构造文件阅读器，读取内容example
        reader = tf.TFRecordReader()
        key,value = reader.read(file_queue)
        # 3.解析example
        features = tf.parse_single_example(value,features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64)
        })

        # 4.解码内容,读取类型为string需要解码，其他不需要
        image = tf.decode_raw(features['image'],tf.uint8)
        label = features['label']
        # 固定图片形状，方便批处理
        image_reshape = tf.reshape(image,[self.height,self.width,self.channel])
        label = tf.cast(label,tf.int32)
        # 5.批处理
        image_batch,label_batch = tf.train.batch([image_reshape,label],num_threads=1,batch_size=10,capacity=10)
        return image_batch,label_batch

if __name__=='__main__':
    # 1.获取文件名+路径列表
    file_name = os.listdir(FLAGS.cifar_dir)
    file_list = [os.path.join(FLAGS.cifar_dir,file) for file in file_name if file[-3:]=='bin']
    # 2.实例化对象
    cr = CifarRead(file_list)
    # 从文件中读取
    # image_batch,label_batch = cr.read_and_decode()
    # 从tfrecords中读取
    image_batch,label_batch = cr.read_from_tfrecords()
    # 3.开启会话
    with tf.Session() as sess:
        # 构造线程协调器
        coord = tf.train.Coordinator()
        # 开启子线程
        threads = tf.train.start_queue_runners(sess,coord=coord)


        # 存入tfrecords文件
        # print('开始存储')
        # cr.write_to_tfrecords(image_batch,label_batch)
        # print('存储完毕')


        # 打印结果
        print(sess.run([image_batch,label_batch]))
        # 关闭线程
        coord.request_stop()
        coord.join()