import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def tensorflow_test():
    a = 10
    b = 20
    c = a + b
    print(c)

    con1 = tf.constant(10,name='a')
    con2 = tf.constant(20,name='b')
    sum_t = tf.add(con1,con2,name='sum')
    print(sum_t)
    # 打印默认图
    print(tf.get_default_graph())
    print(con1.graph)
    print(con2.graph)
    print(sum_t.graph)

    # 创建一个自定义图
    new_g = tf.Graph()
    with new_g.as_default():
        new_a = tf.constant(10)
        new_b = tf.constant(20)
        sum_n = tf.add(new_a, new_b)
    print(new_a.graph)
    print(new_b.graph)
    print(sum_n.graph)


    # 开启会话
    # Session(graph=指定运行该图,不传默认为默认图,而非自定义图)
    # with tf.Session(graph=new_g) as sess:
    with tf.Session() as sess:
        print(sess.run(sum_t))

        # board可视化,先序列化为events文件
        # sess.graph 获取默认图即tf.get_default_graph()
        # 这将在指定目录中生成一个event文件，其名称格式如下：
        # events.out.tfevents.{timestamp}.{hostname}
        file_writter = tf.summary.FileWriter('./temp/summary/',graph=sess.graph)
        print(sess.graph)

if __name__ =="__main__":
    tensorflow_test()