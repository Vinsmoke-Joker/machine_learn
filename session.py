import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def tensorflow_test():
    con1 = tf.constant(10)
    con2 = tf.constant(20)
    sum = tf.add(con1,con2)
    mut = con1 * con2

    with tf.Session(config=tf.ConfigProto(allow_soft_placement= True, # 是否打印设备分配日志
                                           log_device_placement= True)) as sess: #如果你指定的设备不存在，允许TF自动分配设备
        print(sess.run(sum))
        # 打印多个值
        print(sess.run([con1,con2]))
        # 可以通过eval()方法来获取值，但前提必须有session环境
        # 或者先对session初始化
        # sess = Session()
        # print(mut.eval(session=sess))
        print(mut.eval())
    return None
# placeholder提供占位符，run时候通过feed_dict指定参数
# 可以当做函数的参数来看待
def session_run():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add = tf.add(a,b)
    print("add=",add)
    with tf.Session() as sess:
        print(sess.run([a,b,add],feed_dict={a:3.0,b:4.0}))
    return None
def shape_test():
    a_p = tf.placeholder(dtype=tf.float32,shape=[None,None])
    b_p = tf.placeholder(dtype=tf.float32,shape=[None,10])
    c_p = tf.placeholder(dtype=tf.float32,shape=[3,2])
    print('a_p静态形状为:\n',a_p.get_shape())
    print('b_p静态形状为:\n',b_p.get_shape())
    print('c_p静态形状为:\n', b_p.get_shape())
    # 静态修改
    a_p.set_shape([10,12])
    b_p.set_shape([3,10])
    # 形状固定后，不能再用set_shape()进行修改
    # set_shape()只能修改张量本身的形状,且元素个数不能超过原来tensor元素个数
    # c_p.set_shape([2,3])
    print('修改后,a_p静态形状为:\n', a_p.get_shape())
    print('修改后,b_p静态形状为:\n', b_p.get_shape())
    # 动态修改,会创建新的tensor,不会对原来tensor进行修改
    c_reshape = tf.reshape(c_p,[2,3])
    print('修改前,c_p静态形状为:\n', c_p.get_shape())
    print('修改后,c_reshape静态形状为:\n', c_reshape.get_shape())

def Variable_test():
    # 定义变量
    a = tf.Variable(initial_value=30)
    b = tf.Variable(initial_value=40)
    sum = tf.add(a,b)
    # 初始化变量
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(sum))
if __name__ =='__main__':
    # tensorflow_test()
    # session_run()
    # shape_test()
    Variable_test()