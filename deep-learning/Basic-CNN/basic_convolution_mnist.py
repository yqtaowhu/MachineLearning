import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
# w,b,可以复用，因此设为函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
# 卷积层
# x输入,W卷积参数[5,5,1,32]  5*5的卷积核，1个深度，32个卷积核
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_22(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
# 卷积，relu,池化
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_22(h_conv1)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_22(h_conv2)    # 7×7×64
# 全连接层 1024
W_fc1 = weight_variable([7*7*64,1024])    #1d
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
W_fc2 = weight_variable([1024,10])
b_fc2 = weight_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#定义loss,optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step  =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

##
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))       #高维度的
acuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    #要用reduce_mean
for i in range(30000):
    batch_x,batch_y  = mnist.train.next_batch(50)
    if i%1000==0:
        train_accuracy = acuracy.eval({x:batch_x,y_:batch_y,keep_prob:1.0})
        print("step %d,train_accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})
#test

print acuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
