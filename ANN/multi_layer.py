import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
# 定义结构，前向计算
# 定义一个隐含层，含有300个节点
# 总的为   1×784  784×300  300×10   ======    1×10 结果

def add_layer(inputs, in_size, out_size,keep_prob=1.0,activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        outputs = tf.nn.dropout(outputs,keep_prob)  #随机失活
    return outputs
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)     # 概率

h1 = add_layer(x,784,300,keep_prob,tf.nn.relu)
h2 = add_layer(h1,300,400,keep_prob,tf.nn.relu)
##输出层
w = tf.Variable(tf.zeros([400,10]))     #300*10
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(h2,w)+b)

#定义loss,optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step  =tf.train.AdagradOptimizer(0.35).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))       #高维度的
acuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    #要用reduce_mean

tf.global_variables_initializer().run()
for i in range(3000):
    batch_x,batch_y  = mnist.train.next_batch(100)
    train_step.run({x:batch_x,y_:batch_y,keep_prob:0.75})
    if i%1000==0:
        train_accuracy = acuracy.eval({x:batch_x,y_:batch_y,keep_prob:1.0})
        print("step %d,train_accuracy %g"%(i,train_accuracy))
    
#test

print acuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
