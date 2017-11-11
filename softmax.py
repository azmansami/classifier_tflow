import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version "+ tf.__version__)
tf.set_random_seed(0)


# Downloading data

mnist = mnist_data.read_data_sets("data",one_hot=True, reshape=False, validation_size=0)
#model
#   y=softmax(x*w+b)

# x: Input data placeholder
x = tf.placeholder(tf.float32,[None,28,28,1],name='input')

# y_: Actual label
y_= tf.placeholder(tf.float32,[None,10],name='y_actual')

# w: weights 784,10 (28*28 = 784)
w = tf.Variable(tf.zeros([784,10]),name='weights')

# b = biases
b=tf.Variable(tf.zeros([10]),name='biases')

# flatten the image into a single vector
xx = tf.reshape(x,[-1,784],name='input_flat')

# The model graph
with tf.name_scope('softmax_linear_model'):
    y = tf.nn.softmax(tf.matmul(xx,w)+b)

# total cross-entropy (softmax) across all input
with tf.name_scope('cross_entropy_loss'):
    cross_entropy = -tf.reduce_mean(y_*tf.log(y)) * 1000
tf.summary.scalar('train_loss',cross_entropy)

# accuracy of training model, 0 worst, 1 best
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('train_accuracy',accuracy)

# training with learning rate of 0.005
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

merged=tf.summary.merge_all()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# setting up tensorboard filewriter
train_writer=tf.summary.FileWriter('/tmp/tboard/train',sess.graph)
#test_writer=tf.summary.FileWriter('/tmp/tboard/test')


for i in range(1000):
    # stochastic gradient mode
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data={x:batch_x,y_:batch_y}

    #train
    summary,_=sess.run([merged,train_step],feed_dict=train_data)
    train_writer.add_summary(summary,i)

a,c=sess.run([accuracy,cross_entropy],feed_dict=train_data)
print("Train minimized loss: ", c," accuracy: ",a)

# success on test data
test_data={x:mnist.test.images, y_:mnist.test.labels}
a,c=sess.run([accuracy,cross_entropy],feed_dict=test_data)
print("Test loss: ", c," accuracy: ",a)
