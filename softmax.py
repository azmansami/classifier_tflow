import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version "+ tf.__version__)
tf.set_random_seed(0)


# Downloading data
# hand written digit image - MNIST dataset
mnist = mnist_data.read_data_sets("data",one_hot=True, reshape=False, validation_size=0)
# model
# y=softmax(x*w+b)

# x: Input data placeholder
x = tf.placeholder(tf.float32,[None,28,28,1],name='input')
tf.summary.image('input',x,5)

# y_: Actual label
y_= tf.placeholder(tf.float32,[None,10],name='y_actual')

# w: weights 784,10 (28*28 = 784)
# w = tf.Variable(tf.random_uniform([784,10],minval=0,maxval=1),name='weights')
# initializnig w has impact on where the w converges.
# a small initialization of w is best
w = tf.Variable(tf.zeros([784,10]),name='weights')
tf.summary.histogram('weight',w)

# b = biases
b=tf.Variable(tf.zeros([10]),name='biases')
tf.summary.histogram('bias',b)

# flatten the image into a single vector
xx = tf.reshape(x,[-1,784],name='input_flat')

# The model graph
with tf.name_scope('softmax_linear_model'):
    y = tf.nn.softmax(tf.matmul(xx,w)+b)

# total cross-entropy (softmax) across all input
with tf.name_scope('cross_entropy_loss'):
    cross_entropy = -tf.reduce_mean(y_*tf.log(y)) * 1000
    # since the formula above evaluates to very small number (near 0)
    # we scale the loss by multiplying with 1000 and it helps minimizing the
    # loss better
    #cross_entropy += 0.001 * tf.nn.l2_loss(w)
tf.summary.scalar('loss',cross_entropy)

# accuracy of training model, 0 worst, 1 best
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)


merged=tf.summary.merge_all()


for learning_rate in [0.002,0.005,0.008]:
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # setting up tensorboard filewriter
    paramstr_train="train,lr="+str(learning_rate)
    paramstr_test="test,lr="+str(learning_rate)
    train_writer=tf.summary.FileWriter("/tmp/tboard/softmax/"+paramstr_train,sess.graph)
    test_writer=tf.summary.FileWriter("/tmp/tboard/softmax/"+paramstr_test)

    test_data={x:mnist.test.images, y_:mnist.test.labels}

    for i in range(1000):
        # mini-batch gradient mode
        batch_x, batch_y = mnist.train.next_batch(100)
        train_data={x:batch_x,y_:batch_y}
        # train
        summary,_=sess.run([merged,train_step],feed_dict=train_data)
        train_writer.add_summary(summary,i)

        #if i % 20 == 0:
        # check performance on test data
        test_summary,ts_a=sess.run([merged,accuracy],feed_dict=test_data)
        test_writer.add_summary(test_summary,i)
        #print("Test accuracy: ",ts_a)
test_summary,ts_a=sess.run([merged,accuracy],feed_dict=test_data)
test_writer.add_summary(test_summary,i)
print("Test accuracy: ",ts_a)
