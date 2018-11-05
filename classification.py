import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        return Wx_plus_b if (activation_function == None) else activation_function(Wx_plus_b)

def compute_accuracy(v_xs, v_ys):
        global prediction
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})

#define placeholder form inputs to network
xLen = 28 * 28
yLen = 10
xs = tf.placeholder(tf.float32, [None, xLen])  # 28*28
ys = tf.placeholder(tf.float32, [None, yLen])

prediction = add_layer(xs, xLen, yLen, activation_function=tf.nn.softmax)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
print(ax)
plt.ion()
plt.show()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
        if (i % 50 == 0):
                print(compute_accuracy(
                        mnist.test.images,
                        mnist.test.labels
                ))
                try:
                        ax.lines.remove(lines[0])
                except Exception:
                        pass
                result = sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys})
                lines = ax.plot(batch_xs, result, 'r-', lw=5)
                plt.pause(0.1)
