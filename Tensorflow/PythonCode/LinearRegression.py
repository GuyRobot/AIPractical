import tensorflow as tf
import numpy as np

rand = np.random

learning_rate = .01
training_steps = 1000
display_step = 50
# Train data
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])

# Label
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

W = tf.Variable(rand.randn(), name="weight")
b = tf.Variable(rand.randn(), name="bias")
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)


def linear(x):
    """
    Linear regression
    :param x:
    :return:
    """
    return W * x + b


def mean_square(pred, label):
    """
    Mean square error

    :param pred:
    :return: 1/m * (y - pred)^2
    """
    return tf.reduce_mean(tf.square(Y - pred))


def run_optimization():
    with tf.GradientTape() as g:
        pred = linear(x=X)
        loss = mean_square(pred, Y)

    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


if __name__ == '__main__':

    for step in range(1, training_steps + 1):
        run_optimization()

        if step % display_step == 0:
            pred = linear(X)
            loss = mean_square(pred, Y)
            print("%d Step: loss: %.3f, W: %.3f, b: %.3f" % (step, loss, W.numpy(), b.numpy()))





