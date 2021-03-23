import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

num_classes = 10  # 10 images digit
num_features = 784  # 28 x 28 (img dimension)

learning_rate = .1
training_steps = 1000
batch_size = 256
display_step = 50

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Prepare data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
# Flatten data
x_train, x_test = np.reshape(x_train, (-1, num_features)), np.reshape(x_test, (-1, num_features))

# Normalize
x_train, x_test = x_train / 255, x_test / 255

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(2000).batch(batch_size).prefetch(1)

# Variables

W = tf.Variable(tf.ones((num_features, num_classes)), name="weights")
b = tf.Variable(tf.zeros(num_classes), name="bias")
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)


def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


def cross_entropy_error(y_pred, y_true):
    # y_true (256, 10)
    y_true = tf.one_hot(y_true, depth=num_classes)

    # y_pred (256, 10)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1)

    # (256, 10) -> sum(axis=1) -> (256,) -> reduce_mean (1,)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1), axis=0)


def accuracy(predict, label):
    correct = tf.equal(tf.argmax(predict), tf.cast(label, dtype=tf.int64))
    return tf.reduce_mean(tf.cast(correct, dtype=tf.float32))


def run_optimizer(x, y):
    with tf.GradientTape() as g:
        y_pred = logistic_regression(x)
        loss = cross_entropy_error(y_pred, y)

    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


if __name__ == '__main__':
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        run_optimizer(batch_x, batch_y)

        if step % display_step == 0:
            pred = logistic_regression(batch_x)
            loss = cross_entropy_error(pred, batch_y)

            print("Step: %d, loss: %.3f" % (step, loss))

    import matplotlib.pyplot as plt

    start = 10
    n_images = 5
    test_images = x_test[start:start + n_images]
    predictions = logistic_regression(test_images)

    for i in range(n_images):
        plt.imshow(np.reshape(test_images[i], (28, 28)),
                   cmap='gray')
        plt.show()
        print("Model Predictions: %d" % np.argmax(predictions.numpy()[i]))
