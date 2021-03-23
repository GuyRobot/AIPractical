import random
import string

import mxnet as mx
from mxnet import gluon, nd
import numpy as np

max_num = 999
dataset_size = 60000
seq_len = 5
split = 0.8
batch_size = 512
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

X = mx.random.uniform(low=0, high=max_num, shape=(dataset_size, seq_len)).astype('int32').asnumpy()
Y = X.copy()
Y.sort()  # Let's sort X to get the target

print("Input {}\nTarget {}".format(X[0].tolist(), Y[0].tolist()))

vocab = string.digits + " "
print(vocab)
vocab_idx = {c: i for i, c in enumerate(vocab)}
print(vocab_idx)

max_len = len(str(max_num)) * seq_len + (seq_len - 1)
print("Maximum length of the string: %s" % max_len)


def transform(x, y):
    x_string = ' '.join(map(str, x.tolist()))
    x_string_padded = x_string + ' ' * (max_len - len(x_string))
    x = [vocab_idx[c] for c in x_string_padded]
    y_string = ' '.join(map(str, y.tolist()))
    y_string_padded = y_string + ' ' * (max_len - len(y_string))
    y = [vocab_idx[c] for c in y_string_padded]
    return mx.nd.one_hot(mx.nd.array(x), len(vocab)), mx.nd.array(y)


split_idx = int(split * len(X))
train_dataset = gluon.data.ArrayDataset(X[:split_idx], Y[:split_idx]).transform(transform)
test_dataset = gluon.data.ArrayDataset(X[split_idx:], Y[split_idx:]).transform(transform)

print("Input {}".format(X[0]))
print("Transformed data Input {}".format(train_dataset[0][0]))
print("Target {}".format(Y[0]))
print("Transformed data Target {}".format(train_dataset[0][1]))

train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   last_batch='rollover')
test_data = gluon.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  last_batch='rollover')

net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(
        gluon.rnn.LSTM(hidden_size=128, num_layers=2, layout='NTC', bidirectional=True),
        gluon.nn.Dense(len(vocab), flatten=False)
    )

net.initialize(mx.init.Xavier(), ctx=ctx)
loss = gluon.loss.SoftmaxCELoss()
schedule = mx.lr_scheduler.FactorScheduler(step=len(train_data) * 10, factor=0.75)
schedule.base_lr = 0.01
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01, 'lr_scheduler': schedule})
if __name__ == '__main__':

    epochs = 100
    for e in range(epochs):
        epoch_loss = 0.
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with mx.autograd.record():
                output = net(data)
                l = loss(output, label)

            l.backward()
            trainer.step(data.shape[0])

            epoch_loss += l.mean()

        print("Epoch [{}] Loss: {}, LR {}".format(e, epoch_loss.asscalar() / (i + 1), trainer.learning_rate))
