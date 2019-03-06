import sys, time, argparse
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()


# Reading data
data_file = os.path.expanduser('~/data/PhisioNet/MIMIC/processed/out_binary.matrix')
model_path ='weights/ae/weights.ckpt'
data = np.load(data_file)
inputDim = data.shape[1]
data = np.clip(data, 0, 1)


# Params
_VALIDATION_RATIO = 0.2
num_epochs = 15
batch_size = 100
learning_rate = 1e-3
h_dim = inputDim
comressedDim = 128

# Divide into train/test, validation
trainX, X_test = train_test_split(data, test_size=_VALIDATION_RATIO, random_state=1)
trainX, validX = train_test_split(trainX, test_size=_VALIDATION_RATIO, random_state=1)

class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = tf.keras.layers.Dense(comressedDim)
        self.fc2 = tf.keras.layers.Dense(inputDim)

    def encode(self, x):
        h = tf.nn.tanh(self.fc1(x))
        return h

    def decode_logits(self, z):
        h = self.fc2(z)
        return h

    def decode(self, z):
        return self.decode_logits(z)

    def call(self, inputs, training=None, mask=None):
        encoded = self.encode(inputs)
        x_reconstructed_logits = self.decode_logits(encoded)
        return x_reconstructed_logits


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = AE()
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # create train database iterator
    train_dataset = tf.data.Dataset.from_tensor_slices(trainX)
    train_dataset = train_dataset.shuffle(trainX.shape[0])
    train_dataset = train_dataset.batch(batch_size)

    # create validation database iterator
    validation_dataset = tf.data.Dataset.from_tensor_slices(validX)
    validation_dataset = validation_dataset.shuffle(validX.shape[0])
    validation_dataset = validation_dataset.batch(batch_size)

    # IVESTIGATE
    # train_dataset = train_dataset.prefetch(10)

    num_batches = trainX.shape[0] // batch_size
    num_batches_validation = validX.shape[0] // batch_size

    for epoch in range(num_epochs):
        for (batch, data) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                # Forward pass
                x_reconstruction_logits= model(data)

                # Reconstruction loss
                reconstruction_loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(labels=data, logits=x_reconstruction_logits)
                # Sum over all features and mean over the batch.
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss_matrix,1),0)

                # Backprop and optimize
                loss = reconstruction_loss

            gradients = tape.gradient(loss, model.variables)
            grad_vars = zip(gradients, model.variables)
            optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

            if (batch + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, batch + 1, num_batches, loss.numpy(),))

            if batch > num_batches:
                break

        saver = tfe.Saver(model.variables)
        saver.save('weights/ae/weights.ckpt')

        # evaluate
        loss_validation = 0
        for (batch, valid_data) in enumerate(validation_dataset):
            x_reconstruction_logits = model(valid_data)

            # Reconstruction loss
            reconstruction_loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_data,
                                                                                 logits=x_reconstruction_logits)
            # Sum over all features and mean over the batch.
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss_matrix, 1), 0)

            # Backprop and optimize
            loss = reconstruction_loss
            loss_validation += loss

            if (batch + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss Validation: {:.4f}"
                      .format(epoch + 1, num_epochs, batch + 1, num_batches_validation, loss.numpy(),))

            if batch > num_batches:
                break
        print("Loss Validation Average: {:.4f}".format(loss_validation/float(num_batches_validation)))


