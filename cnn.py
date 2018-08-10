import datetime
import os

import numpy as np
import tensorflow as tf

# from dataset import FashionDataset
from network_builder import NetworkBuilder

from train_data_split import train_x, train_y

class ImageCNN:
    def __init__(self, input_dim, output_dim, training=True):
        self.training = training
        nb = NetworkBuilder()
        with tf.name_scope("Input"):
            self.input = tf.placeholder(
                tf.float32, shape=[None, input_dim , input_dim, 1], name="input")

        with tf.name_scope("Output"):
            self.output = tf.placeholder(
                tf.float32, shape=[None, output_dim], name="output")

        with tf.name_scope("ImageModel"):
            model = self.input
            model = nb.add_batch_normalization(model, self.training)
            model = nb.add_conv_layer(
                model, output_size=64, feature_size=(4, 4), padding='SAME', activation=tf.nn.relu)
            model = nb.add_max_pooling_layer(model)
            model = nb.add_dropout(model, 0.1, self.training)
            model = nb.add_conv_layer(model, 64, feature_size=(
                4, 4), activation=tf.nn.relu, padding='VALID')
            model = nb.add_max_pooling_layer(model)
            model = nb.add_dropout(model, 0.3, self.training)
            model = nb.flatten(model)
            model = nb.add_dense_layer(model, 256, tf.nn.relu)
            model = nb.add_dropout(model, 0.5, self.training)
            model = nb.add_dense_layer(model, 64, tf.nn.relu)
            model = nb.add_batch_normalization(model, self.training)
            self.logits = nb.add_dense_layer(
                model, output_dim, activation=tf.nn.softmax)

    def train(self, num_epochs=15, batch_size=125):
        if self.training:
            entro = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.output)
            cost = tf.reduce_mean(entro)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            training_op = optimizer.minimize(cost)
            with tf.name_scope("eval"):
                correct = tf.equal(tf.argmax(self.logits, 1),
                                   tf.argmax(self.output, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            total_size = train_x.shape[0]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # writer = tf.summary.FileWriter('logs', sess.graph)
                # writer.close()
                # saver = tf.train.Saver()
                num_batches = total_size // batch_size
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    # start_index = 0
                    epoch_accuracy = 0
                    for i in range(num_batches):
                        x_batch = train_x[i*batch_size:(i+1)*batch_size, :, :, :]
                        y_batch = train_y[i*batch_size:(i+1)*batch_size, :]
                        _, train_cost = sess.run([training_op, cost], feed_dict={self.input: x_batch, self.output: y_batch})
                        train_accuracy = sess.run(accuracy, feed_dict={self.input: x_batch, self.output: y_batch})
                        epoch_loss += train_cost
                        epoch_accuracy += train_accuracy
                        # start_index = start_index + batch_size
                    epoch_loss /= num_batches
                    epoch_accuracy /= num_batches
                    print("Epoch: {} Cost: {} accuracy: {} ".format(
                        epoch + 1, np.squeeze(epoch_loss), epoch_accuracy))

                # print("Training:End")
                # # Cross validation loss and accuracy
                # cv_loss, cv_accuracy = sess.run(
                #     [loss, accuracy], {self.input: dset.test_x, self.output: dset.test_y})
                # print("Cross validation loss: {} accuracy: {}".format(
                #     np.squeeze(cv_loss), cv_accuracy))
                # saver.save(sess, "output/")
        else:
            raise ValueError("Class set in training mode")

    def predict(self, image):
        pass


def main():
    # dset = FashionDataset(dim=28)
    cnn = ImageCNN(28, 10)
    cnn.train()


if __name__ == '__main__':
    main()
