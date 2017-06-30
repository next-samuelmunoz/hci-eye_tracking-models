# -*- coding: utf-8 -*-

import utils_data

import numpy as np
import tensorflow as tf


class Model:

    def __init__(self, model_name, model, saved_model=False):
        self.model_name = model_name
        self.get_model = model
        self.IMG_SHAPE = (20,30)
        self.FEATURES = [
            'eye_right_x', 'eye_right_y', 'eye_right_width', 'eye_right_height',
            'eye_left_x', 'eye_left_y', 'eye_left_width', 'eye_left_height',
            'face_x', 'face_y', 'face_width', 'face_height'
        ]
        self.TARGETS = ['x','y']
        # Initialize
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.t_features, self.t_imgs_left, self.t_imgs_right, self.t_labels, self.t_keep_prob = self.get_placeholders(len(self.FEATURES), self.IMG_SHAPE, len(self.TARGETS))
            self.model = self.get_model(self.t_features, self.t_imgs_left, self.t_imgs_right, self.t_keep_prob)
            self.loss = self.get_loss(self.t_labels, self.model)
            self.sess = tf.Session(graph=self.graph)
            if saved_model!=False:
                self.model_load(self.sess, saved_model)


    @staticmethod
    def get_placeholders(n_features, img_shape, n_labels):
        return(
            tf.placeholder(dtype=tf.float32, shape=(None, n_features), name="features"),
            tf.placeholder(dtype=tf.float32, shape=(None, *img_shape), name="left_imgs"),
            tf.placeholder(dtype=tf.float32, shape=(None, *img_shape), name="right_imgs"),
            tf.placeholder(dtype=tf.float32, shape=(None, n_labels), name="labels"),
            tf.placeholder(dtype=tf.float32, name="keep_prob"),
        )


    @staticmethod
    def get_loss(labels, predictions):
        '''Average of euclidean distance between labels and predictions
        '''
        return tf.reduce_mean(
            tf.norm(
                tf.subtract(labels, predictions),
                ord='euclidean',
                axis=1,
            )
        )

    def train(self,
        train_data, train_imgs_left, train_imgs_right,
        validation_data, validation_imgs_left, validation_imgs_right,
        batch_size, epochs, learning_rate, keep_prob=1.0
    ):
        with self.graph.as_default():
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss)
            self.sess.run(tf.global_variables_initializer())
            steps = 0
            for epoch in range(epochs):
                print("Epoch: {} of {}".format(epoch+1, epochs))
                losses_train = []
                for b_data, b_imgs_left, b_imgs_right in utils_data.get_batch(train_data, train_imgs_left, train_imgs_right, batch_size):
                    steps += 1
                    self.sess.run(optimizer, feed_dict={
                        self.t_features: b_data[self.FEATURES],
                        self.t_imgs_left: b_imgs_left,
                        self.t_imgs_right: b_imgs_right,
                        self.t_labels: b_data[self.TARGETS],
                        self.t_keep_prob: keep_prob
                    })
                    # Print Info
                    if steps % 20 == 0:
                        train_loss = self.loss.eval({
                            self.t_features: b_data[self.FEATURES],
                            self.t_imgs_left: b_imgs_left,
                            self.t_imgs_right: b_imgs_right,
                            self.t_labels: b_data[self.TARGETS],
                            self.t_keep_prob: 1.0
                        }, session=self.sess)
                        losses_train.append(train_loss)
                        validation_loss = self.loss.eval({
                            self.t_features: validation_data[self.FEATURES],
                            self.t_imgs_left: validation_imgs_left,
                            self.t_imgs_right: validation_imgs_right,
                            self.t_labels:validation_data[self.TARGETS],
                            self.t_keep_prob: 1.0

                        }, session=self.sess)
                        print("\tTrain VS Validation: {} {}".format(train_loss, validation_loss))
                self.model_save(self.sess, self.model_name+"."+str(epoch).zfill(4))  # Save after each epoch
                validation_loss = self.loss.eval({
                    self.t_features: validation_data[self.FEATURES],
                    self.t_imgs_left: validation_imgs_left,
                    self.t_imgs_right: validation_imgs_right,
                    self.t_labels:validation_data[self.TARGETS],
                    self.t_keep_prob: 1.0

                }, session=self.sess)
                print("Train: {}".format(np.mean(losses_train)))
                print("Validation: {}\n\n".format(validation_loss))
            self.model_save(self.sess, self.model_name+".final")


    @staticmethod
    def model_save(session, name):
        saver = tf.train.Saver()
        save_path = saver.save(session, "data/models/"+name)


    @classmethod
    def model_load(cls, session, name):
        saver = tf.train.Saver()
        saver.restore(session, "data/models/"+name)


    def test(self, test_data, test_imgs_left, test_imgs_right):
        test_loss = self.loss.eval({
            self.t_features: test_data[self.FEATURES],
            self.t_imgs_left: test_imgs_left,
            self.t_imgs_right: test_imgs_right,
            self.t_labels: test_data[self.TARGETS],
            self.t_keep_prob: 1.0
        }, session=self.sess)
        return test_loss


    def predict(self, data, imgs_left, imgs_right):
        return self.sess.run(self.model, {
            self.t_features: data[self.FEATURES],
            self.t_imgs_left: imgs_left,
            self.t_imgs_right: imgs_right,
            self.t_keep_prob: 1.0
        })
