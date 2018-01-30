#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

from collections import namedtuple

# ==================
#  Global Variables
# ==================
LOSS_LOG_INITIAL_VALUE = 0.1


# ===========
#  Functions
# ===========
def weight_variable(shape, variableName):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.01, dtype=tf.float32)  # Recommend by Vitor Guzilini
    # initial = tf.truncated_normal(shape, mean=0.00005, stddev=0.0001, dtype=tf.float32) # Nick, try to avoid generate negative values
    # initial = tf.truncated_normal(shape, stddev=10.0)                                   # Test

    return tf.Variable(initial, name=variableName)


def bias_variable(shape, variableName):
    initial = tf.constant(0.0, dtype=tf.float32, shape=shape)  # Recommend by Vitor Guizilini
    # initial = tf.constant(0.1, dtype=tf.float32, shape=shape) # Nick
    # initial = tf.constant(10.0, dtype=tf.float32, shape=shape) # Test

    return tf.Variable(initial, name=variableName)


def gradient_x(img):
    gx = img[:, :, :-1] - img[:, :, 1:]

    # Debug
    # print("img:", img.shape)
    # print("gx:",gx.shape)

    return gx


def gradient_y(img):
    gy = img[:, :-1, :] - img[:, 1:, :]

    # Debug
    # print("img:", img.shape)
    # print("gy:",gy.shape)

    return gy


# ===================
#  Class Declaration
# ===================
class Coarse(object):
    def __init__(self, image_height, image_width, depth_height, depth_width, fc_hiddenNeurons):
        # Weights and Biases
        self.Wh1 = weight_variable([11, 11, 3, 96], "c_Wh1")
        self.bh1 = bias_variable([96], "c_bh1")

        self.Wh2 = weight_variable([5, 5, 96, 256], "c_Wh2")
        self.bh2 = bias_variable([256], "c_bh2")

        self.Wh3 = weight_variable([3, 3, 256, 384], "c_Wh3")
        self.bh3 = bias_variable([384], "c_bh3")

        self.Wh4 = weight_variable([3, 3, 384, 384], "c_Wh4")
        self.bh4 = bias_variable([384], "c_bh4")

        self.Wh5 = weight_variable([3, 3, 384, 256], "c_Wh5")
        self.bh5 = bias_variable([256], "c_bh5")

        Wh5_outputSize_height = round(image_height / 32) + 1
        Wh5_outputSize_width = round(image_width / 32)

        self.Wh6 = weight_variable([Wh5_outputSize_height * Wh5_outputSize_width * 256, fc_hiddenNeurons], "c_Wh6")
        self.bh6 = bias_variable([fc_hiddenNeurons], "c_bh6")

        self.depth_numPixels = depth_height * depth_width
        assert (
                fc_hiddenNeurons == self.depth_numPixels), "The number of Neurons must be iqual to the number of output pixels."

        self.Wh7 = weight_variable([fc_hiddenNeurons, self.depth_numPixels], "c_Wh7")
        self.bh7 = bias_variable([self.depth_numPixels], "c_bh7")

        # Layers
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None

        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.hidden4 = None
        self.hidden5 = None
        self.hidden6 = None
        self.hidden7 = None

        self.pool1 = None
        self.pool2 = None

        self.fc1 = None
        self.fc2 = None


class Fine(object):
    def __init__(self):
        # Weights and Biases
        self.Wh1 = weight_variable([9, 9, 3, 63], "f_Wh1")
        self.bh1 = bias_variable([63], "f_bh1")

        self.Wh2 = weight_variable([5, 5, 64, 64], "f_Wh2")
        self.bh2 = bias_variable([64], "f_bh2")

        self.Wh3 = weight_variable([5, 5, 64, 1], "f_Wh3")
        self.bh3 = bias_variable([1], "f_bh3")

        # Layers
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None

        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

        self.pool1 = None

        self.conc = None


class MonoDeepModel(object):
    def __init__(self, mode, params):
        print(params)

        self.params = params
        self.mode = mode

        model_index = 0
        self.model_collection = ['model_' + str(model_index)]

        # Parses Values
        _, self.image_height, self.image_width, self.image_nchannels = params['inputSize']  # Input
        _, self.depth_height, self.depth_width = params['outputSize']  # Output
        self.fc_hiddenNeurons = self.depth_height * self.depth_width

        self.build_model()

        if self.mode == 'test':
            return

        # self.build_losses()
        self.build_summaries()

        self.countParams()

    def createLayers_monodeep(self):
        self.coarse = Coarse(self.image_height, self.image_width, self.depth_height, self.depth_width,
                             self.fc_hiddenNeurons)
        self.fine = Fine()

    def build_model(self):
        # ======================
        #  Tensorflow Variables
        # ======================
        print("\n[Network/Model] Build Network Model...")

        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch (SGD).
        with tf.name_scope('Inputs'):
            self.tf_image = tf.placeholder(tf.float32,
                                           shape=(None, self.image_height, self.image_width, self.image_nchannels),
                                           name='image')
            self.tf_labels = tf.placeholder(tf.float32, shape=(None, self.depth_height, self.depth_width),
                                            name='labels')
            self.tf_log_labels = tf.log(self.tf_labels + LOSS_LOG_INITIAL_VALUE, name='log_labels')

            self.tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.tf_global_step = tf.Variable(0, trainable=False,
                                              name='global_step')  # Count the number of steps taken.
            self.tf_bn_train = tf.placeholder(tf.bool, name='bn_train')  # Boolean value to guide batchnorm
            self.tf_learningRate = self.params['learning_rate']
            if self.params['ldecay']:
                self.tf_learningRate = tf.train.exponential_decay(self.tf_learningRate, self.tf_global_step, 1000, 0.95,
                                                                  staircase=True, name='ldecay')

            tf.add_to_collection('image', self.tf_image)
            tf.add_to_collection('labels', self.tf_labels)
            tf.add_to_collection('keep_prob', self.tf_keep_prob)
            tf.add_to_collection('global_step', self.tf_global_step)
            tf.add_to_collection('bn_train', self.tf_bn_train)
            tf.add_to_collection('learning_rate', self.tf_learningRate)

        # ==============
        #  Select Model
        # ==============
        try:
            if self.params['model_name'] == 'monodeep':
                self.createLayers_monodeep()
            else:
                raise ValueError
        except ValueError:
            print("[ValueError] Check value of 'model_name' specified.")
            raise SystemExit

        # ----- Network Output (Predictions) ----- #
        with tf.name_scope("Outputs"):
            self.tf_predCoarse = self.build_modelCoarse(self.tf_image)
            self.tf_predFine = self.build_modelFine(self.tf_image, self.tf_predCoarse)

            tf.add_to_collection('predCoarse', self.tf_predCoarse)
            tf.add_to_collection('predFine', self.tf_predFine)

        # ----- Network Losses ----- #
        with tf.name_scope("Losses"):
            # self.tf_lossF = self.tf_MSE(self.tf_predFine, self.tf_log_labels)
            self.tf_lossF = self.tf_L(self.tf_predFine, self.tf_log_labels)

        # ----- Network Optimizer ----- #
        with tf.name_scope("Optimizer"):
            # optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.tf_loss,
            #                                                global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(self.tf_learningRate)
            self.train = optimizer.minimize(self.tf_lossF, global_step=self.tf_global_step)
            tf.add_to_collection("train_step", self.train)

        # Debug
        # print(self.tf_image)
        # print(self.tf_predCoarse)
        # print(self.tf_predFine)
        # print(self.tf_labels)
        # print(self.tf_log_labels)

    def build_modelCoarse(self, image):
        # Network Layers
        self.coarse.conv1 = tf.nn.conv2d(image, filter=self.coarse.Wh1, strides=[1, 4, 4, 1], padding='SAME')
        self.coarse.hidden1 = tf.nn.relu(self.coarse.conv1 + self.coarse.bh1)
        self.coarse.pool1 = tf.nn.max_pool(self.coarse.hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                           padding='SAME')

        self.coarse.conv2 = tf.nn.conv2d(self.coarse.pool1, filter=self.coarse.Wh2, strides=[1, 1, 1, 1],
                                         padding='SAME')
        self.coarse.hidden2 = tf.nn.relu(self.coarse.conv2 + self.coarse.bh2)
        self.coarse.pool2 = tf.nn.max_pool(self.coarse.hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                           padding='SAME')

        self.coarse.conv3 = tf.nn.conv2d(self.coarse.pool2, filter=self.coarse.Wh3, strides=[1, 1, 1, 1],
                                         padding='SAME')
        self.coarse.hidden3 = tf.nn.relu(self.coarse.conv3 + self.coarse.bh3)

        self.coarse.conv4 = tf.nn.conv2d(self.coarse.hidden3, filter=self.coarse.Wh4, strides=[1, 1, 1, 1],
                                         padding='SAME')
        self.coarse.hidden4 = tf.nn.relu(self.coarse.conv4 + self.coarse.bh4)

        self.coarse.conv5 = tf.nn.conv2d(self.coarse.hidden4, filter=self.coarse.Wh5, strides=[1, 2, 2, 1],
                                         padding='SAME')
        self.coarse.hidden5 = tf.nn.relu(self.coarse.conv5 + self.coarse.bh5)
        shape_h5 = self.coarse.hidden5.get_shape().as_list()

        self.coarse.fc1 = tf.reshape(self.coarse.hidden5, [-1, shape_h5[1] * shape_h5[2] * shape_h5[3]])

        self.coarse.hidden6 = tf.nn.relu(tf.matmul(self.coarse.fc1, self.coarse.Wh6) + self.coarse.bh6)

        # TODO: Remover três linhas abaixo
        self.hidden7_drop = tf.nn.dropout(self.coarse.hidden6, self.tf_keep_prob)
        self.hidden7_matmul = tf.matmul(tf.nn.dropout(self.coarse.hidden6, self.tf_keep_prob), self.coarse.Wh7)
        self.hidden7_bias = tf.matmul(tf.nn.dropout(self.coarse.hidden6, self.tf_keep_prob),
                                      self.coarse.Wh7) + self.coarse.bh7

        self.coarse.hidden7 = tf.matmul(tf.nn.dropout(self.coarse.hidden6, self.tf_keep_prob),
                                        self.coarse.Wh7) + self.coarse.bh7  # Linear

        self.coarse.fc2 = tf.reshape(self.coarse.hidden7, [-1, self.depth_height, self.depth_width])

        return self.coarse.fc2

    def build_modelFine(self, image, predCoarse):
        image_shape = image.get_shape().as_list()
        predCoarse_shape = predCoarse.get_shape().as_list()

        self.fine.conv1 = tf.nn.conv2d(image, filter=self.fine.Wh1, strides=[1, 2, 2, 1], padding='SAME')
        self.fine.hidden1 = tf.nn.relu(self.fine.conv1 + self.fine.bh1)
        self.fine.pool1 = tf.nn.max_pool(self.fine.hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        int_coarse_dim = tf.expand_dims(predCoarse, 3)
        self.fine.conc = tf.concat([self.fine.pool1, int_coarse_dim], axis=3)

        self.fine.conv2 = tf.nn.conv2d(self.fine.conc, filter=self.fine.Wh2, strides=[1, 1, 1, 1], padding='SAME')
        self.fine.hidden2 = tf.nn.relu(self.fine.conv2 + self.fine.bh2)

        self.fine.conv3 = tf.nn.conv2d(self.fine.hidden2, filter=self.fine.Wh3, strides=[1, 1, 1, 1], padding='SAME')
        # hidden3 = tf.nn.relu(conv3 + fine.bh3) # ReLU
        self.fine.hidden3 = self.fine.conv3 + self.fine.bh3  # Linear

        return self.fine.hidden3[:, :, :, 0]

    # TODO: Mudar
    @staticmethod
    def tf_MSE(y, y_):
        print("[Network/Model] Loss Function: MSE")
        # Check if y and y* have the same dimensions
        assert ((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

        # Variables
        batchSize, height, width = y_.get_shape().as_list()
        numPixels = height * width
        tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)

        return tf.reduce_sum(tf.pow(y_ - y, 2)) / tf_npixels

    @staticmethod
    def tf_L(tf_log_y, tf_log_y_, gamma=0.5):
        print("[Network/Model] Loss Function: Eigen's Log Depth")
        # Local Variables
        batchSize, height, width = tf_log_y_.get_shape().as_list()
        numPixels = height * width

        # Tensorflow Variables
        # tf_npixels = tf.cast(tf.constant(batchSize*numPixels), tf.float32) # TODO: Posso retirar o tamanho do batch da conta? Lembrando que os tensores foram definidos sem especificar o tamanho do batch, logo nao tenho essa informacao aki.
        tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)
        tf_d = tf_log_y - tf_log_y_

        tf_gx_d = gradient_x(tf_d)
        tf_gy_d = gradient_y(tf_d)

        mean_term = (tf.reduce_sum(tf.pow(tf_d, 2)) / tf_npixels)
        variance_term = ((gamma / tf.pow(tf_npixels, 2)) * tf.pow(tf.reduce_sum(tf_d), 2))
        grads_term = (tf.reduce_sum(tf.pow(tf_gx_d, 2)) + tf.reduce_sum(tf.pow(tf_gy_d, 2))) / tf_npixels

        # FIXME: variance_term should be negative
        tf_loss_d = mean_term + variance_term + grads_term

        return tf_loss_d

    # TODO: Utilizar a tf_mse do stereoCNN e adicionar L2Norm
    def build_losses(self):
        with tf.name_scope("Losses"):
            # self.tf_lossF = self.tf_MSE(self.tf_predFine, self.tf_log_labels)
            self.tf_lossF = self.tf_L(self.tf_predFine, self.tf_log_labels)

    # TODO: Criar summaries das variaveis internas do modelo
    def build_summaries(self):
        # Filling Summary Obj
        with tf.name_scope("Summaries"):
            tf.summary.scalar('learning_rate', self.tf_learningRate, collections=self.model_collection)
            tf.summary.scalar('lossF', self.tf_lossF, collections=self.model_collection)
            tf.summary.scalar('keep_prob', self.tf_keep_prob, collections=self.model_collection)

    @staticmethod
    def countParams():
        # Count Params
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("[Network/Model] Number of trainable parameters: {}".format(total_num_parameters))
