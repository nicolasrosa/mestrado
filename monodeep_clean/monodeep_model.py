#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  Libraries
# ===========
import tensorflow as tf
from collections import namedtuple

# ==================
#  Global Variables
# ==================
monodeep_parameters = namedtuple('parameters', 
                        'height, width, '
                        'batch_size, '
                        # 'num_epochs, '
                        'maxSteps, '                        
                        'dropout, '
                        'full_summary')

# ===================
#  Class Declaration
# ===================
class MonoDeepModel(object):
    def __init__(self,params, mode, inputSize, outputSize):
        self.params = params
        self.mode = mode
        self.tf_keep_prob = params.dropout

        # Variables initialization according to the chosen dataset
        _, self.image_height, self.image_width, self.image_nchannels = inputSize # Input
        _, self.depth_height, self.depth_width = outputSize                      # Output

        self.fc_hiddenNeurons = self.depth_height*self.depth_width

        # print(params)
        # print(self.image_height, self.image_width,self.image_nchannels)
        # print(self.depth_height, self.depth_width)

        self.build_model()
        # self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()  

    def build_model(self):
        print("\n[Network] Build Network Model...")

        self.createLayers()
        
        with tf.name_scope("Inputs"):
            # TODO: Mudar nomes para tf_image e tf_depth/tf_disp
            self.tf_image = tf.placeholder(tf.float32, shape=(None, self.image_height, self.image_width, self.image_nchannels), name="tf_image")
            self.tf_labels = tf.placeholder(tf.float32, shape=(None, self.depth_height, self.depth_width), name="tf_labels")
            self.tf_keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        
        with tf.name_scope("Outputs"):
            self.tf_predCoarse = self.build_modelCoarse(self.tf_image)
            self.tf_predFine = self.build_modelFine(self.tf_image, self.tf_predCoarse)

        # Debug
        # print(self.tf_image)
        # print(self.tf_predCoarse)
        # print(self.tf_predFine)
        # print(self.tf_labels)
       

    def weight_variable(self, shape, variableName):
        initial = tf.truncated_normal(shape, mean=0.00005, stddev=0.0001) # Try to avoid generate negative values
        # initial = tf.truncated_normal(shape, stddev=10.0)    
        return tf.Variable(initial, name=variableName)


    def bias_variable(self, shape, variableName):
        initial = tf.constant(0.1, shape=shape)
        # initial = tf.constant(10.0, shape=shape)
        return tf.Variable(initial, name=variableName)

    def createLayers(self):
        print("[Network] Creating Layers...")
        
        # Weights and Biases - Coarse
        self.c_Wh1 = self.weight_variable([11, 11, 3, 96], "c_Wh1")
        self.c_bh1 = self.bias_variable([96], "c_bh1")

        self.c_Wh2 = self.weight_variable([5, 5, 96, 256], "c_Wh2")
        self.c_bh2 = self.bias_variable([256], "c_bh2")

        self.c_Wh3 = self.weight_variable([3, 3, 256, 384], "c_Wh3")
        self.c_bh3 = self.bias_variable([384], "c_bh3")

        self.c_Wh4 = self.weight_variable([3, 3, 384, 384], "c_Wh4")
        self.c_bh4 = self.bias_variable([384], "c_bh4")

        self.c_Wh5 = self.weight_variable([3, 3, 384, 256], "c_Wh5")
        self.c_bh5 = self.bias_variable([256], "c_bh5")

        Wh5_outputSize_height = round(self.image_height/32)+1
        Wh5_outputSize_width = round(self.image_width/32)

        self.c_Wh6 = self.weight_variable([Wh5_outputSize_height*Wh5_outputSize_width*256, self.fc_hiddenNeurons], "c_Wh6")
        self.c_bh6 = self.bias_variable([self.fc_hiddenNeurons], "c_bh6")

        self.depth_numPixels = self.depth_height * self.depth_width        
        assert (self.fc_hiddenNeurons == self.depth_numPixels), "The number of Neurons must be iqual to the number of output pixels."

        self.c_Wh7 = self.weight_variable([self.fc_hiddenNeurons, self.depth_numPixels], "c_Wh7")
        self.c_bh7 = self.bias_variable([self.depth_numPixels], "c_bh7")

        # Weights and Biases - Fine
        self.f_Wh1 = self.weight_variable([9, 9, 3, 63], "f_Wh1")
        self.f_bh1 = self.bias_variable([63], "f_bh1")

        self.f_Wh2 = self.weight_variable([5, 5, 64, 64], "f_Wh2")
        self.f_bh2 = self.bias_variable([64], "f_bh2")

        self.f_Wh3 = self.weight_variable([5, 5, 64, 1], "f_Wh3")
        self.f_bh3 = self.bias_variable([1], "f_bh3")

    def build_modelCoarse(self, image):
            data_shape = image.get_shape().as_list()

            # Network Layers
            conv1 = tf.nn.conv2d(image, filter=self.c_Wh1, strides=[1, 4, 4, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + self.c_bh1)
            pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv2 = tf.nn.conv2d(pool1, filter=self.c_Wh2, strides=[1, 1, 1, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + self.c_bh2)
            pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3 = tf.nn.conv2d(pool2, filter=self.c_Wh3, strides=[1, 1, 1, 1], padding='SAME')
            hidden3 = tf.nn.relu(conv3 + self.c_bh3)

            conv4 = tf.nn.conv2d(hidden3, filter=self.c_Wh4, strides=[1, 1, 1, 1], padding='SAME')
            hidden4 = tf.nn.relu(conv4 + self.c_bh4)

            conv5 = tf.nn.conv2d(hidden4, filter=self.c_Wh5, strides=[1, 2, 2, 1], padding='SAME')
            hidden5 = tf.nn.relu(conv5 + self.c_bh5)
            shape_h5 = hidden5.get_shape().as_list()

            fc1 = tf.reshape(hidden5, [-1, shape_h5[1] * shape_h5[2] * shape_h5[3]])

            hidden6 = tf.nn.relu(tf.matmul(fc1, self.c_Wh6) + self.c_bh6)
            hidden7 = tf.matmul(tf.nn.dropout(hidden6, self.tf_keep_prob), self.c_Wh7) + self.c_bh7 # Linear

            fc2 = tf.reshape(hidden7, [-1, self.depth_height, self.depth_width])

            return fc2

    def build_modelFine(self, image, predCoarse):
        image_shape = image.get_shape().as_list()
        predCoarse_shape = predCoarse.get_shape().as_list()

        conv1 = tf.nn.conv2d(image, filter=self.f_Wh1, strides=[1, 2, 2, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + self.f_bh1)
        pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        int_coarse_dim = tf.expand_dims(predCoarse, 3)
        conc = tf.concat([pool1, int_coarse_dim], axis=3)

        conv2 = tf.nn.conv2d(conc, filter=self.f_Wh2, strides=[1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + self.f_bh2)

        conv3 = tf.nn.conv2d(hidden2, filter=self.f_Wh3, strides=[1, 1, 1, 1], padding='SAME')
        # hidden3 = tf.nn.relu(conv3 + f_bh3) # ReLU
        hidden3 = conv3 + self.f_bh3 # Linear

        return hidden3[:, :, :, 0]

    # TODO: Terminar
    def build_outputs(self):
        print("terminar")

    # TODO: Mudar
    def tf_MSE(self, y, y_):
        # Check if y and y* have the same dimensions
        assert((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

        # Variables
        batchSize, height, width = y_.get_shape().as_list()
        numPixels = height*width
        tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)

        return tf.reduce_sum(tf.pow(y_ - y, 2))/tf_npixels

    # TODO: Utilizar a tf_mse do stereoCNN e adicionar L2Norm
    def build_losses(self):
        with tf.name_scope("Losses"):
            self.tf_lossC = self.tf_MSE(self.tf_predCoarse, self.tf_labels)
            self.tf_lossF = self.tf_MSE(self.tf_predFine, self.tf_labels)
            

    # TODO: Terminar
    def build_summaries(self):
        print("terminar")



   

  
