#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  Libraries
# ===========
import tensorflow as tf
from classes.training import Training
from classes.validation import Validation
from classes.testing import Testing
from classes.restore import Restore

# ===================
#  Class Declaration
# ===================
class NetworkModel(object):
    def __init__(self, inputSize, outputSize, isRestore=False):
        # Variables initialization according to the chosen dataset
        _, self.imageInput_height, self.imageInput_width, self.imageInput_nchannels = inputSize
        _, self.depthOutput_height, self.depthOutput_width = outputSize
        self.fc_hiddenNeurons = self.depthOutput_height*self.depthOutput_width

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

        Wh5_outputSize_height = round(self.imageInput_height/32)+1
        Wh5_outputSize_width = round(self.imageInput_width/32)

        self.c_Wh6 = self.weight_variable([Wh5_outputSize_height*Wh5_outputSize_width*256, self.fc_hiddenNeurons], "c_Wh6")
        self.c_bh6 = self.bias_variable([self.fc_hiddenNeurons], "c_bh6")

        self.depthOutput_numPixels = self.depthOutput_height * self.depthOutput_width        
        assert (self.fc_hiddenNeurons == self.depthOutput_numPixels), "The number of Neurons must be iqual to the number of output pixels."
        
        self.c_Wh7 = self.weight_variable([self.fc_hiddenNeurons, self.depthOutput_numPixels], "c_Wh7")
        self.c_bh7 = self.bias_variable([self.depthOutput_numPixels], "c_bh7")

        # Weights and Biases - Fine
        self.f_Wh1 = self.weight_variable([9, 9, 3, 63], "f_Wh1")
        self.f_bh1 = self.bias_variable([63], "f_bh1")

        self.f_Wh2 = self.weight_variable([5, 5, 64, 64], "f_Wh2")
        self.f_bh2 = self.bias_variable([64], "f_bh2")

        self.f_Wh3 = self.weight_variable([5, 5, 64, 1], "f_Wh3")
        self.f_bh3 = self.bias_variable([1], "f_bh3")

        # Layers Declaration
        # TODO: Iniciar layers com None

        # Training, Validation, Testing and Restore Obj Variables
        if isRestore:
            self.rest = Restore()
            self.train = Training() # TODO: Restore and Continue Training
        else:
            self.train = Training()
            self.valid = Validation()
            self.test = Testing()


    def weight_variable(self, shape, variableName):
        initial = tf.truncated_normal(shape, mean=0.00005, stddev=0.0001) # Try to avoid generate negative values
        # initial = tf.truncated_normal(shape, stddev=10.0)    
        return tf.Variable(initial, name=variableName)


    def bias_variable(self, shape, variableName):
        initial = tf.constant(0.1, shape=shape)
        # initial = tf.constant(10.0, shape=shape)
        return tf.Variable(initial, name=variableName)


    def model_coarse(self,data, keep_prob, enableDropout):
        data_shape = data.get_shape().as_list()

        # Network Layers
        conv1 = tf.nn.conv2d(data, filter=self.c_Wh1, strides=[1, 4, 4, 1], padding='SAME')
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
        fc1_drop = tf.nn.dropout(hidden6, keep_prob)

        hidden7 = tf.matmul(fc1_drop if enableDropout else hidden6, self.c_Wh7) + self.c_bh7 # Linear
        fc2 = tf.reshape(hidden7, [-1, self.depthOutput_height, self.depthOutput_width])

        # Debug
        print("\nNetwork Layers:")
        print("i:", data_shape) 
        print()

        print("1:", conv1.get_shape().as_list())
        print(hidden1.get_shape().as_list())
        print(pool1.get_shape().as_list())
        print()

        print("2:", conv2.get_shape().as_list())      
        print(hidden2.get_shape().as_list())    
        print(pool2.get_shape().as_list())      
        print()

        print("3:", conv3.get_shape().as_list())      
        print(hidden3.get_shape().as_list())
        print()

        print("4:", conv4.get_shape().as_list())      
        print(hidden4.get_shape().as_list())    
        print()

        print("5:", conv5.get_shape().as_list())
        print(hidden5.get_shape().as_list())
        print()

        print("6:", fc1.get_shape().as_list())
        print(hidden6.get_shape().as_list())
        print(fc1_drop.get_shape().as_list())
        print()

        print("7:", hidden7.get_shape().as_list())     
        print(fc2.get_shape().as_list())

        return fc2

    def model_fine(self, input_image, input_coarse):
        input_image_shape = input_image.get_shape().as_list()
        input_coarse_shape = input_coarse.get_shape().as_list()

        conv1 = tf.nn.conv2d(input_image, filter=self.f_Wh1, strides=[1, 2, 2, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + self.f_bh1)
        pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        int_coarse_dim = tf.expand_dims(input_coarse, 3)
        conc = tf.concat([pool1, int_coarse_dim], axis=3)

        conv2 = tf.nn.conv2d(conc, filter=self.f_Wh2, strides=[1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + self.f_bh2)

        conv3 = tf.nn.conv2d(hidden2, filter=self.f_Wh3, strides=[1, 1, 1, 1], padding='SAME')
        # hidden3 = tf.nn.relu(conv3 + f_bh3) # ReLU
        hidden3 = conv3 + self.f_bh3 # Linear

        # Debug
        print("\nNetwork Layers:")
        print("i:", input_image_shape)
        print("c_o:", input_coarse_shape)
        print()

        print("1:", conv1.get_shape().as_list())
        print(hidden1.get_shape().as_list())
        print(pool1.get_shape().as_list())
        print(conc.get_shape().as_list())
        print()

        print("2:", conv2.get_shape().as_list())
        print(hidden2.get_shape().as_list())
        print()

        print("3:", conv3.get_shape().as_list())
        print(hidden3.get_shape().as_list())
        print()

        return hidden3[:, :, :, 0]
