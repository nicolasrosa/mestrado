#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  Libraries
# ===========
from classes.utils import Utils
import tensorflow as tf
import numpy as np

# =====================
#  Class Configuration
# =====================
TRAINING_BATCH_SIZE= 16
TRAINING_LEARNINGDECAY_STEPS = 1000
TRAINING_LEARNINGDECAY_RATE = 0.95
TRAINING_L2NORM_BETA=1000

LOSS_LOG_INITIAL_VALUE = 1E-6



# ===================
#  Class Declaration
# ===================
class Training(object):
    def __init__(self, learningRate=1e-4):
        print("Training Obj created!")

        # Variables Declaration
        self.batchSize = None
        self.learningRate = None
        self.ldecayRate = None
        self.ldecaySteps = None

        # Sets Variables Values
        self.setBatchSize(TRAINING_BATCH_SIZE)
        self.setldecayRate(TRAINING_LEARNINGDECAY_RATE)
        self.setldecaySteps(TRAINING_LEARNINGDECAY_STEPS)

        # Logs the calculated Network Losses for each Training step
        self.lossC_Hist = []
        self.lossF_Hist = []


    def setBatchSize(self, value):
        self.batchSize = value

    def getBatchSize(self):
        return self.batchSize


    def setLearningRate(self, value):
        self.learningRate = value

    def getLearningRate(self):
        return self.learningRate


    def setldecayRate(self, value):
        self.ldecayRate = value

    def getldecayRate(self):
        return self.ldecayRate


    def setldecaySteps(self, value):
        self.ldecaySteps = value

    def getldecaySteps(self):
        return self.ldecaySteps
    
# ===================
#  Class Declaration
# ===================
class Loss(object):
    def __init__(self):
        pass

    @staticmethod
    def np_maskOutInvalidPixels(y, y_):
        condition = y_ <= 0
        idx_i, idx_j = np.where(condition)

        y_masked = np.copy(y)
        for k in range(0, len(idx_i)):
            y_masked[idx_i[k], idx_j[k]] = 0.0      # Predictions with labels equal to zero are set to zero.
 
        return y_masked

    @staticmethod
    def tf_maskOutInvalidPixels(y, y_):
        # Variables
        tf_y = tf.reshape(y, [y.get_shape().as_list()[0], -1])     # Tensor 'y'  (batchSize, height*width)
        tf_y_ = tf.reshape(y_, [y_.get_shape().as_list()[0], -1])  # Tensor 'y_' (batchSize, height*width)
        tf_c_y_ = tf_y_ > 0                                     # Tensor of Conditions (bool)
        tf_idx = tf.where(tf_c_y_)                              # Tensor 'idx' of Valid Pixel values (batchID, idx)

        # TODO: Lembre-se que o tf.gather_nd() não foi otimizador para rodar em gpu, o mais indicado é utilizar tf.gather() (otimizado). O vitor faz o flatten na imagem e no batch, assim ele consegue um tensor de idx com dimensão 1.
        tf_valid_y = tf.gather_nd(tf_y, tf_idx)                  
        tf_valid_y_ = tf.gather_nd(tf_y_, tf_idx)
        tf_npixels_valid = tf.shape(tf_valid_y_)
        tf_npixels_valid_float32 = tf.cast(tf_npixels_valid, tf.float32)

        return tf_valid_y, tf_valid_y_, tf_npixels_valid_float32

    # -------------------- #
    #  Mean Squared Error  #
    # -------------------- #
    @staticmethod
    def np_MSE(y, y_):
        numPixels = y_.size

        return np.power(y_-y, 2)/numPixels # MSE calculated for each pixel

    @staticmethod
    def tf_MSE(y, y_, onlyValidPixels=False):
        # Check if y and y* have the same dimensions
        assert((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

        # Variables
        batchSize, height, width = y_.get_shape().as_list()
        numPixels = height*width
        tf_npixels = None

        if onlyValidPixels:
            # Mask out invalid values (values <= 0)!
            y, y_, tf_npixels_valid = Loss.tf_maskOutInvalidPixels(y, y_)
            tf_npixels = tf_npixels_valid
        else:
            # tf_npixels = tf.cast(tf.constant(batchSize*numPixels), tf.float32) # TODO: Posso retirar o tamanho do batch da conta? Lembrando que os tensores foram definidos sem especificar o tamanho do batch, logo nao tenho essa informacao aki.
            tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)

        return tf.reduce_sum(tf.pow(y_ - y, 2))/tf_npixels

    # ------- #
    #  BerHu  #
    # ------- # 
    # TODO: Implemente BerHu Loss function
    @staticmethod
    def np_BerHu():
        pass

    # TODO: Implemente BerHu Loss function
    @staticmethod
    def tf_BerHu():
        pass


    # ------------------------------ #
    #  Training Loss - Eigen,Fergus  #
    # ------------------------------ # 
    # TODO: Implementar a loss function do segundo artigo do Fergus (log-domain), ela também apresenta os gradientes verticais e horizontais da diferença b(nabla_x(di) e nabla_y(di), di = y-y*)
    @staticmethod
    def np_L(y, y_):
        pass
    
    @staticmethod
    def tf_L(y, y_, gamma=0.5, onlyValidPixels=False):
        # Check if y and y* have the same dimensions
        assert((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

        # Variables
        batchSize, height, width = y_.get_shape().as_list()
        numPixels = height*width
        tf_npixels = None

        if onlyValidPixels:
            # Mask out invalid values (values <= 0)!
            y, y_, tf_npixels = Loss.tf_maskOutInvalidPixels(y, y_)
            tf_d = tf.log(y) - tf.log(y_)
        
        else:
            # tf_npixels = tf.cast(tf.constant(batchSize*numPixels), tf.float32) # TODO: Posso retirar o tamanho do batch da conta? Lembrando que os tensores foram definidos sem especificar o tamanho do batch, logo nao tenho essa informacao aki.
            tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)
            tf_d = tf.log(y+LOSS_LOG_INITIAL_VALUE) - tf.log(y_+LOSS_LOG_INITIAL_VALUE)

        return (tf.reduce_sum(tf.pow(tf_d, 2))/tf_npixels)-((gamma/tf.pow(tf_npixels, 2))*tf.pow(tf.reduce_sum(tf_d), 2))
    
    # ------------------ #
    #  L2 Normalization  #
    # ------------------ #
    # TODO: Mover
    def getGlobalVars(scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    
    # TODO: Mover
    def getTrainableVars(scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    @staticmethod
    def calculateL2norm_Coarse(model):
        coarse_vars = Loss.getTrainableVars("c_")
        # print(coarse_vars)

        sum = 0
        for i, val in enumerate(coarse_vars):
            sum += tf.nn.l2_loss(val)
            # print (i, val)

        return TRAINING_L2NORM_BETA*sum


    @staticmethod
    def calculateL2norm_Fine(model):
        fine_vars = Loss.getTrainableVars("f_")
        # print(fine_vars)

        sum = 0
        for i, val in enumerate(fine_vars):
            sum += tf.nn.l2_loss(val)
            # print (i, val)

        return TRAINING_L2NORM_BETA*sum
