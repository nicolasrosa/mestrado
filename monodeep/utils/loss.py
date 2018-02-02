# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

# ==================
#  Global Variables
# ==================
TRAINING_BATCH_SIZE = 16
TRAINING_LEARNINGDECAY_STEPS = 1000
TRAINING_LEARNINGDECAY_RATE = 0.95
TRAINING_L2NORM_BETA = 1000
LOSS_LOG_INITIAL_VALUE = 1e-6


# ===========
#  Functions
# ===========
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


def np_maskOutInvalidPixels(y, y_):
    condition = y_ <= 0
    idx_i, idx_j = np.where(condition)

    y_masked = np.copy(y)
    for k in range(0, len(idx_i)):
        y_masked[idx_i[k], idx_j[k]] = 0.0  # Predictions with labels equal to zero are set to zero.

    return y_masked


# FIXME: Não funciona,  ResourceExhaustedError (Pegando muita Memória)
def tf_maskOutInvalidPixels(y, y_):
    # Variables
    tf_y = tf.contrib.layers.flatten(y)  # Tensor 'y'  (batchSize, height*width)
    tf_y_ = tf.contrib.layers.flatten(y_)  # Tensor 'y_' (batchSize, height*width)
    tf_c_y_ = tf_y_ > 0  # Tensor of Conditions (bool)
    tf_idx = tf.where(tf_c_y_)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

    tf_valid_y = tf.gather(tf_y, tf_idx)
    tf_valid_y_ = tf.gather(tf_y_, tf_idx)
    tf_npixels_valid = tf.shape(tf_valid_y_)
    tf_npixels_valid_float32 = tf.cast(tf_npixels_valid, tf.float32)

    # print(tf_y)
    # print(tf_y_)
    # print(tf_c_y_)
    # print(tf_idx)
    print(tf_valid_y)
    print(tf_valid_y_)
    print(tf_npixels_valid)
    print(tf_npixels_valid_float32)
    input("mask")

    return tf_valid_y, tf_valid_y_, tf_npixels_valid_float32


# -------------------- #
#  Mean Squared Error  #
# -------------------- #
def np_MSE(y, y_):
    numPixels = y_.size

    return np.power(y_ - y, 2) / numPixels  # MSE calculated for each pixel


def tf_MSE(y, y_, onlyValidPixels=False):
    print("[Network/Model] Loss Function: MSE")

    # Check if y and y* have the same dimensions
    # assert ((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

    if onlyValidPixels:
        # Mask out invalid values (values <= 0)!
        y, y_, tf_npixels_valid = tf_maskOutInvalidPixels(y, y_)
        tf_npixels = tf_npixels_valid
    else:
        tf_npixels = tf.cast(tf.size(y_), tf.float32)  # (batchSize*height*width)

    return tf.reduce_sum(tf.pow(y_ - y, 2)) / tf_npixels


# ------- #
#  BerHu  #
# ------- #
# TODO: Implemente BerHu Loss function
def np_BerHu():
    pass


# TODO: Implemente BerHu Loss function
def tf_BerHu():
    pass


# ------------------------------ #
#  Training Loss - Eigen,Fergus  #
# ------------------------------ #
def tf_L(log_y, log_y_, gamma=0.5, onlyValidPixels=False):
    print("[Network/Model] Loss Function: Eigen's Log Depth")

    # Check if y and y* have the same dimensions
    # assert ((log_y.shape[1] == log_y_.shape[1]) and (
    #         log_y.shape[2] == log_y_.shape[2])), "Houston we've got a problem"

    # Tensorflow Variables
    if onlyValidPixels:
        # Mask out invalid values (values <= 0)!
        log_y, log_y_, tf_npixels = tf_maskOutInvalidPixels(log_y, log_y_)
        tf_d = log_y - log_y_
    else:
        tf_npixels = tf.cast(tf.size(log_y_), tf.float32)  # (batchSize*height*width)
        tf_d = log_y - log_y_

    tf_gx_d = gradient_x(tf_d)
    tf_gy_d = gradient_y(tf_d)

    mean_term = (tf.reduce_sum(tf.pow(tf_d, 2)) / tf_npixels)
    variance_term = ((gamma / tf.pow(tf_npixels, 2)) * tf.pow(tf.reduce_sum(tf_d), 2))
    grads_term = (tf.reduce_sum(tf.pow(tf_gx_d, 2)) + tf.reduce_sum(tf.pow(tf_gy_d, 2))) / tf_npixels

    # FIXME: variance_term should be negative
    # tf_loss_d = mean_term - variance_term + grads_term
    tf_loss_d = mean_term + variance_term + grads_term  # Workaround

    return tf_loss_d


# TODO: Implementar a loss function do segundo artigo do Fergus (log-domain), ela também apresenta os gradientes verticais e horizontais da diferença b(nabla_x(di) e nabla_y(di), di = y-y*)
def np_L(y, y_):
    pass


# ------------------ #
#  L2 Normalization  #
# ------------------ #
# TODO: Mover
def getGlobalVars(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


# TODO: Mover
def getTrainableVars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def calculateL2norm_Coarse(model):
    coarse_vars = getTrainableVars("c_")
    # print(coarse_vars)

    totalSum = 0
    for i, val in enumerate(coarse_vars):
        totalSum += tf.nn.l2_loss(val)
        # print (i, val)

    return TRAINING_L2NORM_BETA * totalSum


def calculateL2norm_Fine(model):
    fine_vars = getTrainableVars("f_")
    # print(fine_vars)

    totalSum = 0
    for i, val in enumerate(fine_vars):
        totalSum += tf.nn.l2_loss(val)
        # print (i, val)

    return TRAINING_L2NORM_BETA * totalSum
