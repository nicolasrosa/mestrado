# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

# =====================
#  Class Configuration
# =====================
TRAINING_BATCH_SIZE = 16
TRAINING_LEARNINGDECAY_STEPS = 1000
TRAINING_LEARNINGDECAY_RATE = 0.95
TRAINING_L2NORM_BETA = 1000


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
            y_masked[idx_i[k], idx_j[k]] = 0.0  # Predictions with labels equal to zero are set to zero.

        return y_masked

    @staticmethod
    def tf_maskOutInvalidPixels(y, y_):
        # Variables
        tf_y = tf.reshape(y, [y.get_shape().as_list()[0], -1])  # Tensor 'y'  (batchSize, height*width)
        tf_y_ = tf.reshape(y_, [y_.get_shape().as_list()[0], -1])  # Tensor 'y_' (batchSize, height*width)
        tf_c_y_ = tf_y_ > 0  # Tensor of Conditions (bool)
        tf_idx = tf.where(tf_c_y_)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

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

        return np.power(y_ - y, 2) / numPixels  # MSE calculated for each pixel

    @staticmethod
    def tf_MSE_old(y, y_, onlyValidPixels=False):
        # Check if y and y* have the same dimensions
        assert ((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

        # Variables
        batchSize, height, width = y_.get_shape().as_list()
        numPixels = height * width
        tf_npixels = None

        if onlyValidPixels:
            # Mask out invalid values (values <= 0)!
            y, y_, tf_npixels_valid = Loss.tf_maskOutInvalidPixels(y, y_)
            tf_npixels = tf_npixels_valid
        else:
            # tf_npixels = tf.cast(tf.constant(batchSize*numPixels), tf.float32) # TODO: Posso retirar o tamanho do batch da conta? Lembrando que os tensores foram definidos sem especificar o tamanho do batch, logo nao tenho essa informacao aki.
            tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)

        return tf.reduce_sum(tf.pow(y_ - y, 2)) / tf_npixels

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
    @staticmethod
    def tf_L_old(y, y_, gamma=0.5, onlyValidPixels=False):
        # Check if y and y* have the same dimensions
        assert ((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

        # Variables
        batchSize, height, width = y_.get_shape().as_list()
        numPixels = height * width
        tf_npixels = None

        if onlyValidPixels:
            # Mask out invalid values (values <= 0)!
            y, y_, tf_npixels = Loss.tf_maskOutInvalidPixels(y, y_)
            tf_d = tf.log(y) - tf.log(y_)

        else:
            # tf_npixels = tf.cast(tf.constant(batchSize*numPixels), tf.float32) # TODO: Posso retirar o tamanho do batch da conta? Lembrando que os tensores foram definidos sem especificar o tamanho do batch, logo nao tenho essa informacao aki.
            tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)
            tf_d = tf.log(y + LOSS_LOG_INITIAL_VALUE) - tf.log(y_ + LOSS_LOG_INITIAL_VALUE)

        return (tf.reduce_sum(tf.pow(tf_d, 2)) / tf_npixels) - (
                (gamma / tf.pow(tf_npixels, 2)) * tf.pow(tf.reduce_sum(tf_d), 2))

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

    # TODO: Implementar a loss function do segundo artigo do Fergus (log-domain), ela também apresenta os gradientes verticais e horizontais da diferença b(nabla_x(di) e nabla_y(di), di = y-y*)
    @staticmethod
    def np_L(y, y_):
        pass

    # ------------------ #
    #  L2 Normalization  #
    # ------------------ #
    # TODO: Mover
    @staticmethod
    def getGlobalVars(scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    # TODO: Mover
    @staticmethod
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

        return TRAINING_L2NORM_BETA * sum

    @staticmethod
    def calculateL2norm_Fine(model):
        fine_vars = Loss.getTrainableVars("f_")
        # print(fine_vars)

        sum = 0
        for i, val in enumerate(fine_vars):
            sum += tf.nn.l2_loss(val)
            # print (i, val)

        return TRAINING_L2NORM_BETA * sum
