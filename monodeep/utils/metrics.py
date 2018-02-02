# Metrics presented by David Eigen, Christian Puhrsch and Rob Fergus in the article "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf


# ===========
#  Functions
# ===========
def evaluateTesting():
    # Testing Metrics
    # TODO: Criar Gráficos mostrando a evolução as métricas abaixo
    # TODO: Lembro que no Artigo do Eigen, o resultado final era uma média de todos os dados de treinamento(ou validação, não sei ao certo). Os valores abaixo são apenas para uma figura
    # Utils.displayVariableInfo(trPredictions_f[0,:,:])
    # Utils.displayVariableInfo(batch_labels[0,:,:])
    # print(metricsObj.Threshold(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # print(metricsObj.AbsRelativeDifference(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # print(metricsObj.SquaredRelativeDifference(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # print(metricsObj.RMSE_linear(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # print(metricsObj.RMSE_log(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # print(metricsObj.RMSE_log_scaleInv(trPredictions_f[0,:,:],batch_labels[0,:,:]))

    # metricsObj.Threshold_hist.append(metricsObj.np_Threshold(trPredictions_f[0,:,:], batch_labels[0,:,:]))
    # metricsObj.AbsRelativeDifference_hist.append(metricsObj.np_AbsRelativeDifference(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # metricsObj.SquaredRelativeDifference_hist.append(metricsObj.np_SquaredRelativeDifference(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # metricsObj.RMSE_linear_hist.append(metricsObj.np_RMSE_linear(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # metricsObj.RMSE_log_hist.append(metricsObj.np_RMSE_log(trPredictions_f[0,:,:],batch_labels[0,:,:]))
    # metricsObj.RMSE_log_scaleInv_hist.append(metricsObj.np_RMSE_log_scaleInv(trPredictions_f[0,:,:],batch_labels[0,:,:]))

    input("metrics")


def np_maskOutInvalidPixels(y, y_):
    # Index Vectors for Valid Pixels
    nvalids_idx_i = np.where(y_ > 0)[0]
    nvalids_idx_j = np.where(y_ > 0)[1]

    # Check if masked y and y* have the same number of pixels
    assert (len(nvalids_idx_i) == len(nvalids_idx_j)), "Houston we've got a problem"

    # Masking Out Invalid Pixels!
    y = y[nvalids_idx_i, nvalids_idx_j]
    y_ = y_[nvalids_idx_i, nvalids_idx_j]

    npixels_valid = len(nvalids_idx_i)

    return y, y_, nvalids_idx_i, nvalids_idx_j, npixels_valid


# ----------- #
#  Threshold  #
# ----------- #
# TODO: Threshold: % of yi s. t. max(yi/yi*, yi*/yi) = delta/thr
def np_Threshold(y, y_):
    value = -1
    return value


# TODO:
def tf_Threshold(y, y_):
    value = -1
    return value


# ------------------------- #
#  Abs Relative Difference  #
# ------------------------- #
def np_AbsRelativeDifference(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # TODO: T é realmente o número de pixels?
    npixels_total = y.shape[0] * y.shape[1]  # height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx_i, nvalids_idx_j, npixels_valid = Metrics.np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    # value = sum(abs(y-y_)/y_)/abs(npixels_total)
    value = sum(abs(y - y_) / y_) / abs(npixels_valid)

    # Debug
    # print("y:", y)
    # input()
    # print("y_:", y_)
    # input()

    # print("nvalids_idx_i:",nvalids_idx_i)
    # print("nvalids_idx_j:",nvalids_idx_j)
    # print(npixels_valid,"/",npixels_total,sep='')
    # input()

    # print("Mask out invalid pixels!")
    # print("y:",y)
    # print(y.shape,y.dtype)
    # input()
    # print("y_:",y_)
    # print(y_.shape,y.dtype)
    # input()

    # print("y-y_:", y-y_)
    # input()
    # print("abs(y-y_):", abs(y-y_))
    # input()
    # print("abs(y-y_)/y_:", abs(y-y_)/y_)
    # input()
    # print("sum(abs(y-y_)/y_):", sum(abs(y-y_)/y_))
    # input()
    # print("sum(abs(y-y_)/y_)/abs(T):", sum(abs(y-y_)/y_)/abs(npixels_valid))
    # input()
    # print(value)
    # input()

    return value


# TODO:
def tf_AbsRelativeDifference(y, y_):
    pass


# ----------------------------- #
#  Squared Relative Difference  #
# ----------------------------- #
def np_SquaredRelativeDifference(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # TODO: T é realmente o número de pixels?
    npixels_total = y.shape[0] * y.shape[1]  # height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx_i, nvalids_idx_j, npixels_valid = Metrics.np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    # value = sum(pow((abs(y-y_)/y_),2)/abs(npixels_total))
    value = sum(pow((abs(y - y_) / y_), 2) / abs(npixels_valid))

    # Debug
    # print(value)
    # input()

    return value


# TODO:
def tf_SquaredRelativeDifference(y, y_):
    pass


# -------------- #
#  RMSE(linear)  #
# -------------- #
def np_RMSE_linear(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # TODO: T é realmente o número de pixels?
    npixels_total = y.shape[0] * y.shape[1]  # height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx_i, nvalids_idx_j, npixels_valid = Metrics.np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    # value = np.sqrt(sum(pow(abs(y-y_),2))/abs(npixels_total))
    value = np.sqrt(sum(pow(abs(y - y_), 2)) / abs(npixels_valid))

    # Debug
    # print(value)
    # input()

    return value


# TODO:
def tf_RMSE_linear(y, y_):
    pass


# ----------- #
#  RMSE(log)  #
# ----------- #
def np_RMSE_log(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # TODO: T é realmente o número de pixels?
    npixels_total = y.shape[0] * y.shape[1]  # height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx_i, nvalids_idx_j, npixels_valid = Metrics.np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    # value = np.sqrt(sum(pow(abs(np.log(y)-np.log(y_)),2))/abs(npixels_total))
    value = np.sqrt(sum(pow(abs(np.log(y) - np.log(y_)), 2)) / abs(npixels_valid))

    # Debug
    # print(value)
    # input()

    return value


# TODO:
def tf_RMSE_log(y, y_):
    pass


# ---------------------------- #
#  RMSE(log, scale-invariant)  #
# ---------------------------- #
def np_RMSE_log_scaleInv(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # TODO: T é realmente o número de pixels?
    npixels_total = y.shape[0] * y.shape[1]  # height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx_i, nvalids_idx_j, npixels_valid = Metrics.np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    # alfa = sum(np.log(y_)-np.log(y))/npixels_total
    # value = sum(pow(np.log(y)-np.log(y_)+alfa,2))/(2*npixels_total)

    alfa = sum(np.log(y_) - np.log(y)) / npixels_valid
    value = sum(pow(np.log(y) - np.log(y_) + alfa, 2)) / npixels_valid

    # Additional computation way
    # d = np.log(y) - np.log(y_)
    # value2 = sum(pow(d,2))/npixels_valid - pow(sum(d),2)/pow(npixels_valid,2)

    # Debug
    # print("value:",value)
    # input()

    return value


def tf_RMSE_log_scaleInv(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # TODO: Remover?
    # TODO: T é realmente o número de pixels?
    # batchSize, height, width = y_.get_shape().as_list()
    # tf_npixels_total = tf.constant(height*width)

    # Variables

    # Mask out invalid values (values <= 0)!
    tf_y, tf_y_, tf_npixels_valid = tf_maskOutInvalidPixels(y, y_)

    # Calculate Error
    tf_alfa = tf.reduce_sum(tf.log(tf_y_) - tf.log(tf_y)) / tf_npixels_valid
    tf_value = tf.reduce_sum(tf.pow(tf.log(tf_y) - tf.log(tf_y_) + tf_alfa, 2)) / tf_npixels_valid

    # TODO: Remover
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    offset = 0
    batch_data = dataset.train_dataset[offset:(offset + net.train.getBatchSize()), :, :,
                 :]  # (idx, height, width, numChannels)
    batch_labels = dataset.train_labels[offset:(offset + net.train.getBatchSize()), :, :]  # (idx, height, width)
    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}

    rValid_y = sess.run(tf_y, feed_dict=feed_dict)
    rValid_y_ = sess.run(tf_y_, feed_dict=feed_dict)
    rNpixels_valid = sess.run(tf_npixels_valid, feed_dict=feed_dict)

    print()
    print("valid_f_y: ", rValid_y)
    print(rValid_y.shape)
    print()
    print("valid_f_y_: ", rValid_y_)
    print(rValid_y_.shape)
    print()
    print("npixels_valid:", rNpixels_valid)
    print(rNpixels_valid.shape)

    print()
    print("tf_y: ", sess.run([tf_y], feed_dict=feed_dict))
    print()
    print("tf_y_: ", sess.run([tf_y_], feed_dict=feed_dict))
    print()
    print("tf.log(tf_y): ", sess.run([tf.log(tf_y)], feed_dict=feed_dict))
    print()
    print("tf.log(tf_y_): ", sess.run([tf.log(tf_y_)], feed_dict=feed_dict))
    print()
    print("tf.log(tf_y_)-tf.log(tf_y): ", sess.run([tf.log(tf_y_) - tf.log(tf_y)], feed_dict=feed_dict))
    print()
    print("tf.reduce_sum(tf.log(tf_y_)-tf.log(tf_y)): ",
          sess.run([tf.reduce_sum(tf.log(tf_y_) - tf.log(tf_y))], feed_dict=feed_dict))
    print()
    print("tf.reduce_sum(tf.log(tf_y_)-tf.log(tf_y))/tf_npixels_valid_float32: ",
          sess.run([tf.reduce_sum(tf.log(tf_y_) - tf.log(tf_y)) / tf_npixels_valid_float32], feed_dict=feed_dict))
    print()
    print("tf.log(tf_y)-tf.log(tf_y_): ", sess.run([tf.log(tf_y) - tf.log(tf_y_)], feed_dict=feed_dict))
    print()
    print("tf.log(tf_y)-tf.log(tf_y_)+tf_alfa: ",
          sess.run([tf.log(tf_y) - tf.log(tf_y_) + tf_alfa], feed_dict=feed_dict))
    print()
    print("tf.pow(tf.log(tf_y)-tf.log(tf_y_)+tf_alfa,2): ",
          sess.run([tf.pow(tf.log(tf_y) - tf.log(tf_y_) + tf_alfa, 2)], feed_dict=feed_dict))
    print()
    print("tf.reduce_sum(tf.pow(tf.log(tf_y)-tf.log(tf_y_)+tf_alfa,2)): ",
          sess.run([tf.reduce_sum(tf.pow(tf.log(tf_y) - tf.log(tf_y_) + tf_alfa, 2))], feed_dict=feed_dict))
    print()
    print("tf.reduce_sum(tf.pow(tf.log(tf_y)-tf.log(tf_y_)+tf_alfa,2))/tf_npixels_valid_float32: ",
          sess.run([tf.reduce_sum(tf.pow(tf.log(tf_y) - tf.log(tf_y_) + tf_alfa, 2)) / tf_npixels_valid_float32],
                   feed_dict=feed_dict))

    print()
    print("tf_alfa: ", sess.run([tf_alfa], feed_dict=feed_dict))
    print()
    print("value: ", sess.run([tf_value], feed_dict=feed_dict))
    input("Press")

    return tf_value
