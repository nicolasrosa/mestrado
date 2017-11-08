#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  StereoCNN
# ===========

# ===========
#  Libraries
# ===========
from classes.network import NetworkModel
from classes.dataset import DatasetHandler
from classes.utils import *
from classes.plots import Plot

import tensorflow as tf
import os
import sys
import argparse
import time

# ==================
#  Global Variables
# ==================
# App Config #
appName = os.path.splitext(sys.argv[0])[0]
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

ENABLE_RESTORE = 0
ENABLE_TENSORBOARD = 1

# Training Config #
AVG_SIZE = 10
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 30

# Network Config #
batch_size = 1
LOG_INITIAL_VALUE = 1E-6

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape) # TODO: Aparentemente, um valor de 0.1, inicia a rede com um menor valor de loss.
    return tf.Variable(initial)

# ======
#  Main
# ======
def main(_):
    # ----- Arguments Parser ----- #
    # Creating Arguments Parser
    parser = argparse.ArgumentParser("Trains the StereoCNN Network taking the dataset.pickle file as input.")
    parser.add_argument('-i', '--dataset', action='store', dest='dataset',
                        help="Selects the dataset ['kitti2012','kitti2015','nyu_v2']")
    parser.add_argument('-e', '--maxSteps', action='store', dest='maxSteps',
                        help="Defines the maximum number of training epochs", type=int, default=1000)
    parser.add_argument('-p', '--showPlots', action='store_true', dest='enablePlots',
                        help="Allows the plots being displayed", default=False)
    parser.add_argument('-t', '--showTrainingProgress', action='store_true', dest='showTrainingProgress',
                        help="Shows the first batch label and the correspondent Network prediction images", default=False)
    parser.add_argument('-d', '--enableDropout', action='store_true', dest='enableDropout',
                        help="Applies Dropout in the model during training.", default=False)
    args = parser.parse_args()

    # ----- Dataset ----- #
    dataset = DatasetHandler()

    # First reload the data stored in `dataset.ipynb`.
    selected_dataset = os.path.splitext(os.path.basename(args.dataset))[0] # Without Path and extention
    dataset.load(args.dataset)

    imageSize = dataset.getImageSize()
    depthSize = dataset.getDepthSize()
    numChannels = dataset.getImageNumChannels()

    dataset.showDatasetInfo()

    if selected_dataset == 'nyu_v2': 
        depthOutputSize = (57, 76)
        fc_hiddenNeurons = 4332
        Wh6_size = 8*10*256
    elif selected_dataset == 'kitti2012' or 'kitti2015':
        depthOutputSize = (43, 144)
        fc_hiddenNeurons = 6192
        Wh6_size = 6*18*256

    # ----- Network Model ----- #
    print("[Network] Constructing Graph...")
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, imageSize[0], imageSize[1], numChannels))
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, depthSize[0], depthSize[1]))


        tf_test_dataset = tf.constant(dataset.test_dataset)

        # TODO: Implementar rotina para salvar/restaurar valores das variaeveis do ultimo treinamento
        # Variables.
        Wh1 = weight_variable([11, 11, 3, 96])
        bh1 = bias_variable([96])

        Wh2 = weight_variable([5, 5, 96, 256])
        bh2 = bias_variable([256])

        Wh3 = weight_variable([3, 3, 256, 384])
        bh3 = bias_variable([384])

        Wh4 = weight_variable([3, 3, 384, 384])
        bh4 = bias_variable([384])

        Wh5 = weight_variable([3, 3, 384, 256])
        bh5 = bias_variable([256])

        # TODO: Não sei se as dimensões das próximas variáveis estão corretas.
        # TODO: Supostamente, o tamanho deveria ser 4096, não 4060(kitti)
        fc_hiddenNeurons = 4060
        Wh6 = weight_variable([Wh6_size, fc_hiddenNeurons])
        bh6 = bias_variable([fc_hiddenNeurons])
        keep_prob = tf.placeholder(tf.float32)

        # TODO: Criar assert que verifica se o numero de neuronios da ultima camada de saída é igual ao número de pi
        depthOutput_numPixels = depthOutputSize[0] * depthOutputSize[1]
        Wh7 = weight_variable([fc_hiddenNeurons, depthOutput_numPixels])
        bh7 = bias_variable([depthOutput_numPixels])

        # Model.
        def model(data):
            shape_data = data.get_shape().as_list()

            conv1 = tf.nn.conv2d(data, filter=Wh1, strides=[1, 4, 4, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + bh1)
            pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv2 = tf.nn.conv2d(pool1, filter=Wh2, strides=[1, 1, 1, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + bh2)
            pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3 = tf.nn.conv2d(pool2, filter=Wh3, strides=[1, 1, 1, 1], padding='SAME')
            hidden3 = tf.nn.relu(conv3 + bh3)

            conv4 = tf.nn.conv2d(hidden3, filter=Wh4, strides=[1, 1, 1, 1], padding='SAME')
            hidden4 = tf.nn.relu(conv4 + bh4)

            conv5 = tf.nn.conv2d(hidden4, filter=Wh5, strides=[1, 2, 2, 1], padding='SAME')
            hidden5 = tf.nn.relu(conv5 + bh5)
            shape_h5 = hidden5.get_shape().as_list()

            fc1 = tf.reshape(hidden5, [shape_h5[0], shape_h5[1] * shape_h5[2] * shape_h5[3]])
            hidden6 = tf.nn.relu(tf.matmul(fc1, Wh6) + bh6)
            fc1_drop = tf.nn.dropout(hidden6, keep_prob)
            shape_h6 = hidden6.get_shape().as_list()

            # TODO: According to Eigen, the last neurons layer is supposed to be linear, instead of ReLU. However,
            # sometimes we get negative logits values, and the tf.log gets complex numbers (nan).
            hidden7 = tf.matmul(fc1_drop if args.enableDropout else hidden6,Wh7) + bh7 # Linear
            # hidden7 = tf.nn.relu(tf.matmul(fc1_drop if args.enableDropout else hidden6, Wh7) + bh7)  # ReLU
            fc2 = tf.reshape(hidden7, [-1, depthOutputSize[0], depthOutputSize[1]])

            # Debug
            print("\nNetwork Layers:")
            print("i:",shape_data) 
            print()

            print("1:",conv1.get_shape().as_list())      
            print(hidden1.get_shape().as_list())    
            print(pool1.get_shape().as_list())      
            print()

            print("2:",conv2.get_shape().as_list())      
            print(hidden2.get_shape().as_list())    
            print(pool2.get_shape().as_list())      
            print()

            print("3:",conv3.get_shape().as_list())      
            print(hidden3.get_shape().as_list())    
            print()

            print("4:",conv4.get_shape().as_list())      
            print(hidden4.get_shape().as_list())    
            print()

            print("5:",conv5.get_shape().as_list())      
            print(hidden5.get_shape().as_list())    
            print()

            print("6:",fc1.get_shape().as_list())     
            print(hidden6.get_shape().as_list())
            print(fc1_drop.get_shape().as_list())    
            print()

            print("7:",hidden7.get_shape().as_list())     
            print(fc2.get_shape().as_list())

            return fc2

        # Predictions for the training, validation, and test data.
        prediction = model(tf_dataset)

        # Compute Loss Function
        loss = tf.reduce_sum(tf.pow(tf_labels - prediction, 2) / (batch_size*4060)) # numPixels

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

       

    # ----- Running Model ----- #
    movMeanLast = 0
    movMean = deque()
    stabCounter = 0

    lossHistory = []
    trainAccHist = []
    validAccHist = []

    print("[Network] Running built graph...")
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        print("[Network] Training Initialized!\n")
        
        # TODO: Colocar em um lugar melhor
        colorbar = False 
 
        for step in range(args.maxSteps):
            offset = (step * batch_size) % (dataset.train_labels.shape[0] - batch_size)
            batch_data = dataset.train_dataset[0:(0 + batch_size), :, :, :]               # (idx, height, width, numChannels)
            batch_labels = dataset.train_labels[0:(0 + batch_size), :, :]                 # (idx, height, width)

            start = time.time()
            feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels, keep_prob: 0.5}
            _, rLoss, trPredictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
            end = time.time()

            if args.enablePlots:
                lossHistory.append(rLoss)

            # Prints Training Progressing
            if step % 1 == 0:
                print('step: %d | t: %f | loss: %f' % (step, end - start, rLoss))

                if args.showTrainingProgress:
                    Plot.displayTrainingProgress(batch_labels[0,:,:], trPredictions[0,:,:], 1)


                    # plt.figure(1)
                    # plt.imshow(batch_labels[0,:,:],cmap='gray')
                    # plt.title("batch_labels[0,:,:]")
                    # if colorbar == False:
                    #     plt.colorbar()  # TODO: O range da barra deve estar de acordo com cada imagem, e não seguir o da primeira imagem.

                    # plt.figure(2)
                    # plt.imshow(trPredictions[0,:,:], cmap='gray')
                    # plt.title("trPredictions[0,:,:]")
                    # if colorbar == False:
                    #     plt.colorbar() # TODO: O range da barra deve estar de acordo com cada imagem, e não seguir o da primeira imagem.
                    #     colorbar=True

                    plt.draw()
                    plt.pause(0.01)
                    # input("Press ENTER to continue...")

            # TODO: Descomentar Linhas abaixo
            # Avoids Networks overfitting.
            # movMean.append(validAccRate)

            # if step > AVG_SIZE:
            #     movMean.popleft()

            # movMeanAvg = np.sum(movMean) / AVG_SIZE
            # movMeanAvgLast = np.sum(movMeanLast) / AVG_SIZE

            # if (movMeanAvg <= movMeanAvgLast) and step > MIN_EVALUATIONS:
            #     # print(step,stabCounter)

            #     stabCounter += 1
            #     if stabCounter > MAX_STEPS_AFTER_STABILIZATION:
            #         print("\nSTOP TRAINING! New samples may cause overfitting!!!")
            #         break
            # else:
            #     stabCounter = 0

            # movMeanLast = deque(movMean)

            # TODO: Descomentar linhas abaixo
            # if checkOverfitting(validAccRate, step):
            #     break

        # Training Finished, prints Test Dataset accuracy.
        # TODO: Lembrar que o model de Test não deve apresentar dropout, isto é, keep_prob = 1.0. Implementar mudanças
        #  que respeitem isso.
        # TODO: Aqui é o melhor lugar pra normalizar essa variável?

        # ----- Saving ----- #
        # save_path = os.path.join(os.getcwd(), getRestoreFilesPath(appName))
        if args.enablePlots or ENABLE_RESTORE:
            # Saves the model variables to disk.
            save_path = os.path.join(os.getcwd(), getRestoreFilesPath(appName,datetime))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        if ENABLE_RESTORE:
            # Creates saver obj which backups all the variables.
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
                print(i)  # i.name if you want just a name

            saver = tf.train.Saver([prediction])

            file_path = saver.save(session, os.path.join(save_path, appName + "_model.ckpt"))
            print("\n[Results] Model saved in file: %s" % file_path)

            # print("Wh: ",session.run('Wh:0'))
            # print("bh: ",session.run('bh:0'))

        # Generates logs for TensorBoard
        if ENABLE_TENSORBOARD:
            writer = tf.summary.FileWriter("/tmp/tensorflow/1")
            writer.add_graph(session.graph)

    # ----- Results ------ #
    # Generates the Training Loss and Accuracy plots
    if args.enablePlots:
        print()
        savefig_path = getSaveImagesPath(appName,datetime) + appName + '_loss.png'
        print("[Results] Saving generated figure to: ", savefig_path)
        plotLossHistoryGraph(step, lossHistory, 'CNN')
        # TODO: Descomentar
        # plt.savefig(getSaveImagesPath(appName) + appName + '_loss.png', bbox_inches='tight')

        # plotValidAccHistGraph(validAccHist, options.mode)

        # TODO: Descomentar
        # plotAccHistGraph(step, trainAccHist, validAccHist, 'CNN')
        # savefig_path = getSaveImagesPath(appName) + appName + '_acc.png'
        # print("[Results] Saving generated figure to: ", savefig_path)
        # plt.savefig(getSaveImagesPath(appName) + appName + '_acc.png', bbox_inches='tight')

        plt.show()  # Must be close to the end of code!

    sys.exit()


# ======
#  Main
# ======
if __name__ == '__main__':
    tf.app.run(main=main)