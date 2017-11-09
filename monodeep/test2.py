#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  MonoDeep
# ===========
# This Coarse-to-Fine Network Architecture predicts the log depth (log y).

# TODO: Adaptar o código para tambem funcionar com o tamanho do nyuDepth
# TODO: Adicionar metricas
# TODO: Adicionar funcao de custo do Eigen, pegar parte do calculo de gradientes da funcao de custo do monodepth
# FIXME: Arrumar dataset_preparation.py, kitti2012.pkl nao possui imagens de teste
# FIXME: Apos uma conversa com o vitor, aparentemente tanto a saida do coarse/fine devem ser lineares, nao eh necessario apresentar o otimizar da Coarse e a rede deve prever log(depth), para isso devo converter os labels para log(y_)

# ===========
#  Libraries
# ===========
import os
import argparse
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import pprint

from scipy.misc import imshow
from monodeep_model import *
from monodeep_dataloader import *

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ===========
#  Functions
# ===========
def argumentHandler():
    # Creating Arguments Parser
    parser = argparse.ArgumentParser("Train the Monodeep Tensorflow implementation taking the dataset.pkl file as input.")

    # Input
    parser.add_argument('-m', '--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument(      '--model_name',                type=str,   help='model name', default='monodeep')
    # parser.add_argument(    '--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    # parser.add_argument(      '--dataset',                   type=str,   help='dataset to train on, kitti, or nyuDepth', default='kitti')
    parser.add_argument('-i', '--data_path',                 type=str,   help="set relative path to the dataset <filename>.pkl file", required=True)
    # parser.add_argument(    '--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument(      '--input_height',              type=int,   help='input height', default=172)
    parser.add_argument(      '--input_width',               type=int,   help='input width', default=576)
    parser.add_argument(      '--batch_size',                type=int,   help='batch size', default=16)
    parser.add_argument('-e', '--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument(      '--max_steps',                 type=int,   help='number of max Steps', default=1000)
    
    parser.add_argument('-l', '--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('-d', '--dropout',                   type=float,  help="enable dropout in the model during training", default=0.5)
    parser.add_argument(      '--ldecay',                    type=bool,  help="enable learning decay", default=False)
    parser.add_argument('-n', '--l2norm',                    type=bool,  help="Enable L2 Normalization", default=False)
    
    parser.add_argument('-t', '--show_train_progress',action='store_true',  help="Show Training Progress Images", default=False)

    parser.add_argument('-o', '--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='output/')
    parser.add_argument(      '--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='log/')
    parser.add_argument(      '--restore_path',              type=str,   help='path to a specific restore to load', default='')
    parser.add_argument(      '--retrain',                               help='if used with restore_path, will restart training from step zero', action='store_true')
    parser.add_argument(      '--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

    # TODO: Adicionar acima
    # parser.add_argument('-t','--showTrainingErrorProgress', action='store_true', dest='showTrainingErrorProgress', help="Show the first batch label, the correspondent Network predictions and the MSE evaluations.", default=False)
    # parser.add_argument('-v','--showValidationErrorProgress', action='store_true', dest='showValidationErrorProgress', help="Show the first validation example label, the Network predictions, and the MSE evaluations", default=False)
    # parser.add_argument('-u', '--showTestingProgress', action='store_true', dest='showTestingProgress', help="Show the first batch testing Network prediction img", default=False)
    # parser.add_argument('-p', '--showPlots', action='store_true', dest='enablePlots', help="Allow the plots being displayed", default=False) # TODO: Correto seria falar o nome dos plots habilitados
    # parser.add_argument('-s', '--save', action='store_true', dest='enableSave', help="Save the trained model for later restoration.", default=False)

    # parser.add_argument('--saveValidFigs', action='store_true', dest='saveValidFigs', help="Save the figures from Validation Predictions.", default=False) # TODO: Add prefix char '-X'
    # parser.add_argument('--saveTestPlots', action='store_true', dest='saveTestPlots', help="Save the Plots from Testing Predictions.", default=False)   # TODO: Add prefix char '-X'
    # parser.add_argument('--saveTestFigs', action='store_true', dest='saveTestFigs', help="Save the figures from Testing Predictions", default=False)    # TODO: Add prefix char '-X'

    return parser.parse_args()


# ===================== #
#  Training/Validation  #
# ===================== #
def train(params, args):
    print('[App] Selected mode: Train')
    print('[App] Selected Params: ')
    print("\t",args)

    # -----------------------------------------
    #  Network Training Model - Building Graph
    # -----------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # Optimizer
        dataloader = MonoDeepDataloader(params, args.mode, args.data_path)
        # model = MonoDeepModel(params, args.mode, dataloader.inputSize, dataloader.outputSize)

        # Variables initialization according to the chosen dataset
        _, image_height, image_width, image_nchannels = dataloader.inputSize # Input
        _, depth_height, depth_width = dataloader.outputSize                      # Output
        fc_hiddenNeurons = depth_height*depth_width

        # print(params)
        # print(image_height, image_width,image_nchannels)
        # print(depth_height, depth_width)

        print("\n[Network/Model] Build Network ..")

        # Layers
        coarse = Coarse(image_height, image_width, depth_height, depth_width, fc_hiddenNeurons)
        fine = Fine()

        with tf.name_scope("Inputs"):
            # TODO: Mudar nomes para tf_image e tf_depth/tf_disp
            tf_image = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_nchannels), name="tf_image")
            tf_labels = tf.placeholder(tf.float32, shape=(None, depth_height, depth_width), name="tf_labels")
            tf_keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            tf_log_labels = tf.log(tf_labels + LOSS_LOG_INITIAL_VALUE)
        
        with tf.name_scope("Outputs"):
            Wh1 = weight_variable([11, 11, 3, 96], "c_Wh1")
            bh1 = bias_variable([96], "c_bh1")

            Wh2 = weight_variable([5, 5, 96, 256], "c_Wh2")
            bh2 = bias_variable([256], "c_bh2")

            Wh3 = weight_variable([3, 3, 256, 384], "c_Wh3")
            bh3 = bias_variable([384], "c_bh3")

            Wh4 = weight_variable([3, 3, 384, 384], "c_Wh4")
            bh4 = bias_variable([384], "c_bh4")

            Wh5 = weight_variable([3, 3, 384, 256], "c_Wh5")
            bh5 = bias_variable([256], "c_bh5")

            Wh5_outputSize_height = round(image_height/32)+1
            Wh5_outputSize_width = round(image_width/32)

            Wh6 = weight_variable([Wh5_outputSize_height*Wh5_outputSize_width*256, fc_hiddenNeurons], "c_Wh6")
            bh6 = bias_variable([fc_hiddenNeurons], "c_bh6")

            depth_numPixels = depth_height * depth_width
            assert (fc_hiddenNeurons == depth_numPixels), "The number of Neurons must be iqual to the number of output pixels."

            Wh7 = weight_variable([fc_hiddenNeurons, depth_numPixels], "c_Wh7")
            bh7 = bias_variable([depth_numPixels], "c_bh7")

            # Network Layers

            conv1 = tf.nn.conv2d(tf_image, filter=Wh1, strides=[1, 4, 4, 1], padding='SAME')
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

            fc1 = tf.reshape(hidden5, [-1, shape_h5[1] * shape_h5[2] * shape_h5[3]])

            hidden6 = tf.nn.relu(tf.matmul(fc1, Wh6) + bh6)

            # TODO: Remover três linhas abaixo
            hidden7_drop = tf.nn.dropout(hidden6, tf_keep_prob)
            hidden7_matmul = tf.matmul(tf.nn.dropout(hidden6, tf_keep_prob), Wh7)
            hidden7_bias = tf.matmul(tf.nn.dropout(hidden6, tf_keep_prob), Wh7) + bh7

            hidden7 = tf.matmul(tf.nn.dropout(hidden6, tf_keep_prob), Wh7) + bh7 # Linear

            fc2 = tf.reshape(hidden7, [-1, depth_height, depth_width])
            tf_predCoarse = fc2

            # Weights and Biases
            Wh1 = weight_variable([9, 9, 3, 63], "f_Wh1")
            bh1 = bias_variable([63], "f_bh1")

            Wh2 = weight_variable([5, 5, 64, 64], "f_Wh2")
            bh2 = bias_variable([64], "f_bh2")

            Wh3 = weight_variable([5, 5, 64, 1], "f_Wh3")
            bh3 = bias_variable([1], "f_bh3")

            image_shape = tf_image.get_shape().as_list()
            predCoarse_shape = tf_predCoarse.get_shape().as_list()

            conv1 = tf.nn.conv2d(tf_image, filter=Wh1, strides=[1, 2, 2, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + bh1)
            pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            int_coarse_dim = tf.expand_dims(tf_predCoarse, 3)
            conc = tf.concat([pool1, int_coarse_dim], axis=3)

            conv2 = tf.nn.conv2d(conc, filter=Wh2, strides=[1, 1, 1, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + bh2)

            conv3 = tf.nn.conv2d(hidden2, filter=Wh3, strides=[1, 1, 1, 1], padding='SAME')
            # hidden3 = tf.nn.relu(conv3 + bh3) # ReLU
            hidden3 = conv3 + bh3 # Linear

            tf_predFine =  hidden3[:, :, :, 0]

        # Debug
        # print(tf_image)
        # print(tf_predCoarse)
        # print(tf_predFine)
        # print(tf_labels)
        # print(tf_log_labels)

        if args.mode == 'test':
            return

        with tf.name_scope("Losses"):
            def tf_MSE(y, y_):
                # Check if y and y* have the same dimensions
                assert((y.shape[1] == y_.shape[1]) and (y.shape[2] == y_.shape[2])), "Houston we've got a problem"

                # Variables
                batchSize, height, width = y_.get_shape().as_list()
                numPixels = height*width
                tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)

                return tf.reduce_sum(tf.pow(y_ - y, 2))/tf_npixels

            def tf_L(y, y_, gamma=0.5):
                # Local Variables
                batchSize, height, width = y_.get_shape().as_list()
                numPixels = height*width

                # Tensorflow Variables

                # tf_npixels = tf.cast(tf.constant(batchSize*numPixels), tf.float32) # TODO: Posso retirar o tamanho do batch da conta? Lembrando que os tensores foram definidos sem especificar o tamanho do batch, logo nao tenho essa informacao aki.
                tf_npixels = tf.cast(tf.constant(numPixels), tf.float32)
                tf_y = y
                tf_y_ = y_
                tf_log_y = tf.log(tf_y + LOSS_LOG_INITIAL_VALUE)
                tf_log_y_ = tf.log(tf_y_ + LOSS_LOG_INITIAL_VALUE)
                tf_d = tf_log_y - tf_log_y_

                tf_loss_d = tf.reduce_sum(tf.pow(tf_d, 2))/tf_npixels
                # tf_loss_d = (tf.reduce_sum(tf.pow(tf_d, 2))/tf_npixels)-((gamma/tf.pow(tf_npixels, 2))*tf.pow(tf.reduce_sum(tf_d), 2))
                # tf_loss_d = (tf.reduce_sum(tf.pow(tf_d, 2))/tf_npixels)-((gamma/tf.pow(tf_npixels, 2))*tf.pow(tf.reduce_sum(tf_d), 2))
                
                return tf_loss_d

            tf_lossF = tf_MSE(tf_predFine, tf_log_labels)\
            # tf_lossF = tf_L(tf_predFine, tf_log_labels)

        # build_summaries()

        # Count Params
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("[Network/Model] Number of trainable parameters: {}".format(total_num_parameters))

        with tf.name_scope("Optimizer"):
            # TODO: Add Learning Decay
            global_step = tf.Variable(0, trainable=False)                                # Count the number of steps taken.
            learningRate = args.learning_rate
            optimizer_f = tf.train.AdamOptimizer(learningRate).minimize(tf_lossF, global_step=global_step)

        with tf.name_scope("Summaries"):
            # Summary/Saver Objects
            saver_folder_path = args.log_directory + args.model_name
            summary_writer = tf.summary.FileWriter(saver_folder_path)                         # FIXME: Tensorboard files not found, not working properly
            # train_saver = tf.train.Saver()                                                  # ~4.3 Gb 
            train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) # ~850 mb

            tf.summary.scalar('learning_rate', args.learning_rate, ['model_0'])
            tf.summary.scalar('tf_lossF', tf_lossF, ['model_0'])
            summary_op = tf.summary.merge_all('model_0')


        # Load checkpoint if set
        # TODO: Terminar
        # if args.checkpoint_path != '':
        #     train_saver.restore(sess, args.checkpoint_path.split(".")[0])

        # if args.retrain:
        #     sess.run(global_step.assign(0))

    # ----------------------------------------
    #  Network Training Model - Running Graph
    # ----------------------------------------
    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as session:
        # Init
        print("[Network/Training] Training Initialized!\n")
        start = time.time()
        tf.global_variables_initializer().run()
        
        fig, axes = plt.subplots(5, 1) # TODO: Mover

        """Training Loop"""
        # TODO: Adicionar loop de epocas
        for step in range(args.max_steps):
            start2 = time.time()
            
            # Training Batch Preparation
            offset = (step * args.batch_size) % (dataloader.train_labels.shape[0] - args.batch_size)      # Pointer
            # print("offset: %d/%d" % (offset,dataloader.train_labels.shape[0]))
            batch_data_colors = dataloader.train_dataset_crop[offset:(offset + args.batch_size), :, :, :] # (idx, height, width, numChannels) - Raw
            batch_data = dataloader.train_dataset[offset:(offset + args.batch_size), :, :, :]             # (idx, height, width, numChannels) - Normalized
            batch_labels = dataloader.train_labels[offset:(offset + args.batch_size), :, :]               # (idx, height, width)

            feed_dict_train = {tf_image: batch_data, tf_labels: batch_labels, tf_keep_prob: args.dropout}
            feed_dict_valid = {tf_image: dataloader.valid_dataset, tf_labels: dataloader.valid_labels, tf_keep_prob: 1.0}

            # ----- Session Run! ----- #
            _, log_labels,trPredictions_c, trPredictions_f, trLoss_f,summary_str = session.run([optimizer_f, tf_log_labels, tf_predCoarse, tf_predFine, tf_lossF, summary_op], feed_dict=feed_dict_train) # Training
            vPredictions_c, vPredictions_f, vLoss_f = session.run([tf_predCoarse, tf_predFine, tf_lossF], feed_dict=feed_dict_valid) # Validation
            
            # TODO: Remover assim que terminar de arrumar o bug dos NaNs na funcao de custo com log
            # def debug():
            # conv1, conv2, conv3, conv4, conv5, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7, fc1, fc2 = session.run([conv1, conv2, conv3, conv4, conv5, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7, fc1, fc2], feed_dict=feed_dict_valid) # Network Evaluation
            # Wh7, bh7, hidden7_drop,hidden7_matmul, hidden7_bias = session.run([Wh7, bh7, hidden7_drop, hidden7_matmul, hidden7_bias], feed_dict=feed_dict_train) # Training
            # y, y_, log_y, log_y_, d, loss_d = session.run([tf_y, tf_y_,tf_log_y, tf_log_y_, tf_d,tf_loss_d], feed_dict=feed_dict_valid) # Loss Function Evaluation

            # print("conv1:",conv1, '\n', conv1.shape, '\n')
            # print("conv2:",conv2, '\n', conv2.shape, '\n')
            # print("conv3:",conv3, '\n', conv3.shape, '\n')
            # print("conv4:",conv4, '\n', conv4.shape, '\n')
            # print("conv5:",conv5, '\n', conv5.shape, '\n')
            # print("hidden1:",hidden1, '\n', hidden1.shape, '\n')
            # print("hidden2:",hidden2, '\n', hidden2.shape, '\n')
            # print("hidden3:",hidden3, '\n', hidden3.shape, '\n')
            # print("hidden4:",hidden4, '\n', hidden4.shape, '\n')
            # print("hidden5:",hidden5, '\n', hidden5.shape, '\n')
            # print("hidden6:",hidden6, '\n', hidden6.shape, '\n')
            # print("hidden7_drop:",hidden7_drop, '\n', hidden7_drop.shape, '\n')
            # print("hidden7_matmul:",hidden7_matmul, '\n', hidden7_matmul.shape, '\n')
            # print("Wh7:",Wh7, '\n', Wh7.shape, '\n')
            # print("bh7:",bh7, '\n', bh7.shape, '\n')
            # print("Wh7:",Wh7.eval(), '\n', Wh7.shape, '\n')
            # print("bh7:",bh7.eval(), '\n', bh7.shape, '\n')
            # print("hidden7_bias:",hidden7_bias, '\n', hidden7_bias.shape, '\n')
            # print("hidden7:",hidden7, '\n', hidden7.shape, '\n')
            # print("fc1:",fc1, '\n', fc2.shape, '\n')
            # print("fc2:",fc2, '\n', fc2.shape, '\n')

            # # print("y:",y, '\n', y.shape, '\n')
            # # print("y_:",y_, '\n', y_.shape, '\n')
            # # print("log_y:",log_y, '\n', log_y.shape,'\n')
            # # print("log_y_:",log_y_, '\n', log_y_.shape, '\n')
            # # print("d:",d, '\n', d.shape, '\n')
            # # print("loss_d:",loss_d, '\n', loss_d.shape, '\n')
            # # print('\n',np.isnan(y).any(), np.isnan(y_).any(), np.isnan(log_y).any(),np.isnan(log_y_).any(),np.isnan(d).any(), np.isnan(loss_d).any())
            # input()

            # debug()
            # -----

            # summary_writer.add_summary(summary_str, global_step=step)

            # Prints Training Progress
            if step % 10 == 0:                
                def plot1(raw, label, log_label, coarse, fine):
                    axes[0].imshow(raw)
                    axes[1].imshow(label)
                    axes[2].imshow(log_label)
                    axes[3].imshow(coarse)
                    axes[4].imshow(fine)
                    plt.pause(0.001)

                if args.show_train_progress:
                    plot1(raw=batch_data_colors[0,:,:], label=batch_labels[0,:,:], log_label=log_labels[0,:,:],coarse=trPredictions_c[0,:,:], fine=trPredictions_f[0,:,:])

                end2 = time.time()

                print('step: %d/%d | t: %f | Batch trLoss_f: %0.4E | vLoss_f: %0.4E' % (step, args.max_steps, end2-start2, trLoss_f, vLoss_f))

        end = time.time()
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s" % (end-start))

        # Saves trained model
        train_saver.save(session, args.log_directory + '/' + args.model_name + '/model') # global_step=last


# ========= #
#  Testing  #
# ========= #
def test(params, args):
    print('[App] Selected mode: Test')
    print('[App] Selected Params: ', args)

    # Load Dataset and Model
    restore_graph = tf.Graph()
    with restore_graph.as_default():
        dataloader = MonoDeepDataloader(params, args.mode, args.data_path)
        model = MonoDeepModel(params, args.mode, dataloader.inputSize, dataloader.outputSize)

    # Session
    with tf.Session(graph=restore_graph) as sess_restore:

        # Saver
        train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
        # Restore
        if args.restore_path == '':
            restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
        else:
            restore_path = args.restore_path.split(".")[0]
        train_saver.restore(sess_restore, restore_path)
        print('\n[Network/Restore] Restoring model from: %s' % restore_path)
        train_saver.restore(sess_restore, restore_path)
        print("[Network/Restore] Model restored!")
        print("[Network/Restore] Restored variables:\n",tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),'\n')
       
        """Testing Loop"""
        num_test_samples = dataloader.test_dataset.shape[0]
        test_predFine = np.zeros((num_test_samples, dataloader.outputSize[1], dataloader.outputSize[2]), dtype=np.float32) 
        for step in range(num_test_samples):
            # Testing Batch Preparation
            feed_dict_test = {tf_image: np.expand_dims(dataloader.test_dataset[step],0), tf_keep_prob: 1.0} # tf_image: (1, height, width, numChannels)

            # ----- Session Run! ----- #
            test_predFine[step] = sess_restore.run(tf_predFine, feed_dict=feed_dict_test)
            # -----

            # Prints Testing Progress
            # print('k: %d | t: %f' % (k, app.timer2.elapsedTime)) # TODO: ativar
            print('step: %d/%d | t: %f' % (step+1, num_test_samples,-1))
        
        # Testing Finished.
        print("\n[Network/Testing] Testing FINISHED!")

        print("[Network/Testing] Saving testing predictions...")
        if args.output_directory == '':
            output_directory = os.path.dirname(args.restore_path)
        else:
            output_directory = args.output_directory

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        np.save(output_directory + 'test_disparities.npy', test_predFine)

        # Show Results
        show_test_disparies = True # TODO: Criar argumento
        if show_test_disparies:
            fig, axes = plt.subplots(3, 1) # TODO: Mover

            for i, image in enumerate(test_predFine):
                
                # TODO: Codigo mais rapido
                # if i == 0:
                #     plt.figure(1)

                # plt.imshow(image)
                # plt.imshow(dataloader.test_dataset_crop[i])
                # # plt.draw()
                # plt.pause(0.01)
                # # plt.show()

                # TODO: Codigo temporario
                def plot2(raw, label, fine):
                    axes[0].imshow(raw)
                    axes[1].imshow(label)
                    axes[2].imshow(fine)
                    plt.title("test_disparities[%d]" % i)

                    plt.pause(0.001)
                
                plot2(dataloader.test_dataset_crop[i],dataloader.test_labels_crop[i],image)



        # TODO: print save done
        
    # TODO: Adaptar e remover
        # for k in range(dataloader.test_dataset.shape[0]):
        #     # Testing Batch Preparation
        #     offset = (k * net.test.getBatchSize()) % (dataloader.test_labels.shape[0] - net.test.getBatchSize())
        #     batch_data = dataloader.test_dataset[offset:(offset + net.test.getBatchSize()), :, :, :]   # (idx, height, width, numChannels)
            
        #     # TODO: Nem todos os datasets possuem testing labels
        #     # batch_labels = dataloader.test_labels[offset:(offset + net.test.getBatchSize()), :, :]     # (idx, height, width)
        #     # feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels, keep_prob: 1.0}
        #     feed_dict_test = {tf_dataset: batch_data, keep_prob: 1.0}

        #     # Session Run!
        #     app.timer2.start()
        #     tPredictions_f = session.run(tf_prediction_f, feed_dict=feed_dict_test)
        #     app.timer2.end()
            
        #     # Prints Testing Progress
        #     print('k: %d | t: %f' % (k, app.timer2.elapsedTime))

        #     if app.args.showTestingProgress or app.args.saveTestPlots:
        #         # Plot.displayImage(dataloader.test_dataset[k,:,:],"dataloader.test_dataset[k,:,:]",6)
        #         # Plot.displayImage(tPredictions_f[0,:,:],"tPredictions_f[0,:,:]",7)
                
        #         # TODO: A variavel dataloader.test_dataset_crop[k] passou a ser um array, talvez a seguinte funcao de problema
        #         Plot.displayTestingProgress(k, dataloader.test_dataset_crop[k], dataloader.test_dataset[k],tPredictions_f[0,:,:],3)

        #         if app.args.saveTestPlots:
        #             app.saveTestPlot(dataset.list_test_colors_files_filename[k])
                
        #     if app.args.saveTestFigs:
        #         app.saveTestFig(dataset.list_test_colors_files_filename[k], tPredictions_f[0,:,:])

        #     if app.args.showTestingProgress:
        #         plt.draw()
        #         plt.pause(0.1)

        # # Testing Finished.
        # # if app.args.showTestingProgress:
        # #     plt.close('all')
        # print("[Network/Testing] Testing FINISHED!")




# ======
#  Main
# ======
def main(args):
    print("[App] Running...")

    params = monodeep_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        dropout=args.dropout,
        full_summary=args.full_summary)

    
    if args.mode == 'train':
        train(params, args)
    elif args.mode == 'test':
        test(params, args)

    print("\n[App] DONE!")
    sys.exit()


# ======
#  Main 
# ======
if __name__ == '__main__':
    args = argumentHandler();
    tf.app.run(main=main(args))
