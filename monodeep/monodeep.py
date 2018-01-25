#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =======
#  To-Do
# =======
# This Coarse-to-Fine Network Architecture predicts the log depth (log y).

# TODO: Adaptar o código para tambem funcionar com o tamanho do nyuDepth
# TODO: Adicionar metricas (T is the total number of pixels in all the evaluated images)
# TODO: Adicionar funcao de custo do Eigen, pegar parte do calculo de gradientes da funcao de custo do monodepth
# FIXME: Arrumar dataset_preparation.py, kitti2012.pkl nao possui imagens de teste
# FIXME: Apos uma conversa com o vitor, aparentemente tanto a saida do coarse/fine devem ser lineares, nao eh necessario apresentar o otimizar da Coarse e a rede deve prever log(depth), para isso devo converter os labels para log(y_)

# ===========
#  Libraries
# ===========
import argparse
import matplotlib.pyplot as plt
import sys
import time

# import numpy as np
# import pprint

from collections import deque
# from scipy.misc import imshow

from monodeep_dataloader import *
from monodepth_dataloader import * # TODO: Remover
from monodeep_model import *
from importNetwork import *

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

appName = 'monodeep'
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

ENABLE_EARLY_STOP = True
ENABLE_RESTORE = True
ENABLE_TENSORBOARD = True
ENABLE_SAVE_CHECKPOINT = False
SHOW_TEST_DISPARITIES = True

# Early Stop Configuration
AVG_SIZE = 20
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 10000


# ===========
#  Functions
# ===========
def argumentHandler():
    # Creating Arguments Parser
    parser = argparse.ArgumentParser("Train the Bitnet Tensorflow implementation taking the dataset.pkl file as input.")

    # Input
    parser.add_argument('-m', '--mode', type=str, help="Select 'train' or 'test' mode", default='train')
    parser.add_argument('--model_name', type=str, help="Select Network topology: 'monodeep', etc",
                        default='monodeep')  # TODO: Adicionar mais topologias
    # parser.add_argument(    '--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    parser.add_argument('-i', '--data_path', type=str,
                        help="Set relative path to the input dataset <filename>.pkl file",
                        required=True)
    parser.add_argument('--batch_size', type=int, help="Define the Training batch size", default=16)
    parser.add_argument('--max_steps', type=int, help="Define the number of max Steps", default=1000)
    parser.add_argument('-l', '--learning_rate', type=float, help="Define the initial learning rate", default=1e-4)
    parser.add_argument('-d', '--dropout', type=float, help="Enable dropout in the model during training", default=0.5)
    parser.add_argument('--ldecay', action='store_true', help="Enable learning decay", default=False)
    parser.add_argument('-n', '--l2norm', action='store_true', help="Enable L2 Normalization", default=False)

    parser.add_argument('--full_summary', action='store_true',
                        help="If set, will keep more data for each summary. Warning: the file can become very large")

    parser.add_argument('--log_directory', type=str, help="Set directory to save checkpoints and summaries",
                        default='log_tb/')
    parser.add_argument('-r', '--restore_path', type=str, help="Set path to a specific restore to load", default='')

    parser.add_argument('-t', '--show_train_progress', action='store_true', help="Show Training Progress Images",
                        default=False)

    parser.add_argument('-o', '--output_directory', type=str,
                        help='output directory for test disparities, if empty outputs to checkpoint folder',
                        default='output/')

    # TODO: Adicionar acima
    # parser.add_argument('-t','--showTrainingErrorProgress', action='store_true', dest='showTrainingErrorProgress', help="Show the first batch label, the correspondent Network predictions and the MSE evaluations.", default=False)
    # parser.add_argument('-v','--showValidationErrorProgress', action='store_true', dest='showValidationErrorProgress', help="Show the first validation example label, the Network predictions, and the MSE evaluations", default=False)
    # parser.add_argument('-u', '--showTestingProgress', action='store_true', dest='showTestingProgress', help="Show the first batch testing Network prediction img", default=False)
    # parser.add_argument('-p', '--showPlots', action='store_true', dest='enablePlots', help="Allow the plots being displayed", default=False) # TODO: Correto seria falar o nome dos plots habilitados
    # parser.add_argument('-s', '--save', action='store_true', dest='enableSave', help="Save the trained model for later restoration.", default=False)

    # parser.add_argument('--saveValidFigs', action='store_true', dest='saveValidFigs', help="Save the figures from Validation Predictions.", default=False) # TODO: Add prefix char '-X'
    # parser.add_argument('--saveTestPlots', action='store_true', dest='saveTestPlots', help="Save the Plots from Testing Predictions.", default=False)   # TODO: Add prefix char '-X'
    # parser.add_argument('--saveTestFigs', action='store_true', dest='saveTestFigs', help="Save the figures from Testing Predictions", default=False)    # TODO: Add prefix char '-X'

    # parser.add_argument('--retrain', help='if used with restore_path, will restart training from step zero', action='store_true')

    return parser.parse_args()


def createSaveFolder():
    save_path = None
    save_restore_path = None

    if ENABLE_RESTORE or ENABLE_TENSORBOARD:
        # Saves the model variables to disk.
        relative_save_path = 'output/' + appName + '/' + datetime + '/'
        save_path = os.path.join(os.getcwd(), relative_save_path)
        save_restore_path = os.path.join(save_path, 'restore/')

        if not os.path.exists(save_restore_path):
            os.makedirs(save_restore_path)

    return save_path, save_restore_path


def createPlotsObj(mode, title):
    if mode == 'train':
        # TODO: Mover, Validar
        fig, axes = plt.subplots(5, 1)
        axes[0] = plt.subplot(321)
        axes[1] = plt.subplot(323)
        axes[2] = plt.subplot(325)
        axes[3] = plt.subplot(322)
        axes[4] = plt.subplot(324)

    elif mode == 'test':
        fig, axes = plt.subplots(4, 1)
        axes[0] = plt.subplot(411)
        axes[1] = plt.subplot(412)
        axes[2] = plt.subplot(413)
        axes[3] = plt.subplot(414)

    fig = plt.gcf()
    fig.canvas.set_window_title(title)

    return fig, axes


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args, params):
    # Local Variables
    movMeanLast = 0
    movMean = deque()
    fig, axes = createPlotsObj(args.mode, title='Train Predictions')

    print('[%s] Selected mode: Train' % appName)
    print('[%s] Selected Params: ' % appName)
    print("\t", args)

    save_path, save_restore_path = createSaveFolder()

    # -----------------------------------------
    #  Network Training Model - Building Graph
    # -----------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # TODO: Terminar
        # MonoDeep
        dataloader = MonoDeepDataloader(args, params, args.data_path)
        params['inputSize'] = dataloader.inputSize
        params['outputSize'] = dataloader.outputSize

        # MonoDepth
        # dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        # left = dataloader.left_image_batch
        # right = dataloader.right_image_batch

        model = MonoDeepModel(args.mode, params)

        with tf.name_scope("Summaries"):
            # Summary Objects
            summary_writer = tf.summary.FileWriter(save_path + args.log_directory, graph)
            summary_op = tf.summary.merge_all('model_0')

        # Creates Saver Obj
        train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # ----------------------------------------
    #  Network Training Model - Running Graph
    # ----------------------------------------
    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as session:
        # Local Variables
        stabCounter = 0
        tf.global_variables_initializer().run()

        print("[Network/Training] Training Initialized!\n")

        # Proclaim the epochs
        epochs = np.floor(
            args.batch_size * args.max_steps / dataloader.numTrainSamples)
        print('Train with approximately %d epochs' % epochs)

        # =================
        #  Training Loop
        # =================
        start = time.time()
        for step in range(args.max_steps):
            start2 = time.time()

            # Training Batch and Feed Dictionary Preparation
            offset = (step * args.batch_size) % (dataloader.numTrainSamples - args.batch_size)  # Pointer
            # print("offset: %d/%d" % (offset,dataloader.train_labels.shape[0]))
            batch_data_colors = dataloader.train_dataset_crop[offset:(offset + args.batch_size), :, :,
                                :]  # (idx, height, width, numChannels) - Raw
            batch_data = dataloader.train_dataset[offset:(offset + args.batch_size), :, :,
                         :]  # (idx, height, width, numChannels) - Normalized
            batch_labels = dataloader.train_labels[offset:(offset + args.batch_size), :, :]  # (idx, height, width)

            feed_dict_train = {model.tf_image: batch_data, model.tf_labels: batch_labels,
                               model.tf_keep_prob: args.dropout}

            feed_dict_valid = {model.tf_image: dataloader.valid_dataset, model.tf_labels: dataloader.valid_labels,
                               model.tf_keep_prob: 1.0}

            # ----- Session Run! ----- #
            _, log_labels, train_PredCoarse, train_PredFine, train_lossFine, summary_str = session.run(
                [model.train, model.tf_log_labels, model.tf_predCoarse, model.tf_predFine, model.tf_lossF,
                 summary_op], feed_dict=feed_dict_train)  # Training
            valid_PredCoarse, valid_PredFine, valid_lossF = session.run(
                [model.tf_predCoarse, model.tf_predFine, model.tf_lossF], feed_dict=feed_dict_valid)  # Validation
            # -----

            if ENABLE_TENSORBOARD:
                # Write information to TensorBoard
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file


            # TODO: Não faz sentido eu ter validAccRate, uma vez que eu não meço Accuracy, eu apenas monitoro o erro.
            # TODO: Validar, original era movMeanAvg <= movMeanAvgLast
            if ENABLE_EARLY_STOP:
                movMean.append(valid_lossF)

                if step > AVG_SIZE:
                    movMean.popleft()

                movMeanAvg = np.sum(movMean) / AVG_SIZE
                movMeanAvgLast = np.sum(movMeanLast) / AVG_SIZE

                if (movMeanAvg >= movMeanAvgLast) and step > MIN_EVALUATIONS:
                    # print(step,stabCounter)

                    stabCounter += 1
                    if stabCounter > MAX_STEPS_AFTER_STABILIZATION:
                        print("\nSTOP TRAINING! New samples may cause overfitting!!!")
                        break
                else:
                    stabCounter = 0

                movMeanLast = deque(movMean)

            # Prints Training Progress
            if step % 10 == 0:
                def plot1(raw, label, log_label, coarse, fine):
                    axes[0].imshow(raw)
                    axes[0].set_title("Raw")
                    axes[1].imshow(label)
                    axes[1].set_title("Label")
                    axes[2].imshow(log_label)
                    axes[2].set_title("log(Label)")
                    axes[3].imshow(coarse)
                    axes[3].set_title("Coarse")
                    axes[4].imshow(fine)
                    axes[4].set_title("Fine")
                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

                    plt.pause(0.001)

                if args.show_train_progress:
                    plot1(raw=batch_data_colors[0, :, :], label=batch_labels[0, :, :], log_label=log_labels[0, :, :],
                          coarse=train_PredCoarse[0, :, :], fine=train_PredFine[0, :, :])

                end2 = time.time()

                print('step: {0:d}/{1:d} | t: {2:f} | Batch trLoss: {3:>16.4f} | vLoss: {4:>16.4f} '.format(step, args.max_steps, end2 - start2, train_lossFine, valid_lossF))

        end = time.time()
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s" % (end - start))

        # ==============
        #  Save Results
        # ==============
        # TODO: Mover
        def saveTrainedModel(save_path, session):
            """ Saves trained model """
            # Creates saver obj which backups all the variables.
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(i)  # i.name if you want just a name

            # train_saver = tf.train.Saver()                                                  # ~4.3 Gb
            # train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # ~850 mb

            file_path = train_saver.save(session, os.path.join(save_path,
                                                               "model." + args.model_name + ".ckpt"))  # TODO: Acredito que seja possível remover .ckpt. Rodar networkTraining em modo 'test' e networkPredict_example.py para validar essa mudanca.

            print("\n[Results] Model saved in file: %s" % file_path)

        if ENABLE_RESTORE:
            saveTrainedModel(save_restore_path, session)

        # Logs the obtained test result
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\tsteps: %d\ttrain_lossFine: %f\n" % (datetime, appName, step, train_lossFine))
        f.close()


# ========= #
#  Testing  #
# ========= #
def test(args, params):
    # Local Variables
    fig, axes = createPlotsObj(args.mode, title='Test Predictions')

    print('[%s] Selected mode: Test' % appName)
    print('[%s] Selected Params: %s' % (appName, args))

    # TODO: fazer rotina para pegar imagens externas, nao somente do dataset
    # Loads the dataset and restores a specified trained model.
    dataloader = MonoDeepDataloader(args, params, args.data_path)
    model = ImportGraph(args.restore_path)

    predCoarse = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                                 dtype=np.float32)
    predFine = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                                 dtype=np.float32)

    # ==============
    #  Testing Loop
    # ==============
    start = time.time()
    for i, image in enumerate(dataloader.test_dataset_crop):
        start2 = time.time()
        predCoarse[i], predFine[i] = model.networkPredict(image)

        def plot_test(raw, coarse, fine):
            # print(raw.shape)
            # print(coarse.shape)
            # print(fine.shape)
            axes[0].imshow(raw)
            axes[0].set_title("Raw")
            axes[1].imshow(coarse)
            axes[1].set_title("Coarse")
            axes[2].imshow(fine)
            axes[2].set_title("Fine")
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            plt.pause(0.001)

        # plot_test(raw=image, coarse=predCoarse[0, :, :], fine=predFine[0, :, :])

        # Prints Testing Progress
        # print('k: %d | t: %f' % (k, app.timer2.elapsedTime)) # TODO: ativar
        end2 = time.time()
        print('step: %d/%d | t: %f' % (i + 1, numTestSamples, end2 - start2))

    # Testing Finished.
    end = time.time()
    print("\n[Network/Testing] Testing FINISHED! Time elapsed: %f s" % (end - start))

    # ==============
    #  Save Results
    # ==============
    # Saves the Test Predictions
    print("[Network/Testing] Saving testing predictions...")
    if args.output_directory == '':
        output_directory = os.path.dirname(args.restore_path)
    else:
        output_directory = args.output_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(output_directory + 'test_coarse_disparities.npy', predCoarse)
    np.save(output_directory + 'test_fine_disparities.npy', predFine)

    # Show Results
    if SHOW_TEST_DISPARITIES:
        for i in dataloader.numTestSamples:
            def showTestResults(raw, label, coarse, fine):
                plt.figure(1)
                axes[0].set_title("Raw[%d]" % i)
                axes[0].imshow(raw)
                axes[1].set_title("Label")
                axes[1].imshow(label)
                axes[2].set_title("Coarse")
                axes[2].imshow(coarse)
                axes[3].set_title("Fine")
                axes[3].imshow(fine)

                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.pause(0.001)

            showTestResults(dataloader.test_dataset_crop[i], dataloader.test_labels_crop[i], predCoarse[i], predFine[i])


# ======
#  Main
# ======
def main(args):
    print("[%s] Running..." % appName)

    modelParams = {'inputSize': -1, 'outputSize': -1, 'model_name': args.model_name,
                   'learning_rate': args.learning_rate, 'batch_size': args.batch_size,
                   'max_steps': args.max_steps, 'dropout': args.dropout, 'ldecay': args.ldecay, 'l2norm': args.l2norm,
                   'full_summary': args.full_summary}

    if args.mode == 'train':
        train(args, modelParams)
    elif args.mode == 'test':
        test(args, modelParams)

    print("\n[%s] Done." % appName)
    sys.exit()


# ======
#  Main
# ======
if __name__ == '__main__':
    args = argumentHandler()
    tf.app.run(main=main(args))
