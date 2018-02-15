#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =======
#  To-Do
# =======
# FIXME: Após uma conversa com o vitor, aparentemente tanto a saida do coarse/fine devem ser lineares, nao eh necessario apresentar o otimizar da Coarse e a rede deve prever log(depth), para isso devo converter os labels para log(y_)
# TODO: Validar Métricas.
# TODO: Adicionar mais topologias

# Known Bugs
# Leitura e Processamento das Imagens estão sendo feitos na CPU
# Bilinear está sendo utilizado corretamente?
# Data Augmentation - Brightness not working
# SAVE_TEST_DISPARITIES - Funciona, mas nao uso propriamente

import os
import sys
import time
import warnings
from collections import deque

import numpy as np
# ===========
#  Libraries
# ===========
import tensorflow as tf

import utils.args as args
import utils.metrics as metrics
from utils.importNetwork import ImportGraph
from utils.monodeep_dataloader import MonodeepDataloader
from utils.monodeep_model import MonodeepModel
from utils.plot import Plot

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings

appName = 'monodeep'
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

ENABLE_EARLY_STOP = True
SAVE_TRAINED_MODEL = True
ENABLE_TENSORBOARD = True
SAVE_TEST_DISPARITIES = True
APPLY_BILINEAR_OUTPUT = False

# Early Stop Configuration
AVG_SIZE = 20
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 10000
LOSS_LOG_INITIAL_VALUE = 0.1

# ===========
#  Functions
# ===========
def createSaveFolder():
    save_path = None
    save_restore_path = None

    if SAVE_TRAINED_MODEL or ENABLE_TENSORBOARD:
        # Saves the model variables to disk.
        relative_save_path = 'output/' + appName + '/' + datetime + '/'
        save_path = os.path.join(os.getcwd(), relative_save_path)
        save_restore_path = os.path.join(save_path, 'restore/')

        if not os.path.exists(save_restore_path):
            os.makedirs(save_restore_path)

    return save_path, save_restore_path


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args, params):
    print('[%s] Selected mode: Train' % appName)
    print('[%s] Selected Params: ' % appName)
    print("\t", args)

    # Local Variables
    movMeanLast = 0
    movMean = deque()

    save_path, save_restore_path = createSaveFolder()

    # -----------------------------------------
    #  Network Training Model - Building Graph
    # -----------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # MonoDepth
        dataloader = MonodeepDataloader(args.data_path, params, args.dataset, args.mode)
        params['inputSize'] = dataloader.inputSize
        params['outputSize'] = dataloader.outputSize

        model = MonodeepModel(args.mode, params)

        def test_dataAug():
            for i in range(10):
                image, _, _, _ = dataloader.readImage(dataloader.train_dataset[i], dataloader.train_labels[i],
                                                      mode='train')
            input("Continue")

        # test_dataAug() # TODO: Remover, após terminar de implementar dataAugmentation Transforms

        with tf.name_scope("Summaries"):
            # Summary Objects
            summary_writer = tf.summary.FileWriter(save_path + args.log_directory, graph)
            summary_op = tf.summary.merge_all('model_0')

        # Creates Saver Obj
        train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # ----------------------------------------
    #  Network Training Model - Running Graph
    # ----------------------------------------
    # Local Variables and Memory Allocation
    step, stabCounter = 0, 0
    train_lossF, valid_lossF = None, None

    batch_data = np.zeros((args.batch_size,
                           dataloader.inputSize[1],
                           dataloader.inputSize[2],
                           dataloader.inputSize[3]),
                          dtype=np.float64)  # (?, 172, 576, 3)

    batch_data_crop = np.zeros((args.batch_size,
                                dataloader.inputSize[1],
                                dataloader.inputSize[2],
                                dataloader.inputSize[3]),
                               dtype=np.uint8)  # (?, 172, 576, 3)

    valid_data_o = np.zeros((len(dataloader.valid_dataset),
                                dataloader.inputSize[1],
                                dataloader.inputSize[2],
                                dataloader.inputSize[3]),
                               dtype=np.float64)  # (?, 172, 576, 3) # FIXME: Nao deveria ser uint8 para cada canal?

    valid_data_crop_o = np.zeros((len(dataloader.valid_dataset),
                             dataloader.inputSize[1],
                             dataloader.inputSize[2],
                             dataloader.inputSize[3]),
                            dtype=np.uint8)  # (?, 172, 576, 3)

    batch_labels = np.zeros((args.batch_size,
                             dataloader.outputSize[1],
                             dataloader.outputSize[2]),
                            dtype=np.int32)  # (?, 43, 144)

    valid_labels_o = np.zeros((len(dataloader.valid_labels),
                               dataloader.outputSize[1],
                               dataloader.outputSize[2]),
                              dtype=np.int32)  # (?, 43, 144)

    # TODO: Faz sentido ter bilinear no train e no valid?
    # if APPLY_BILINEAR_OUTPUT:
    #     batch_labelsBilinear = np.zeros((args.batch_size,
    #                              dataloader.inputSize[1],
    #                              dataloader.inputSize[2]),
    #                             dtype=np.int32)  # (?, 172, 576)
    #
    #     valid_labelsBilinear_o = np.zeros((len(dataloader.valid_labels),
    #                                dataloader.inputSize[1],
    #                                dataloader.inputSize[2]),
    #                               dtype=np.int32)  # (?, 172, 576)

    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        print("[Network/Training] Training Initialized!\n")

        # Proclaim the epochs
        epochs = np.floor(args.batch_size * args.max_steps / dataloader.numTrainSamples)
        print('Train with approximately %d epochs' % epochs)

        # =================
        #  Training Loop
        # =================
        start = time.time()
        if args.show_train_progress:
            train_plotObj = Plot(args.mode, title='Train Predictions')

        if args.show_valid_progress:
            valid_plotObj = Plot(args.mode, title='Validation Prediction')

        for i in range((len(dataloader.valid_dataset))):
            image, depth, image_crop, _ = dataloader.readImage(dataloader.valid_dataset[i],
                                                      dataloader.valid_labels[i],
                                                      mode='valid',
                                                      showImages=False)

            valid_data_o[i] = image
            valid_labels_o[i] = depth
            valid_data_crop_o[i] = image_crop

        for step in range(args.max_steps):
            start2 = time.time()

            # Training and Validation Batches and Feed Dictionary Preparation
            offset = (step * args.batch_size) % (dataloader.numTrainSamples - args.batch_size)  # Pointer
            batch_data_path = dataloader.train_dataset[offset:(offset + args.batch_size)]
            batch_labels_path = dataloader.train_labels[offset:(offset + args.batch_size)]

            # print("offset: %d/%d" % (offset,dataloader.numTrainSamples))
            # print(batch_data_path)
            # print(len(batch_data_path))

            for i in range(len(batch_data_path)):
                # FIXME: os tipos retornados das variaveis estao errados, quando originalmente eram uint8 e int32, lembrar que o placeholder no tensorflow é float32
                image, depth, image_crop, _ = dataloader.readImage(batch_data_path[i],
                                                                   batch_labels_path[i],
                                                                   mode='train',
                                                                   showImages=False)

                # print(image.dtype,depth.dtype, image_crop.dtype, depth_crop.dtype)

                batch_data[i] = image
                batch_labels[i] = depth
                batch_data_crop[i] = image_crop

            feed_dict_train = {model.tf_image: batch_data, model.tf_labels: batch_labels,
                               model.tf_keep_prob: args.dropout}

            feed_dict_valid = {model.tf_image: valid_data_o, model.tf_labels: valid_labels_o,
                               model.tf_keep_prob: 1.0}

            # ----- Session Run! ----- #
            _, log_labels, train_PredCoarse, train_PredFine, train_lossF, summary_str = session.run(
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
                if args.show_train_progress:
                    train_plotObj.showTrainResults(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],
                                                   log_label=log_labels[0, :, :],
                                                   coarse=train_PredCoarse[0, :, :], fine=train_PredFine[0, :, :])

                    # Plot.plotTrainingProgress(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],log_label=log_labels[0, :, :], coarse=train_PredCoarse[0, :, :],fine=train_PredFine[0, :, :], fig_id=3)

                if args.show_train_error_progress:
                    Plot.plotTrainingErrorProgress(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],
                                                   coarse=train_PredCoarse[0, :, :], fine=train_PredFine[0, :, :],
                                                   figId=8)

                if args.show_valid_progress:
                    valid_plotObj.showValidResults(raw=valid_data_crop_o[0, :, :, :], label=valid_labels_o[0], log_label=np.log(valid_labels_o[0]+LOSS_LOG_INITIAL_VALUE),coarse=valid_PredCoarse[0], fine=valid_PredFine[0])

                end2 = time.time()
                print('step: {0:d}/{1:d} | t: {2:f} | Batch trLoss: {3:>16.4f} | vLoss: {4:>16.4f} '.format(step,
                                                                                                            args.max_steps,
                                                                                                            end2 - start2,
                                                                                                            train_lossF,
                                                                                                            valid_lossF))

        end = time.time()
        sim_train = end - start
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s\n" % sim_train)

        # ==============
        #  Save Results
        # ==============
        if SAVE_TRAINED_MODEL:
            model.saveTrainedModel(save_restore_path, session, train_saver, args.model_name)

        # Logs the obtained test result
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\t%s\t\t%s\t\tsteps: %d\ttrain_lossF: %f\tvalid_lossF: %f\t%f\n" % (
            datetime, args.model_name, args.dataset, model.loss_name,  step, train_lossF,
            valid_lossF, sim_train))
        f.close()


# ========= #
#  Testing  #
# ========= #
def test(args, params):
    print('[%s] Selected mode: Test' % appName)
    print('[%s] Selected Params: %s' % (appName, args))

    # TODO: fazer rotina para pegar imagens externas, nao somente do dataset
    # -----------------------------------------
    #  Network Testing Model - Importing Graph
    # -----------------------------------------
    # Loads the dataset and restores a specified trained model.
    dataloader = MonodeepDataloader(args.data_path, params, args.dataset, args.mode)
    model = ImportGraph(args.restore_path)

    # Memory Allocation
    # Length of test_dataset used, so when there is not test_labels, the variable will still be declared.
    predCoarse = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                          dtype=np.float32)  # (?, 43, 144)

    predFine = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                        dtype=np.float32)  # (?, 43, 144)

    test_labels_o = np.zeros((len(dataloader.test_dataset), dataloader.outputSize[1], dataloader.outputSize[2]),
                             dtype=np.int32)  # (?, 43, 144)

    test_data_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    test_data_crop_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    predCoarseBilinear = np.zeros((dataloader.numTestSamples, dataloader.inputSize[1], dataloader.inputSize[2]),
                                  dtype=np.float32)  # (?, 172, 576)

    predFineBilinear = np.zeros((dataloader.numTestSamples, dataloader.inputSize[1], dataloader.inputSize[2]),
                                dtype=np.float32)  # (?, 172, 576)

    # TODO: Usar?
    # test_labelsBilinear_o = np.zeros(
    #     (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2]),
    #     dtype=np.int32)  # (?, 172, 576)

    # ==============
    #  Testing Loop
    # ==============
    start = time.time()
    for i, image_path in enumerate(dataloader.test_dataset):
        start2 = time.time()

        if dataloader.test_labels:  # It's not empty
            image, depth, image_crop, depth_bilinear = dataloader.readImage(dataloader.test_dataset[i],
                                                                            dataloader.test_labels[i],
                                                                            mode='test')

            test_labels_o[i] = depth
            # test_labelsBilinear_o[i] = depth_bilinear # TODO: Usar?
        else:
            image, _, image_crop, _ = dataloader.readImage(dataloader.test_dataset[i], None, mode='test')

        test_data_o[i] = image
        test_data_crop_o[i] = image_crop

        if APPLY_BILINEAR_OUTPUT:
            predCoarse[i], predFine[i], predCoarseBilinear[i], predFineBilinear[i] = model.networkPredict(image,
                                                                                                          APPLY_BILINEAR_OUTPUT)
        else:
            predCoarse[i], predFine[i] = model.networkPredict(image)

        # Prints Testing Progress
        end2 = time.time()
        print('step: %d/%d | t: %f' % (i + 1, dataloader.numTestSamples, end2 - start2))
        # break # Test

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

    if SAVE_TEST_DISPARITIES:
        np.save(output_directory + 'test_coarse_disparities.npy', predCoarse)
        np.save(output_directory + 'test_fine_disparities.npy', predFine)

    # Calculate Metrics
    if dataloader.test_labels:
        metrics.evaluateTesting(predFine, test_labels_o)
    else:
        print(
            "[Network/Testing] It's not possible to calculate Metrics. There are no corresponding labels for Testing Predictions!")

    # Show Results
    if args.show_test_results:
        test_plotObj = Plot(args.mode, title='Test Predictions')
        for i in range(dataloader.numTestSamples):
            test_plotObj.showTestResults(test_data_crop_o[i], test_labels_o[i], predCoarse[i], predFine[i], i)


# ======
#  Main
# ======
def main(args):
    """ This Coarse-to-Fine Network Architecture predicts the log depth (log y)."""
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
    args = args.argumentHandler()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tf.app.run(main=main(args))
