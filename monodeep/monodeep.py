#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =======
#  To-Do
# =======
# FIXME: Após uma conversa com o vitor, aparentemente tanto a saida do coarse/fine devem ser lineares, nao eh necessario apresentar o otimizar da Coarse e a rede deve prever log(depth), para isso devo converter os labels para log(y_)
# TODO: Validar Métricas.

# ===========
#  Libraries
# ===========
import tensorflow as tf
import numpy as np
import time
import os
import sys
import utils.metrics as metrics
import utils.args as args

from collections import deque

from utils.monodeep_dataloader import MonodeepDataloader
from utils.monodeep_model import MonodeepModel
from utils.importNetwork import ImportGraph
from utils.plot import Plot

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

appName = 'monodeep'
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

ENABLE_EARLY_STOP = True
ENABLE_RESTORE = True
ENABLE_TENSORBOARD = True
SAVE_TEST_DISPARITIES = True
SHOW_TEST_DISPARITIES = True
APPLY_BILINEAR_ON_OUTPUT = False  # FIXME: Will not work if onlyValidPixels is True

# Early Stop Configuration
AVG_SIZE = 20
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 10000


# ===========
#  Functions
# ===========
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


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args, params):
    # Local Variables
    movMeanLast = 0
    movMean = deque()

    print('[%s] Selected mode: Train' % appName)
    print('[%s] Selected Params: ' % appName)
    print("\t", args)

    save_path, save_restore_path = createSaveFolder()

    # -----------------------------------------
    #  Network Training Model - Building Graph
    # -----------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # MonoDepth
        dataloader = MonodeepDataloader(args.data_path, params, args.dataset, args.mode, APPLY_BILINEAR_ON_OUTPUT)
        params['inputSize'] = dataloader.inputSize
        params['outputSize'] = dataloader.outputSize

        model = MonodeepModel(args.mode, params, APPLY_BILINEAR_ON_OUTPUT)

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
    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as session:
        # Local Variables
        step, stabCounter = 0, 0
        train_lossF, valid_lossF = None, None

        tf.global_variables_initializer().run()

        print("[Network/Training] Training Initialized!\n")

        # Proclaim the epochs
        epochs = np.floor(args.batch_size * args.max_steps / dataloader.numTrainSamples)
        print('Train with approximately %d epochs' % epochs)

        # Memory Allocation
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

        valid_dataset_o = np.zeros((len(dataloader.valid_dataset),
                                    dataloader.inputSize[1],
                                    dataloader.inputSize[2],
                                    dataloader.inputSize[3]),
                                   dtype=np.uint8)  # (?, 172, 576, 3)

        if APPLY_BILINEAR_ON_OUTPUT:
            batch_labels = np.zeros((args.batch_size,
                                     dataloader.inputSize[1],
                                     dataloader.inputSize[2]),
                                    dtype=np.int32)  # (?, 172, 576)

            valid_labels_o = np.zeros((len(dataloader.valid_labels),
                                       dataloader.inputSize[1],
                                       dataloader.inputSize[2]),
                                      dtype=np.int32)  # (?, 172, 576)
        else:
            batch_labels = np.zeros((args.batch_size,
                                     dataloader.outputSize[1],
                                     dataloader.outputSize[2]),
                                    dtype=np.int32)  # (?, 43, 144)

            valid_labels_o = np.zeros((len(dataloader.valid_labels),
                                       dataloader.outputSize[1],
                                       dataloader.outputSize[2]),
                                      dtype=np.int32)  # (?, 43, 144)

        # =================
        #  Training Loop
        # =================
        start = time.time()
        if args.show_train_progress:
            train_plotObj = Plot(args.mode, title='Train Predictions')  # TODO: Qual o melhor lugar para essa linhas?

        for i in range((len(dataloader.valid_dataset))):
            valid_dataset_o[i], valid_labels_o[i], _, _ = dataloader.readImage(dataloader.valid_dataset[i],
                                                                               dataloader.valid_labels[i],
                                                                               mode='valid',
                                                                               showImages=False)

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
                image, depth, image_crop, depth_crop = dataloader.readImage(batch_data_path[i],
                                                                            batch_labels_path[i],
                                                                            mode='train',
                                                                            showImages=False)

                # print(image.dtype,depth.dtype, image_crop.dtype, depth_crop.dtype)

                batch_data[i] = image
                batch_labels[i] = depth
                batch_data_crop[i] = image_crop

            feed_dict_train = {model.tf_image: batch_data, model.tf_labels: batch_labels,
                               model.tf_keep_prob: args.dropout}

            feed_dict_valid = {model.tf_image: valid_dataset_o, model.tf_labels: valid_labels_o,
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

                end2 = time.time()
                print('step: {0:d}/{1:d} | t: {2:f} | Batch trLoss: {3:>16.4f} | vLoss: {4:>16.4f} '.format(step,
                                                                                                            args.max_steps,
                                                                                                            end2 - start2,
                                                                                                            train_lossF,
                                                                                                            valid_lossF))

        end = time.time()
        sim_train = end - start
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s" % sim_train)

        # ==============
        #  Save Results
        # ==============
        if ENABLE_RESTORE:
            model.saveTrainedModel(save_restore_path, session, train_saver, args.model_name)

        # Logs the obtained test result
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\t%s\t\tsteps: %d\ttrain_lossF: %f\tvalid_lossF: %f\t%f\n" % (
            datetime, appName, args.dataset, step, train_lossF,
            valid_lossF, sim_train))  # TODO: Nao salvar o appName, e sim o nome do model utilizado.
        f.close()


# ========= #
#  Testing  #
# ========= #
def test(args, params):
    # Local Variables
    if SHOW_TEST_DISPARITIES:
        test_plotObj = Plot(args.mode, title='Test Predictions')  # TODO: Qual o melhor lugar para essa linhas?

    print('[%s] Selected mode: Test' % appName)
    print('[%s] Selected Params: %s' % (appName, args))

    # TODO: fazer rotina para pegar imagens externas, nao somente do dataset
    # Loads the dataset and restores a specified trained model.
    dataloader = MonodeepDataloader(args.data_path, params, args.dataset, args.mode, APPLY_BILINEAR_ON_OUTPUT)
    model = ImportGraph(args.restore_path)

    # Memory Allocation
    # Length of test_dataset used, so when there is not test_labels, the variable will still be declared.

    if APPLY_BILINEAR_ON_OUTPUT:
        predCoarse = np.zeros((dataloader.numTestSamples, dataloader.inputSize[1], dataloader.inputSize[2]),
                              dtype=np.float32)  # (?, 172, 576)

        predFine = np.zeros((dataloader.numTestSamples, dataloader.inputSize[1], dataloader.inputSize[2]),
                            dtype=np.float32)  # (?, 172, 576)

        test_labels_o = np.zeros((len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2]),
                                 dtype=np.int32)  # (?, 172, 576)
    else:
        predCoarse = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                              dtype=np.float32)  # (?, 43, 144)

        predFine = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                            dtype=np.float32)  # (?, 43, 144)

        test_labels_o = np.zeros((len(dataloader.test_dataset), dataloader.outputSize[1], dataloader.outputSize[2]),
                                 dtype=np.int32)  # (?, 43, 144)

    test_dataset_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    test_dataset_crop_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    # ==============
    #  Testing Loop
    # ==============
    start = time.time()

    for i, image_path in enumerate(dataloader.test_dataset):
        start2 = time.time()

        if dataloader.test_labels:  # It's not empty
            image, depth, image_crop = dataloader.readImage(dataloader.test_dataset[i], dataloader.test_labels[i],
                                                            mode='test')

            # print(image.shape, depth.shape, image_crop.shape)
            test_labels_o[i] = depth
        else:
            image, _, image_crop = dataloader.readImage(dataloader.test_dataset[i], None, mode='test')

        test_dataset_o[i] = image
        test_dataset_crop_o[i] = image_crop

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
    if SHOW_TEST_DISPARITIES:
        for i in range(dataloader.numTestSamples):
            test_plotObj.showTestResults(test_dataset_crop_o[i], test_labels_o[i], predCoarse[i], predFine[i], i)


# ======
#  Main
# ======
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
    tf.app.run(main=main(args))
