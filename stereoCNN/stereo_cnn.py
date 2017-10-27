#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  StereoCNN
# ===========
# TODO: Quanto tempo vai demorar?
# TODO: Estudar unidade do Kitti (Como transformar essa escala em metros)?
# TODO: O Vitor deu uma ideia de aplicar aquela ideia de importance sampling (Proporciona uma maior probabilidade de selecionar os exemplos que possuem maior valor de erro para compor o batch de treinamento). Atualmente, o batch criado por exemplos em sequencia.
# TODO: Integrar com a library do Vitor de visualização - Salvar algumas imagens em numpy em txt e enviar a matrix de calibração das cameras.
# TODO: Salvar a imagem de predicao da rede em png com o Range Correto (0 à 15000) e não normalizado como está ocorrendo.
# TODO: No segundo artigo de Fergus, ele utiliza uma técnica que realiza random crops na imagem para acelerar o treinamento da Scale 3 da rede neural desenvolvida. Estudar como implementar essa técnica.

# TODO: Salvar imagens de predição para o caso de test para o Vitor poder gerar a nuvem de pontos. Criar uma pasta na pasta de dataset dele(pasta "disp_algumacoisa") e salvar com o mesmo nome do arquivo 00000X.png
# TODO: Fazer Data Augmentation
# TODO: Criar um novo modelo de Rede utilizando modelos mais complexos (AlexNet ou VGG)
# TODO: Criar um outro modelo baseado em ReNet (Recurrent Networks) assim como apresentado no trabalho de Grigorev
# TODO: Existe também aquela ideia de utilizar layers Partially-Connected, ideia proposta pela tese de Doutorado do Caio Mendes

# ===========
#  Libraries
# ===========
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import misc as scp
from PIL import Image

# TODO: https://docs.python.org/2/library/profile.html
import profile
import pstats

from classes.application import Application
from classes.dataset import DatasetHandler
from classes.network import NetworkModel
from classes.training import Loss
from classes.plots import Plot
from classes.metrics import Metrics
from classes.utils import Utils


# ==================
#  Global Variables
# ==================
enableProfilling = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

metricsObj = Metrics()

# ===========
#  Functions
# ===========


# ======
#  Main
# ======
def main(Application=None):
    # ============================
    #  Application Initialization
    # ============================
    app.timer1.start()
    app.createFolders()

    # =========
    #  Dataset
    # =========
    # Creates the DatasetHandler Object
    dataset = DatasetHandler()

    # First reload the data stored in `dataset.pkl`.
    dataset.load(app.args.pickle_path)

    imageSize = dataset.getImageSize()
    depthSize = dataset.getDepthSize()
    numChannels = dataset.getImageNumChannels()

    dataset.showDatasetInfo()

    # If Restoration is not selected, then train the network.
    if app.args.restore_path is None: 
        # =========================================
        #  Network Training Model - Building Graph
        # =========================================
        print("\n[Network] Constructing Graph...")
        graph = tf.Graph()

        with graph.as_default():
            # TODO: Restoration
            if app.args.restoreContinueTraining:
                pass
            else:
                # Creates the Network Obj
                net = NetworkModel(dataset.train_dataset.shape, dataset.train_labels.shape)
                net.train.setLearningRate(app.args.learningRate)



            # Variables.
            global_step = tf.Variable(0, trainable=False)  # Count the number of steps taken.

            with tf.name_scope("InputData"):
                # The variable `tf_labels` should have the type `tf.int32`, but they're casted to tf.float32. I don't know if this may cause any harm.
                tf_dataset = tf.placeholder(tf.float32, shape=(None, imageSize[0], imageSize[1], numChannels), name="tf_dataset")
                tf_labels = tf.placeholder(tf.float32, shape=(None, depthSize[0], depthSize[1]), name="tf_labels")

                keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # Predictions for the training, validation, and test data.
            with tf.name_scope("NetworkModel"):
                tf_prediction_c = net.model_coarse(tf_dataset, keep_prob, app.args.dropout)
                tf_prediction_f = net.model_fine(tf_dataset, tf_prediction_c)

                

            with tf.name_scope("LossFunction"):
                # TODO: Mudar de tf_MSE para tf_L
                # TODO: Criar uma flag para selecionar qual função de custo deve ser selecionada
                tf_loss_c = Loss.tf_MSE(tf_prediction_c, tf_labels, onlyValidPixels=app.args.maskOut)
                tf_loss_f = Loss.tf_MSE(tf_prediction_f, tf_labels, onlyValidPixels=app.args.maskOut)

                # TODO: Calcular erro se o dataset tiver labels para os casos de testes
                # tf_test_loss_f = Loss.tf_MSE(tf_test_prediction_f, tf_test_labels, onlyValidPixels=app.args.maskOut)

                # tf_loss_c = Loss.tf_L(tf_prediction_c, tf_labels, onlyValidPixels=app.args.maskOut)
                # tf_loss_f = Loss.tf_L(tf_prediction_f, tf_labels, onlyValidPixels=app.args.maskOut)

                if app.args.l2norm:
                    tf_loss_c += Loss.calculateL2norm_Coarse(net)
                    tf_loss_f += Loss.calculateL2norm_Fine(net)
                
            with tf.name_scope("Optimizer"):
                learningRate = net.train.getLearningRate()
                if app.args.ldecay:
                    learningRate = tf.train.exponential_decay(learningRate, global_step, net.train.getldecaySteps(), net.train.getldecayRate(), staircase=True)

                optimizer_c = tf.train.AdamOptimizer(learningRate).minimize(tf_loss_c, global_step=global_step)
                optimizer_f = tf.train.AdamOptimizer(learningRate).minimize(tf_loss_f, global_step=global_step)

        # ========================================
        #  Network Training Model - Running Graph
        # ========================================
        print("[Network/Training] Running built graph...")
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            # TODO: Restoration
            # Restore model weights from previously saved model
            # saver.restore(sess, model_path)
            # print("Model restored from file: %s" % save_path)

            # --------------------- #
            #  Training/Validation  #
            # --------------------- #
            print("[Network/Training] Training Initialized!\n")
            # TODO: Desalocar variaveis desnecessárias, ou que ja foram utilizadas.
            for step in range(app.args.maxSteps):
                app.timer3.start()
                # Training Batch Preparation
                offset = (step * net.train.getBatchSize()) % (dataset.train_labels.shape[0] - net.train.getBatchSize())

                batch_data_colors = dataset.train_dataset_crop[offset:(offset + net.train.getBatchSize()), :, :, :]
                batch_data = dataset.train_dataset[offset:(offset + net.train.getBatchSize()), :, :, :]   # (idx, height, width, numChannels)
                batch_labels = dataset.train_labels[offset:(offset + net.train.getBatchSize()), :, :]     # (idx, height, width)

                # Session Run!
                app.timer2.start()
                
                # Training
                feed_dict_train = {tf_dataset: batch_data, tf_labels: batch_labels, keep_prob: 0.5}
                _, _, trPredictions_c, trPredictions_f, trLoss_c, trLoss_f = session.run([optimizer_c, optimizer_f, tf_prediction_c, tf_prediction_f, tf_loss_c, tf_loss_f], feed_dict=feed_dict_train)
                
                # Validation
                feed_dict_valid = {tf_dataset: dataset.valid_dataset, tf_labels: dataset.valid_labels, keep_prob: 1.0}
                vPredictions_c, vPredictions_f, vLoss_f = session.run([tf_prediction_c, tf_prediction_f, tf_loss_f], feed_dict=feed_dict_valid)
                
                app.timer2.end()

                # Prints Training Progress
                if step % 10 == 0:
                    print('step: %d | t: %f | Batch trLoss_c: %0.4E | Batch trLoss_f: %0.4E | vLoss_f: %0.4E' % (step, app.timer2.elapsedTime, trLoss_c, trLoss_f, vLoss_f))
                    
                    if app.args.maskOut:
                        trPredictions_c[0, :, :] = Loss.np_maskOutInvalidPixels(trPredictions_c[0, :, :], batch_labels[0, :, :])
                        trPredictions_f[0, :, :] = Loss.np_maskOutInvalidPixels(trPredictions_f[0, :, :], batch_labels[0, :, :])
                        vPredictions_f[0, :, :] =  Loss.np_maskOutInvalidPixels( vPredictions_f[0, :, :], dataset.valid_labels[0, :, :])

                    if app.args.showTrainingProgress:
                        # Plot.displayImage(batch_labels[0,:,:], "batch_labels[0,:,:]",1)
                        # Plot.displayImage(trPredictions_c[0,:,:], "trPredictions_c[0,:,:]",2)
                        # Plot.displayImage(trPredictions_f[0,:,:], "trPredictions_f[0,:,:]",3)
                       
                        Plot.displayTrainingProgress(batch_labels[0,:,:], trPredictions_c[0,:,:], trPredictions_f[0,:,:], 1)

                    if app.args.showTrainingErrorProgress:
                        # Lembre que a Training Loss utilizaRMSE_log_scaleInv, porém o resultado é avaliado utilizando MSE
                        train_MSE_c_img = Loss.np_MSE(batch_labels[0,:,:],trPredictions_c[0,:,:])
                        train_MSE_f_img = Loss.np_MSE(batch_labels[0,:,:],trPredictions_f[0,:,:])

                        Plot.displayTrainingErrorProgress(batch_data_colors[0,:,:,:], batch_labels[0,:,:], trPredictions_c[0,:,:], trPredictions_f[0,:,:], train_MSE_c_img, train_MSE_f_img, 8)
                        
                    if app.args.showValidationProgress:
                        # Plot.displayImage(dataset.valid_labels[0,:,:],"dataset.valid_labels[0,:,:]",4)
                        # Plot.displayImage(vPredictions_f[0,:,:],"vPredictions_f[0,:,:]",5)

                        Plot.displayValidationProgress(dataset.valid_labels[0,:,:],vPredictions_f[0,:,:],2)

                    if app.args.showValidationErrorProgress:
                        # print(len(dataset.valid_dataset_crop))
                        # print(len(dataset.valid_labels))
                        # input("Press")

                        app.showValidationErrorProgress(dataset.valid_dataset_crop[0,:,:], vPredictions_f[0,:,:], dataset.valid_labels[0,:,:])
                            

                    # Updates the above figures at the same time
                    if app.args.showTrainingProgress or \
                       app.args.showValidationProgress or \
                       app.args.showTrainingErrorProgress or \
                       app.args.showValidationErrorProgress:
                        plt.draw()
                        plt.pause(0.1)

                # Logs Training/Validation Losses
                if app.args.enablePlots:
                    net.train.lossC_Hist.append(trLoss_c)
                    net.train.lossF_Hist.append(trLoss_f)
                    net.valid.lossF_Hist.append(vLoss_f)

                # Avoids Networks overfitting.
                if net.valid.checkOverfitting(step,vLoss_f):
                    break

                # Save Validation Predictions Figures
                # TODO: Salvar imagens de predição do conjunto de validação seguindo o estilo implementado pra salvar as imagens de teste.
                if app.args.saveValidFigs:
                    # TODO: Encapsular em uma função
                    for k in range(0,len(vPredictions_f)):
                        # print(dataset.list_valid_colors_files_filename)
                        # print(dataset.list_valid_depth_files_filename)
                        # print(len(dataset.list_valid_colors_files_filename))
                        # print(len(dataset.list_valid_depth_files_filename))
                        # print(len(vPredictions_f))
                        # print(len(dataset.valid_labels))
                        # input("Press")

                        app.saveValidFig(dataset.list_valid_colors_files_filename[k], vPredictions_f[k,:,:], dataset.valid_labels[k,:,:])

                app.timer3.end()
                # print("t_loop: %f" % app.timer3.elapsedTime) # TODO: Descomentar para tentar solucionar aquele erro do loop ficar lento


            # Training Finished, clears Training Dataset variables.
            del batch_data, batch_labels
            # if app.args.showTrainingProgress or app.args.showValidationProgress:
            #     plt.close('all')

            print("[Network/Training] Training FINISHED!")

            # --------- #
            #  Testing  #
            # --------- #
            if app.args.showTestingProgress or app.args.saveTestPlots or app.args.saveTestFigs:
                print("[Network/Testing] Testing Evaluation started...")

                # TODO: Printar erro se o dataset possuir labels para os casos de teste
                # print("Test Loss: %f" % tLoss_f.eval())
                # TODO: Descomentar
                for k in range(dataset.test_dataset.shape[0]):
                    # Testing Batch Preparation
                    offset = (k * net.test.getBatchSize()) % (dataset.test_labels.shape[0] - net.test.getBatchSize())
                    batch_data = dataset.test_dataset[offset:(offset + net.test.getBatchSize()), :, :, :]   # (idx, height, width, numChannels)
                    
                    # TODO: Nem todos os datasets possuem testing labels
                    # batch_labels = dataset.test_labels[offset:(offset + net.test.getBatchSize()), :, :]     # (idx, height, width)
                    # feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels, keep_prob: 1.0}
                    feed_dict_test = {tf_dataset: batch_data, keep_prob: 1.0}

                    # Session Run!
                    app.timer2.start()
                    tPredictions_f = session.run(tf_prediction_f, feed_dict=feed_dict_test)
                    app.timer2.end()
                    
                    # Prints Testing Progress
                    print('k: %d | t: %f' % (k, app.timer2.elapsedTime))

                    if app.args.showTestingProgress or app.args.saveTestPlots:
                        # Plot.displayImage(dataset.test_dataset[k,:,:],"dataset.test_dataset[k,:,:]",6)
                        # Plot.displayImage(tPredictions_f[0,:,:],"tPredictions_f[0,:,:]",7)
                        
                        # TODO: A variavel dataset.test_dataset_crop[k] passou a ser um array, talvez a seguinte funcao de problema
                        Plot.displayTestingProgress(k, dataset.test_dataset_crop[k], dataset.test_dataset[k],tPredictions_f[0,:,:],3)

                        if app.args.saveTestPlots:
                            app.saveTestPlot(dataset.list_test_colors_files_filename[k])
                        
                    if app.args.saveTestFigs:
                        app.saveTestFig(dataset.list_test_colors_files_filename[k], tPredictions_f[0,:,:])

                    if app.args.showTestingProgress:
                        plt.draw()
                        plt.pause(0.1)

                # Testing Finished.
                # if app.args.showTestingProgress:
                #     plt.close('all')
                print("[Network/Testing] Testing FINISHED!")

            # ========= #
            #  Results  #
            # ========= #
            # Evaluates the Testing Predictions

            # TODO: Terminar
            # TODO: Mover
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
                
                # TODO: Metricas são aplicadas sobre uma ou várias imagens, acho que não faz sentido apresentarem plots, mas sim apenas os valores obtidos de cada metrica.
                # Plot.displayPlot(step, metricsObj.AbsRelativeDifference_hist, title="AbsRelativeDifference x Step", fig_id=15)
                # Plot.displayPlot(step, metricsObj.SquaredRelativeDifference_hist, title="SquaredRelativeDifference x Step", fig_id=16)
                # Plot.displayPlot(step, metricsObj.RMSE_linear_hist, title="RMSE_linear x Step", fig_id=17)
                # Plot.displayPlot(step, metricsObj.RMSE_log_hist, title="RMSE_log x Step", fig_id=18)
                # Plot.displayPlot(step, metricsObj.RMSE_log_scaleInv_hist, title="RMSE_log_scaleInv x Step", fig_id=19)
                # plt.draw()
                # plt.pause(0.1)
                # input()
                pass

            evaluateTesting()

            # Logs the obtained training, validation and testing results
            f = open(app.save_filename_path_results, 'a')
            f.write("%s\t%s\tsteps: %d/%d\tl: %0.2e\t dropout: %r\tl2norm: %r\tldecay: %r\tmaskOut: %r\trLoss_c: %0.4E\ttrLoss_f: %0.4E\tvLoss_f: %0.4E\n" % (app.datetime, app.appName, step+1, app.args.maxSteps, app.args.learningRate, app.args.dropout, app.args.l2norm, app.args.ldecay, app.args.maskOut, trLoss_c, trLoss_f, vLoss_f))
            f.close()

            # Saves the model variables to disk.
            # TODO: Implementar rotina para salvar/restaurar valores das variaveis do ultimo treinamento
            if app.args.enableSave:
                # Creates saver obj which backups all the variables.
                for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
                    print(i)  # i.name if you want just a name

                saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                
                savemodel_file_path = saver.save(session, app.save_filename_path_net_model)
                print("\n[Results] Model saved to file: %s" % savemodel_file_path)

            # Generates logs for TensorBoard
            if app.args.enableTensorBoard:
                writer = tf.summary.FileWriter(app.save_folder_path_tensorboard_files)
                writer.add_graph(session.graph)

            # Generates the Training Losses and Accuracy plots
            if app.args.enablePlots:
                # ----- Plot 1 ----- #
                print()

                # Plot.displayLossesHistGraph(step, net.train.lossC_Hist, net.train.lossF_Hist, 'CNN')
                Plot.displayLossesHistGraph2(step, net.train.lossC_Hist, net.train.lossF_Hist, net.valid.lossF_Hist, 'CNN')

                print("[Results] Saving generated figure to:", app.savefig_filename_path_loss)            
                plt.savefig(app.savefig_filename_path_loss, bbox_inches='tight')

    else:
        # ============================================
        #  Network Restoration Model - Building Graph
        # ============================================
        restore_graph = tf.Graph()
        with restore_graph.as_default():
            net = NetworkModel(dataset.train_dataset.shape, dataset.train_labels.shape, isRestore=True) 
            
            with tf.name_scope("InputData"):
                # The variable `tf_labels` should have the type `tf.int32`, but they're casted to tf.float32. I don't know if this may cause any harm.
                tf_dataset = tf.placeholder(tf.float32, shape=(None, imageSize[0], imageSize[1], numChannels), name="tf_dataset")
                tf_labels = tf.placeholder(tf.float32, shape=(None, depthSize[0], depthSize[1]), name="tf_labels")

                keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # Predictions for the training, validation, and test data.
            with tf.name_scope("NetworkModel"):
                tf_prediction_c = net.model_coarse(tf_dataset, keep_prob, app.args.dropout)
                tf_prediction_f = net.model_fine(tf_dataset, tf_prediction_c)


        # ===========================================
        #  Network Restoration Model - Running Graph
        # ===========================================
        with tf.Session(graph=restore_graph) as sess_restore:
            print('\n[Network/Restore] Restoring model from: %s' % app.restore_filename_path_net_model)
    
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            saver.restore(sess_restore, app.restore_filename_path_net_model)

            print("[Network/Restore] Model restored!")
            print("[Network/Restore] Restored variables:\n",tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
            # TODO: Fazer algum esquema que me permita analisar o modelo restaurado tanto para as imagens de treinamento, validacao e treinamento

            for k in range(dataset.train_dataset.shape[0]):
                # Training Batch Preparation
                # TODO: FIGURAS NAO ESTAO SINCRONIZADAS, VERIFICAR variaveis de indexacao 'offset' e 'k' lembrar que a rede foi desenvolvida para tamanho de batch 16, talvez nao possa dar feed nela só com uma imagem. 
                offset = (k * net.rest.getBatchSize()) % (dataset.train_labels.shape[0] - net.rest.getBatchSize())
                batch_data = dataset.train_dataset[offset:(offset + net.rest.getBatchSize()), :, :, :]   # (idx, height, width, numChannels)
                batch_raw = dataset.train_labels[offset:(offset + net.rest.getBatchSize()), :, :]     # (idx, height, width)

                # Session Run!
                # batch_labels = dataset.train_labels[offset:(offset + net.rest.getBatchSize()), :, :]     # (idx, height, width)
                # feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels, rT_keep_prob: 1.0}
                feed_dict_rest = {tf_dataset: batch_data, keep_prob: 1.0}
                rtrPredictions_c, rtrPredictions_f = sess_restore.run((tf_prediction_c, tf_prediction_f), feed_dict=feed_dict_rest)

                # TODO: A variavel dataset.train_dataset_crop[k] passou a ser um array, talvez a seguinte funcao de problema
                Plot.displayTrainingProgress(batch_raw[0,:,:], rtrPredictions_c[0,:,:],rtrPredictions_f[0,:,:],4)
                plt.show()                

            for k in range(dataset.test_dataset.shape[0]):
                # Testing Batch Preparation
                offset = (k * net.rest.getBatchSize()) % (dataset.test_labels.shape[0] - net.rest.getBatchSize())
                batch_data = dataset.test_dataset[offset:(offset + net.rest.getBatchSize()), :, :, :]   # (idx, height, width, numChannels)
                
          
                # Session Run!
                # TODO: Nem todos os datasets possuem testing labels
                # batch_labels = dataset.test_labels[offset:(offset + net.rest.getBatchSize()), :, :]     # (idx, height, width)
                # feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels, rT_keep_prob: 1.0}
                feed_dict = {tf_dataset: batch_data, keep_prob: 1.0}
                rtPredictions_f = sess_restore.run(tf_rTest_prediction_f, feed_dict=feed_dict)

                # TODO: A variavel dataset.test_dataset_crop[k] passou a ser um array, talvez a seguinte funcao de problema
                Plot.displayTestingProgress(k, dataset.test_dataset_crop[k], dataset.test_dataset[k],rtPredictions_f[0,:,:],5)
                plt.show()


    # =======================
    #  Finishing Application 
    # =======================
    # Stops Timer 1
    app.timer1.end()
    print("\n[App] The elapsed time was: %f s"% app.timer1.elapsedTime)

    # Shows the Simulations Plots
    if app.args.enablePlots:
        plt.show()  # Must be close to the end of code!
    
    # TODO: Identificar sessões abertas para serem fechadas
    # session.close()
    # del session
    print("[App] DONE")
    sys.exit()


# ======
#  Main 
# ======
if __name__ == '__main__':    
    app = Application(enableProfilling=False)

    if enableProfilling:
        profile.run('main()',app.save_filename_path_profile) 
        p = pstats.Stats(app.save_filename_path_profile)
        p.sort_stats('cumulative').print_stats(20)
    else: 
        tf.app.run(main=main(app))
        