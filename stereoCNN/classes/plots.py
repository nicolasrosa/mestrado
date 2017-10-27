#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  Libraries
# ===========
from matplotlib.legend_handler import HandlerLine2D

import matplotlib.pyplot as plt
import numpy as np

# ===================
#  Class Declaration
# ===================
class Plot():
    def __init__(self):
        pass
      
    def displayImage(image, plotTitle=None, fig_id=None):
        fig = plt.figure(fig_id)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        cax = ax.imshow(image, cmap='gray')
        plt.title(plotTitle)
        fig.colorbar(cax)

    # TODO: Terminar
    def displayPlot(step, var, title=None, xlabel=None, ylabel=None, fig_id=None):
        x = np.arange(step + 1)
        y = np.array(var)

        plt.figure()
        # line1, = plt.plot(x, y, 'b-', label='Training Coarse Loss')
        plt.plot(x,y)

        plt.xlim(0, step)
        plt.ylim(0, np.amax(np.array([np.amax(y)])) * 1.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
        plt.grid()

        # plt.show()        

    def displayLossesHistGraph2(step, trLossC_Hist, trLossF_Hist, vLossF_Hist, netType):
        print("[Results] Generating Training Losses History graphic...")

        x1 = np.arange(step + 1)
        y1 = np.array(trLossC_Hist)
        x2 = np.arange(step + 1)
        y2 = np.array(trLossF_Hist)
        x3 = np.arange(step + 1)
        y3 = np.array(vLossF_Hist)

        plt.figure()
        line1, = plt.plot(x1, y1, 'b-', label='Training Coarse Loss')
        line2, = plt.plot(x2, y2, 'r-', label='Training Fine Loss')
        line3, = plt.plot(x3, y3, 'g-', label='Validation Fine Loss')
        
        plt.xlim(0, step)
        plt.ylim(0, np.amax(np.array([np.amax(y1), np.amax(y2), np.amax(y3)])) * 1.1)

        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('[' + netType + ']' + ' Training/Validation Losses x numSteps')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
        plt.grid()

        # plt.show()

    def displayTrainingProgress(batch_labels_img, trPredictions_c_img, trPredictions_f_img, fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        ax = fig.add_subplot(3, 1, 1)
        cax = ax.imshow(batch_labels_img, cmap='gray')
        plt.title("batch_labels[0,:,:]")
        fig.colorbar(cax)
        
        ax = fig.add_subplot(3, 1, 2)
        cax = ax.imshow(trPredictions_c_img, cmap='gray')
        plt.title("trPredictions_c[0,:,:]")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 1, 3)
        cax = ax.imshow(trPredictions_f_img, cmap='gray')
        plt.title("trPredictions_f[0,:,:]")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    def displayTrainingErrorProgress(batch_data_crop_img,batch_labels_img, trPredictions_c_img, trPredictions_f_img, MSE_c_error, MSE_f_error, fig_id):
        
        fig = plt.figure(fig_id)
        fig.clf()

        ax = fig.add_subplot(3, 2, 1)
        cax = ax.imshow(batch_labels_img, cmap='gray')
        plt.title("batch_labels[0,:,:]")
        fig.colorbar(cax)

        # TODO: Tentar resolver o problema que a imagem rgb não era correspondente com o label        
        # ax = fig.add_subplot(3, 2, 2)
        # cax = ax.imshow(batch_data_crop_img)
        # plt.title("batch_data_crop[0,:,:]")
        # fig.colorbar(cax)
        
        ax = fig.add_subplot(3, 2, 3)
        cax = ax.imshow(trPredictions_c_img, cmap='gray')
        plt.title("trPredictions_c[0,:,:]")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 5)
        cax = ax.imshow(trPredictions_f_img, cmap='gray')
        plt.title("trPredictions_f[0,:,:]")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 4)
        cax = ax.imshow(MSE_c_error, cmap='jet')
        plt.title("MSE - Coarse")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 6)
        cax = ax.imshow(MSE_f_error, cmap='jet')
        plt.title("MSE - Fine")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # TODO: Tem alguma coisa errada com esse plot, espaco grande à esquerda do plot.
    def displayValidationErrorProgress(valid_data_colors_img, vPredictions_f_img, valid_labels_img, MSE_f_error, fig_id):
        
        fig = plt.figure(fig_id)
        fig.clf()

        # TODO: Tentar resolver o problema que a imagem rgb não era correspondente com o label        
        # ax = fig.add_subplot(4, 1, 1)
        # cax = ax.imshow(valid_data_colors_img)
        # plt.title("valid_dataset_crop[0,:,:]")
        # fig.colorbar(cax)

        ax = fig.add_subplot(3, 1, 1)
        # ax = fig.add_subplot(4, 1, 2)
        cax = ax.imshow(valid_labels_img, cmap='gray')
        plt.title("valid_labels[0,:,:]")
        fig.colorbar(cax)
        
        ax = fig.add_subplot(3, 1, 2)
        # ax = fig.add_subplot(4, 1, 3)
        cax = ax.imshow(vPredictions_f_img, cmap='gray')
        plt.title("vPredictions_f[0,:,:]")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 1, 3)
        # ax = fig.add_subplot(4, 1, 4)
        cax = ax.imshow(MSE_f_error, cmap='jet')
        plt.title("MSE - Fine")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    def displayValidationProgress(valid_labels_img, vPredictions_f_img,fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        cax = ax.imshow(valid_labels_img, cmap='gray')
        plt.title("valid_labels[0,:,:]")
        fig.colorbar(cax)
        
        ax = fig.add_subplot(2, 1, 2)
        cax = ax.imshow(vPredictions_f_img, cmap='gray')
        plt.title("vPredictions_f[0,:,:]")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    def displayTestingProgress(k,test_dataset_crop,test_dataset_normed, tPredictions_f_img,fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        ax = fig.add_subplot(3, 1, 1)
        cax = ax.imshow(test_dataset_crop, cmap='gray')
        plt.title("test_dataset_crop[%s]" % str(k))
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 1, 2)
        cax = ax.imshow(test_dataset_normed, cmap='gray')
        plt.title("test_dataset_normed[%s]" % str(k))
        fig.colorbar(cax)
        
        ax = fig.add_subplot(3, 1, 3)
        cax = ax.imshow(tPredictions_f_img, cmap='gray')
        plt.title("tPredictions_f[%s,:,:]"  % str(k))
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # def plotLossHistoryGraph(step, lossHist, netType):
    #     print("[Results] Generating average lossHistory graphic...")

    #     x = np.arange(step + 1)
    #     y = np.array(lossHist)

    #     plt.figure()
    #     plt.plot(x, y, 'r-')
    #     plt.xlim(0, step + 1)
    #     plt.ylim(0, np.amax(y) * 1.1)

    #     plt.xlabel('Step')
    #     plt.ylabel('Average Loss L')
    #     plt.title('[' + netType + ']' + ' Average Loss L x numSteps')
    #     plt.grid()

    #     # plt.show()


    # def plotLossesHistGraph(step, trLossC_Hist, trLossF_Hist, netType):
    #     print("[Results] Generating Training Losses History graphic...")

    #     x = np.arange(step + 1)
    #     y = np.array(trLossC_Hist)
    #     x2 = np.arange(step + 1)
    #     y2 = np.array(trLossF_Hist)

    #     plt.figure()
    #     line1, = plt.plot(x, y, 'b-', label='Coarse Loss')
    #     line2, = plt.plot(x2, y2, 'r-', label='Fine Loss')
    #     plt.xlim(0, step)
    #     plt.ylim(0, np.amax(y) * 1.1 if np.amax(y) > np.amax(y2) else np.amax(y2) * 1.1)

    #     plt.xlabel('Step')
    #     plt.ylabel('Accuracy')
    #     plt.title('[' + netType + ']' + ' Training Losses x numSteps')
    #     plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    #     plt.grid()

    #     # plt.show()

    # def plotAccHistGraph(step, trAccHist, vAccHist, netType):
    #     print("[Results] Generating Training/Validation Accuracies History graphic...")

    #     x = np.arange(step + 1)
    #     y = np.array(trAccHist)
    #     x2 = np.arange(step + 1)
    #     y2 = np.array(vAccHist)

    #     plt.figure()
    #     line1, = plt.plot(x, y, 'b-', label='Training')
    #     line2, = plt.plot(x2, y2, 'r-', label='Validation')
    #     plt.xlim(0, step)
    #     plt.ylim(0, np.amax(np.array([np.amax(y), np.amax(y2)])) * 1.1)
    #     plt.xlabel('Step')
    #     plt.ylabel('Accuracy')
    #     plt.title('[' + netType + ']' + ' Training/Validation Accuracies x numSteps')
    #     plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    #     plt.grid()

    #     # plt.show()


    # def plotValidAccHistGraph(step, vAccHist, netType):
    #     print("\n[Results] Generating Validation Accuracy History graphic...")

    #     x = np.arange(step + 1)
    #     y = np.array(vAccHist)

    #     plt.figure()
    #     plt.plot(x, y, 'r-')
    #     plt.xlim(0, step)
    #     plt.ylim(0, np.amax(y) * 1.1)

    #     plt.xlabel('Step')
    #     plt.ylabel('Validation Accuracy')
    #     plt.title('[' + netType + ']' + ' Validation Accuracy x numSteps')
    #     plt.grid()

    #     # plt.show()