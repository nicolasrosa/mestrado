#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  Libraries
# ===========
from matplotlib.legend_handler import HandlerLine2D
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

# =====================
#  Class Configuration
# =====================
# Training Config #
AVG_SIZE = 15
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 30


# ===================
#  Class Declaration
# ===================
class Utils:
    def __init__(self):
        pass

    @staticmethod
    def displayVariableInfo(var):
        try:
            print(var, var.shape,type(var),var.dtype)
        except AttributeError:
            try:
                print(var,type(var),var.dtype)
            except AttributeError:
                print(var,type(var))

    @staticmethod
    def plotLossesHistGraph2(step, trLossC_Hist, trLossF_Hist, vLossF_Hist, netType):
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
    
    @staticmethod
    def displayImage(image, plotTitle=None, fig_id=None):
        fig = plt.figure(fig_id)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        cax = ax.imshow(image, cmap='gray')
        plt.title(plotTitle)
        fig.colorbar(cax)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def displayTestingProgress(test_dataset_crop, test_dataset_normed, tPredictions_f_img, fig_id):
        fig = plt.figure(fig_id)
        fig.clf()
        ax = fig.add_subplot(3, 1, 1)
        cax = ax.imshow(test_dataset_crop, cmap='gray')
        plt.title("test_dataset_crop[k]")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 1, 2)
        cax = ax.imshow(test_dataset_normed, cmap='gray')
        plt.title("test_dataset_normed[k]")
        fig.colorbar(cax)
        
        ax = fig.add_subplot(3, 1, 3)
        cax = ax.imshow(tPredictions_f_img, cmap='gray')
        plt.title("tPredictions_f[k,:,:]")
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


    # def checkOverfitting(validAccRate, step):
    #     global movMean, movMeanLast, stabCounter
    #
    #     movMean.append(validAccRate)
    #
    #     if step > AVG_SIZE:
    #         movMean.popleft()
    #
    #     movMeanAvg = np.sum(movMean) / AVG_SIZE
    #     movMeanAvgLast = np.sum(movMeanLast) / AVG_SIZE
    #
    #     if (movMeanAvg <= movMeanAvgLast) and step > MIN_EVALUATIONS:
    #         # print(step,stabCounter)
    #
    #         stabCounter += 1
    #         if stabCounter > MAX_STEPS_AFTER_STABILIZATION:
    #             print("\nSTOP TRAINING! New samples may cause overfitting!!!")
    #             return True
    #     else:
    #         stabCounter = 0
    #         movMeanLast = deque(movMean)
    #
    #     return False