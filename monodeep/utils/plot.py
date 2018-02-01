# ===========
#  Libraries
# ===========
import matplotlib.pyplot as plt
import numpy as np

# ===========
#  Functions
# ===========


# ===================
#  Class Declaration
# ===================
class Plot(object):
    def __init__(self, mode, title):
        self.fig, self.axes = None, None

        if mode == 'train':
            # TODO: Mover, Validar
            self.fig, self.axes = plt.subplots(5, 1)
            self.axes[0] = plt.subplot(321)
            self.axes[1] = plt.subplot(323)
            self.axes[2] = plt.subplot(325)
            self.axes[3] = plt.subplot(322)
            self.axes[4] = plt.subplot(324)

        elif mode == 'test':
            self.fig, self.axes = plt.subplots(4, 1)
            self.axes[0] = plt.subplot(221)
            self.axes[1] = plt.subplot(222)
            self.axes[2] = plt.subplot(223)
            self.axes[3] = plt.subplot(224)

        self.fig = plt.gcf() # TODO: Posso remover?
        self.fig.canvas.set_window_title(title)

    # TODO: Add ColorBar
    def plot_train(self,raw, label, log_label, coarse, fine):
        self.axes[0].imshow(raw)
        self.axes[0].set_title("Raw")
        cax1 = self.axes[1].imshow(label)
        self.axes[1].set_title("Label")
        # self.fig.colorbar(cax1)
        cax2 = self.axes[2].imshow(log_label)
        self.axes[2].set_title("log(Label)")
        # self.fig.colorbar(cax2)
        cax3 = self.axes[3].imshow(coarse)
        self.axes[3].set_title("Coarse")
        # self.fig.colorbar(cax3)
        cax4 = self.axes[4].imshow(fine)
        self.axes[4].set_title("Fine")
        # self.fig.colorbar(cax4)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.pause(0.001)

    def showTestResults(self,raw, label, coarse, fine, i):
        plt.figure(1)
        # print(raw.shape)
        # print(label.shape)
        # print(coarse.shape)
        # print(fine.shape)

        self.axes[0].set_title("Raw")
        self.axes[0].imshow(raw)
        self.axes[1].set_title("Label")
        self.axes[1].imshow(label)
        self.axes[2].set_title("Coarse")
        self.axes[2].imshow(coarse)
        self.axes[3].set_title("Fine")
        self.axes[3].imshow(fine)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self.fig.canvas.set_window_title("Test Predictions [%d]" % i)

        plt.pause(0.001)

    @staticmethod
    def plotTrainingProgress(raw, label, log_label, coarse, fine, figId):
        fig = plt.figure(figId)
        fig.clf()

        # fig, axes = plt.subplots(5, 1)

        ax = fig.add_subplot(3, 2, 1)
        cax = ax.imshow(raw, cmap='gray')
        plt.title("Raw")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 3)
        cax = ax.imshow(label, cmap='gray')
        plt.title("Label")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 5)
        cax = ax.imshow(log_label, cmap='gray')
        plt.title("log(Label)")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 2)
        cax = ax.imshow(coarse, cmap='gray')
        plt.title("Coarse")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 4)
        cax = ax.imshow(fine, cmap='gray')
        plt.title("Fine")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        fig = plt.gcf() # TODO: Posso remover?
        fig.canvas.set_window_title("Train Predictions")

        plt.pause(0.001)

    @staticmethod
    # TODO: Add raw, log_labels, coarse
    def displayValidationProgress(label, fine, figId):
        fig = plt.figure(figId)
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        cax = ax.imshow(label, cmap='gray')
        plt.title("valid_labels[0,:,:]")
        fig.colorbar(cax)

        ax = fig.add_subplot(2, 1, 2)
        cax = ax.imshow(fine, cmap='gray')
        plt.title("vPredictions_f[0,:,:]")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    @staticmethod
    def plotTrainingErrorProgress(raw, label, coarse, fine, coarseMSE,fineMSE, figId):
        fig = plt.figure(figId)
        fig.clf()

        ax = fig.add_subplot(3, 2, 1)
        cax = ax.imshow(label, cmap='gray')
        plt.title("Label")
        fig.colorbar(cax)

        # TODO: Tentar resolver o problema que a imagem rgb não era correspondente com o label
        # ax = fig.add_subplot(3, 2, 2)
        # cax = ax.imshow(batch_data_crop_img)
        # plt.title("batch_data_crop[0,:,:]")
        # fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 3)
        cax = ax.imshow(coarse, cmap='gray')
        plt.title("Coarse")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 5)
        cax = ax.imshow(fine, cmap='gray')
        plt.title("Fine")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 4)
        cax = ax.imshow(coarseMSE, cmap='jet')
        plt.title("MSE(Coarse)")
        fig.colorbar(cax)

        ax = fig.add_subplot(3, 2, 6)
        cax = ax.imshow(fineMSE, cmap='jet')
        plt.title("MSE(Fine)")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        fig = plt.gcf() # TODO: Posso remover?
        fig.canvas.set_window_title("Train Predictions + MSE")

        plt.pause(0.001)