# ===========
#  Libraries
# ===========
import argparse
import os
import sys
import time
import matplotlib.pyplot as plt

from .timer import Timer
from scipy import misc as scp

from classes.training import Loss
from classes.plots import Plot


# ===================
#  Class Declaration
# ===================
class Application(object):
    def __init__(self, enableProfilling):
        # App Config #
        self.appName = os.path.splitext(sys.argv[0])[0]
        self.datetime = '%s_%s' % (time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
        self.enableProfilling = enableProfilling

        # Timing #
        self.timer1 = Timer()
        self.timer2 = Timer()
        self.timer3 = Timer()

        # Arguments #
        self.args = self.createArgsParser()
        self.showSelectedOptions()

        # Folders and Files Paths #
        self.selectedDataset = os.path.splitext(os.path.basename(self.args.pickle_path))[0] # Dataset name without path and extention

        self.save_folder_path_tensorboard_files = '/tmp/tensorflow/%s/%s' % (self.appname, self.selectedDataset)
        self.save_folder_path_net_model = '%s/output/%s/%s/restore/' % (os.getcwd(),  self.selectedDataset, self.datetime)
        self.save_folder_path_profile = 'output/%s/%s/' % ( self.selectedDataset, self.datetime)
        self.save_filename_path_results = 'output/%s/results.txt' % ( self.selectedDataset)
        self.save_filename_path_net_model = '%s%s_model.ckpt' % (self.save_folder_path_net_model, self.appName)
        self.save_filename_path_profile = '%sprofile.tmp' % (self.save_folder_path_profile)

        self.savefig_folder_path_loss = 'output/%s/%s/' % ( self.selectedDataset, self.datetime)
        self.savefig_folder_path_img = 'output/%s/%s/testing/disp_pred/' % ( self.selectedDataset, self.datetime)
        self.savefig_folder_path_vPredictions_f = 'output/%s/%s/validation/disp_pred/' % ( self.selectedDataset, self.datetime)
        self.savefig_folder_path_valid_labels = 'output/%s/%s/validation/disp_pred/' % ( self.selectedDataset, self.datetime)
        self.savefig_filename_path_loss = '%s%s_loss.png' % (self.savefig_folder_path_loss, self.appName)

        self.saveplot_folder_path_img = 'output/%s/%s/testing/disp_plots/' % ( self.selectedDataset, self.datetime)

        self.restore_filename_path_net_model = "%s/%s_model.ckpt" % (self.args.restore_path, self.appName)

        
    # ==================
    #  Arguments Parser 
    # ==================
    def createArgsParser(self):
        # Creating Arguments Parser
        parser = argparse.ArgumentParser("Train the StereoCNN Network taking the dataset.pkl file as input.")
        
        # Input
        parser.add_argument('-i', '--dataset', action='store', dest='pickle_path', help="Set Relative Path to the dataset <filename>.pkl file")
        
        # Training
        parser.add_argument('-e', '--maxSteps', action='store', dest='maxSteps', help="Define the maximum number of training epochs.", type=int, default=1000)
        parser.add_argument('-l', '--learningRate', action='store', dest='learningRate', help="Define the Training Learning Rate.", type=float, default=1e-4)
        parser.add_argument('--showTrainingProgress', action='store_true', dest='showTrainingProgress', help="Show the first batch label and the correspondent Network predictions", default=False) # TODO: Add prefix char '-X'
        parser.add_argument('-t','--showTrainingErrorProgress', action='store_true', dest='showTrainingErrorProgress', help="Show the first batch label, the correspondent Network predictions and the MSE evaluations.", default=False)

        # Regularization Options
        parser.add_argument('-d', '--dropout', action='store_true', dest='dropout', help="Apply Dropout in the model during training.", default=False)
        parser.add_argument('--ldecay', action="store_true", dest="ldecay", help="Enable Learning Decay", default=False) # TODO: Add prefix char '-X'
        parser.add_argument('-n', '--l2norm', action="store_true", dest="l2norm", help="Enable L2 Normalization", default=False)
        parser.add_argument('-m', '--maskOut', action="store_true", dest="maskOut", help="Only consider valid pixels on calculations.", default=False)


        # Validation        
        parser.add_argument('--showValidationProgress', action='store_true', dest='showValidationProgress', help="Show the first validation example label and the Network predictions", default=False) # TODO: Add prefix char '-X'
        parser.add_argument('-v','--showValidationErrorProgress', action='store_true', dest='showValidationErrorProgress', help="Show the first validation example label, the Network predictions, and the MSE evaluations", default=False)
        
        # Testing
        parser.add_argument('-u', '--showTestingProgress', action='store_true', dest='showTestingProgress', help="Show the first batch testing Network prediction img", default=False)

        # Outputs
        parser.add_argument('-p', '--showPlots', action='store_true', dest='enablePlots', help="Allow the plots being displayed", default=False) # TODO: Correto seria falar o nome dos plots habilitados
        parser.add_argument('-s', '--save', action='store_true', dest='enableSave', help="Save the trained model for later restoration.", default=False)
        parser.add_argument('--saveValidFigs', action='store_true', dest='saveValidFigs', help="Save the figures from Validation Predictions.", default=False) # TODO: Add prefix char '-X'
        parser.add_argument('--saveTestPlots', action='store_true', dest='saveTestPlots', help="Save the Plots from Testing Predictions.", default=False)   # TODO: Add prefix char '-X'
        parser.add_argument('--saveTestFigs', action='store_true', dest='saveTestFigs', help="Save the figures from Testing Predictions", default=False)    # TODO: Add prefix char '-X'
        parser.add_argument('-b', '--tensorboard', action='store_true', dest='enableTensorBoard', help="Generate files for TensorBoard support.", default=False)
        
        # Restoration
        parser.add_argument('-r', '--restore', action='store', dest='restore_path', help="Restore an already trained model.")
        parser.add_argument('-z', '--restoreContinueTraining', action='store_true', dest='restoreContinueTraining', help="Restore an already trained model and continue training it.")
 
        return parser.parse_args()
    
    def showSelectedOptions(self):
        print("[App] Selected Options:\n",self.args)

    def createFoldersSaveFolder(self, path):
        if not os.path.exists(path):
            print("[App] Creating save folder: %s" % path)
            os.makedirs(path)

    def createFolders(self):
        print()
        
        # Creates Save Folders
        if self.enableProfilling:
            self.createFoldersSaveFolder(self.save_folder_path_profile)
        if self.args.enablePlots:
            self.createFoldersSaveFolder(self.savefig_folder_path_loss)
        # if self.args.enablePlots or self.args.enableSave:
        if self.args.enableSave:
            self.createFoldersSaveFolder(self.save_folder_path_net_model)
        if self.args.saveTestPlots:
            self.createFoldersSaveFolder(self.saveplot_folder_path_img)
        if self.args.saveTestFigs:
            self.createFoldersSaveFolder(self.savefig_folder_path_img)   
        if self.args.saveValidFigs:
            self.createFoldersSaveFolder(self.savefig_folder_path_vPredictions_f)

    @staticmethod
    def saveImg_asTxt(filename, img):
        file = open(filename,'w')

        for i in range(0, img.shape[0]):
            for j in range (0,img.shape[1]):
                # print(i,j, img[i,j])
                file.write('%s %s %s\n' % (i, j, img[i,j])) 

        file.close() 

    def saveValidFig(self, filename, prediction, label):
        for ext in ['png','txt']:
            savefig_filename_path_vPredictions_f = '%s%s.%s' % (self.savefig_folder_path_vPredictions_f, os.path.splitext(filename)[0],ext)
            savefig_filename_path_valid_labels = '%s%s_label.%s' % (self.savefig_folder_path_valid_labels, os.path.splitext(filename)[0],ext)

            print(savefig_filename_path_vPredictions_f)
            print(savefig_filename_path_valid_labels)
            
            if ext == 'png':
                # plt.imsave(savefig_filename_path_vPredictions_f, prediction, cmap='gray') # "TODO: Ele está normalizando, essa linha perde o range dos dados de disparidade. Salva arquivos entre 0 e 1, sendo que deveria ser 0 à 35000"
                scp.imsave(savefig_filename_path_vPredictions_f, prediction) # "TODO: Ele está normalizando, essa linha perde o range dos dados de disparidade. Salva arquivos entre 0 e 1, sendo que deveria ser 0 à 35000"
                scp.imsave(savefig_filename_path_valid_labels, label) # "TODO: Ele está normalizando, essa linha perde o range dos dados de disparidade. Salva arquivos entre 0 e 1, sendo que deveria ser 0 à 35000"
                
            if ext == 'txt':
                prediction_int32 = prediction.astype(dtype='int32')
                label_int32 = label.astype(dtype='int32')

                self.saveImg_asTxt(savefig_filename_path_vPredictions_f,prediction_int32)
                self.saveImg_asTxt(savefig_filename_path_valid_labels,label_int32)

    def saveTestPlot(self,filename):
        for ext in ['png','pdf','svg','ps']:
            saveplot_filename_path_img = '%s%s.%s' % (self.saveplot_folder_path_img,os.path.splitext(filename)[0], ext)
                                
            print(saveplot_filename_path_img)
            plt.savefig(saveplot_filename_path_img)

    def saveTestFig(self, filename, img):
        for ext in ['png','txt']:
            savefig_filename_path_img = '%s%s.%s' % (self.savefig_folder_path_img,os.path.splitext(filename)[0], ext)

            # Debug
            # print(dataset.list_test_colors_files_filename[k])
            # print(os.path.splitext(dataset.list_test_colors_files_filename[k])[0])
            # print(savefig_filename_path_img)
            # input("Press")

            if ext == 'png':
                # plt.imsave(savefig_filename_path_img,img, cmap='gray') # "TODO: Ele está normalizando, essa linha perde o range dos dados de disparidade. Salva arquivos entre 0 e 1, sendo que deveria ser 0 à 35000"
                scp.imsave(savefig_filename_path_img, img) # "TODO: Ele está normalizando, essa linha perde o range dos dados de disparidade. Salva arquivos entre 0 e 1, sendo que deveria ser 0 à 35000"

            if ext == 'txt':
                img_int32 =img.astype(dtype='int32')

                # Debug
                # print(img)
                # print(img.shape)
                # print(img.dtype)
                # input()

                # print(img_int32)
                # print(img_int32.shape)
                # print(img_int32.dtype)
                # input()
                
                file = open(savefig_filename_path_img,'w')
                for i in range(0,img_int32.shape[0]):
                    for j in range (0,img_int32.shape[1]):
                        # print(i,j, img_int32[i,j])
                        file.write('%s %s %s\n' % (i, j, img_int32[i,j])) 
                
                file.close()

    @staticmethod
    def showValidationErrorProgress(img_rgb, img_prediction, img_label):
        # TODO: Não é necessário passar o número de pixels, é possível calculá-lo a partir do tamanho das outras imagens
        img_MSE_f_img = Loss.np_MSE(img_label,img_prediction)

        Plot.displayValidationErrorProgress(img_rgb, img_prediction, img_label, img_MSE_f_img, 9)