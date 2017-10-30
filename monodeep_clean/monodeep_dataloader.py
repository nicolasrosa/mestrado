# ===========
#  Libraries
# ===========
import numpy as np
from six.moves import cPickle as pickle


# ===================
#  Class Declaration
# ===================
class MonoDeepDataloader(object):
    def __init__(self, params, mode, data_path):
        self.params = params
        self.mode = mode
        self.data_path = data_path
        self.inputSize = -1
        self.outputSize = -1

        # print(self.params)
        # print(self.mode)
        # print(self.data_path)

        self.load(self.data_path)
        self.showDatasetInfo()

    def load(self, filename):
        print("[Dataset] Loading Dataset...")
        with open(filename, 'rb') as file:
            data = pickle.load(file)

            # Images filenames
            self.list_train_colors_files_filename = data['list_train_colors_files_filename']
            self.list_train_depth_files_filename = data['list_train_depth_files_filename']
            self.list_valid_colors_files_filename = data['list_valid_colors_files_filename']
            self.list_valid_depth_files_filename = data['list_valid_depth_files_filename']
            self.list_test_colors_files_filename = data['list_test_colors_files_filename']
            self.list_test_depth_files_filename = data['list_test_depth_files_filename'] 

            # Original Images - Cropped
            self.train_dataset_crop = data['train_dataset_crop']
            self.train_labels_crop = data['train_labels_crop']
            self.valid_dataset_crop = data['valid_dataset_crop']
            self.valid_labels_crop = data['valid_labels_crop']
            self.test_dataset_crop = data['test_dataset_crop']
            self.test_labels_crop = data['test_labels_crop']

            # Processed Images
            self.train_dataset = data['train_dataset']
            self.train_labels = data['train_labels']
            self.valid_dataset = data['valid_dataset']
            self.valid_labels = data['valid_labels']
            self.test_dataset = data['test_dataset']
            self.test_labels = data['test_labels']

            self.inputSize = self.train_dataset.shape
            self.outputSize = self.train_labels.shape

            # print(self.inputSize)
            # print(self.outputSize)

            print("[Dataset] Loading Dataset...DONE!")

    def showDatasetInfo(self):
        print("\nDataset Summary")
        print("\t\t\tInputs\t\t     Labels")
        print('Training set:\t', self.train_dataset.shape,'\t', self.train_labels.shape)
        print('Validation set:\t', self.valid_dataset.shape,'\t', self.valid_labels.shape)
        print('Test set:\t', self.test_dataset.shape,'\t', self.test_labels.shape)

    def getImageSize(self):
        imageSize = (self.train_dataset.shape[1], self.train_dataset.shape[2])
        return imageSize

    def getDepthSize(self):
        depthSize = (self.train_labels.shape[1], self.train_labels.shape[2])
        return depthSize

    def getImageNumChannels(self):
        numChannels = self.train_dataset.shape[3]
        return numChannels
