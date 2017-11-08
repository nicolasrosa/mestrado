# ===========
#  Libraries
# ===========
import numpy as np
from six.moves import cPickle as pickle


# ===================
#  Class Declaration
# ===================
class DatasetHandler(object):
    def __init__(self, image_size=None, num_labels=None, num_channels=None):
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels

        print("\n[Dataset] Dataset obj. created.")

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

            del data    # hint to help gc free up memory

    # Reformat into a TensorFlow-friendly shape:
    # - convolutions need the image data formatted as a cube (width by height by #channels)
    # - labels as float 1-hot encodings.
    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size[0], self.image_size[1], self.num_channels))
        labels = (np.arange(self.num_labels) == labels[:, None])
        return dataset, labels

    def showDatasetInfo(self):
        print('Training set:', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set:', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set:', self.test_dataset.shape, self.test_labels.shape)

    def getImageSize(self):
        imageSize = (self.train_dataset.shape[1], self.train_dataset.shape[2])
        return imageSize

    def getDepthSize(self):
        depthSize = (self.train_labels.shape[1], self.train_labels.shape[2])
        return depthSize

    def getImageNumChannels(self):
        numChannels = self.train_dataset.shape[3]
        return numChannels
