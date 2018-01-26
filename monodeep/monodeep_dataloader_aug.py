# ===========
#  Libraries
# ===========
from __future__ import absolute_import, division, print_function
from scipy.misc import imread, imshow

import tensorflow as tf
import os
import sys
import glob
import numpy as np


# from temp.datasetAugmentation.dataset_preparation2 import *

# ==================
#  Global Variables
# ==================
# DATASET_PATH_ROOT = '/media/nicolas/Documentos/workspace/datasets'
# DATASET_PATH_ROOT = '/media/olorin/Documentos/datasets'

# SHOW_IMAGES = False
KITTI_OCCLUSION = True  # According to Kitti Description, occluded == All Pixels
TRAIN_VALID_RATIO = 0.8  # 80% for Training and 20% for Validation

# ===========
#  Functions
# ===========
def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

def checkArgumentsIntegrity(dataset):
    print("[monodeep/Dataloader] Selected Dataset:", dataset)

    try:
        if dataset != 'nyudepth' and dataset[0:5] != 'kitti':
            raise ValueError

    except ValueError as e:
        print(e)
        print("[Error] ValueError: '", dataset,
              "' is not a valid name! Please select one of the following datasets: "
              "'kitti<dataset_identification>' or 'nyudepth'", sep='')
        print("e.g: python3 ", os.path.splitext(sys.argv[0])[0], ".py -s kitti2012", sep='')
        raise SystemExit


def selectedDataset(DATASET_PATH_ROOT, dataset):
    dataset_path = None

    if dataset[0:5] == 'kitti':  # If the first five letters are equal to 'kitti'
        if dataset == 'kitti2012':
            dataset_path = DATASET_PATH_ROOT + 'kitti/data_stereo_flow'

        elif dataset == 'kitti2015':
            dataset_path = DATASET_PATH_ROOT + 'kitti/data_scene_flow'

        elif dataset == 'kittiraw_city':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/city/2011_09_29_drive_0071'

        elif dataset == 'kittiraw_road':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/road/2011_10_03_drive_0042'

        elif dataset == 'kittiraw_residential':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/residential/2011_09_30_drive_0028'

        elif dataset == 'kittiraw_campus':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/campus/2011_09_28_drive_0039'

        elif dataset == 'kittiraw_residential_continuous':
            dataset_path = DATASET_PATH_ROOT + 'nicolas_kitti/dataset1/residential_continuous'

        # print(dataset_path)
        # input()
        kitti = datasetKitti(dataset, dataset_path)

        return kitti, dataset_path

    elif dataset == 'nyudepth':
        dataset_path = DATASET_PATH_ROOT + '/nyu-depth-v2/images'

        nyudepth = datasetNyuDepth(dataset, dataset_path)

        return nyudepth, dataset_path

def getListFolders(path):
    list = next(os.walk(path))[1]
    # print(list)
    return list


def removeUnusedFolders(test_folders, train_folders, datasetObj):
    print("[monodeep/Dataloader] Removing unused folders for Kitti datasets...")
    unused_test_folders_idx = []
    unused_train_folders_idx = []

    if datasetObj.name == 'kitti2012':
        unused_test_folders_idx = [0, 3, 4]

        if KITTI_OCCLUSION:
            unused_train_folders_idx = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        else:
            unused_train_folders_idx = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11]

    if datasetObj.name == 'kitti2015':
        unused_test_folders_idx = []
        if KITTI_OCCLUSION:
            unused_train_folders_idx = [0, 1, 3, 4, 5, 7, 8, 9, 10]
        else:
            unused_train_folders_idx = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    if datasetObj.name[0:8] == 'kittiraw':
        unused_test_folders_idx = []
        unused_train_folders_idx = []

    test_folders = np.delete(test_folders, unused_test_folders_idx).tolist()
    train_folders = np.delete(train_folders, unused_train_folders_idx).tolist()

    # Debug
    print(test_folders)
    print(train_folders)
    print()

    return test_folders, train_folders


def getListTestFiles(folders, datasetObj):
    colors, depth = [], []

    for i in range(len(folders)):
        if datasetObj.name == 'nyudepth':
            colors = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*_colors.png'))
            depth = depth + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*_depth.png'))

        elif datasetObj.name == 'kitti2012' or datasetObj.name == 'kitti2015':
            colors = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*.png'))
            depth = []

        elif datasetObj.name[0:8] == 'kittiraw':
            if i == 1:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[i], '*.png'))
            if i == 0:
                depth = colors + glob.glob(os.path.join(datasetObj.path, 'testing', folders[0], '*.png'))

    # Debug
    # print("Testing")
    # print("colors:", colors)
    # print(len(colors))
    # print("depth:", depth)
    # print(len(depth))
    # print()

    return colors, depth


def getListTrainFiles(folders, datasetObj):
    colors, depth = [], []

    for i in range(len(folders)):
        if datasetObj.name == 'nyudepth':
            colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*_colors.png'))
            depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*_depth.png'))

        elif datasetObj.name == 'kitti2012':
            if i == 0:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 1:
                depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))

        elif datasetObj.name == 'kitti2015':
            if i == 1:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 0:
                depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))

        elif datasetObj.name[0:8] == 'kittiraw':
            if i == 1:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 0:
                depth = depth + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))

    # Debug
    # print("Training")
    # print("colors:", colors)
    # print(len(colors))
    # print("depth:", depth)
    # print(len(depth))
    # print()

    return colors, depth

def getFilesFilename(file_path_list):
    filename_list = []

    for i in range(0, len(file_path_list)):
        filename_list.append(os.path.split(file_path_list[i])[1])

    # print(filename_list)
    # print(len(filename_list))

    return filename_list

def getValidPairFiles(colors_filename, depth_filename, datasetObj):
    valid_pairs_idx = []

    colors_filename_short = []
    depth_filename_short = []

    if datasetObj.name == 'nyudepth':
        for i in range(len(colors_filename)):
            # print(colors_filename[i])
            # print(depth_filename[i])
            # print(colors_filename[i][:-11])
            # print(depth_filename[i][:-10])

            for k in range(len(colors_filename)):
                colors_filename_short.append(colors_filename[k][:-11])

            for l in range(len(depth_filename)):
                depth_filename_short.append(depth_filename[l][:-10])

            if colors_filename_short[i] in depth_filename_short:
                j = depth_filename_short.index(colors_filename_short[i])
                # print(i,j)
                valid_pairs_idx.append([i, j])

    if datasetObj.name[0:5] == 'kitti':
        for i in range(len(colors_filename)):
            if colors_filename[i] in depth_filename:
                j = depth_filename.index(colors_filename[i])

                valid_pairs_idx.append([i, j])

    # Debug
    # print("valid_pairs_idx:", valid_pairs_idx)
    # print(len(valid_pairs_idx))

    return valid_pairs_idx

# ===================
#  Class Declaration
# ===================
class datasetKitti(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'kitti'

        self.imageInputSize = [376, 1241]
        self.depthInputSize = [376, 1226]

        self.imageOutputSize = [172, 576]
        self.depthOutputSize = [43, 144]

        print("[monodeep/Dataloader] datasetKitti object created.")


class datasetNyuDepth(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'nyudepth'

        self.imageInputSize = [480, 640]
        self.depthInputSize = [480, 640]

        self.imageOutputSize = [228, 304]
        self.depthOutputSize = [57, 76]

        print("[monodeep/Dataloader] datasetNyuDepth object created.")

class MonodepthDataloader_new(object): # TODO: Assim q tudo estiver funcionando, Mudar nome
    """monodepth dataloader"""

    def __init__(self, data_path, params, dataset, mode):

        checkArgumentsIntegrity(dataset)
        print("\n[monodeep/Dataloader] Description: This script prepares the ", dataset,
              ".pkl file for posterior Networks training.",
              sep='')

        # Creates Dataset Handler
        datasetObj, dataset_path = selectedDataset(data_path, dataset)

        # Gets the list of folders inside the 'DATASET_PATH' folder
        print('\n[monodeep/Dataloader] Getting list of folders...')
        test_folders = getListFolders(os.path.join(dataset_path, 'testing'))
        train_folders = getListFolders(os.path.join(dataset_path, 'training'))

        print("Num of folders inside '", os.path.split(dataset_path)[1], "/testing': ", len(test_folders), sep='')
        print("Num of folders inside '", os.path.split(dataset_path)[1], "/training': ", len(train_folders), sep='')
        print(test_folders, '\n', train_folders, '\n', sep='')

        # Removing unused folders for Kitti datasets
        if datasetObj.type == 'kitti':
            test_folders, train_folders = removeUnusedFolders(test_folders, train_folders, datasetObj)

        # Gets the list of files inside each folder grouping *_colors.png and *_depth.png files.
        test_files_colors_path, test_files_depth_path = getListTestFiles(test_folders, datasetObj)
        train_files_colors_path, train_files_depth_path = getListTrainFiles(train_folders, datasetObj)

        print("Summary")
        print("Num of colored images found in '", os.path.split(dataset_path)[1], "/testing/*/: ",
              len(test_files_colors_path), sep='')
        print("Num of   train_depth_filepath images found in '", os.path.split(dataset_path)[1], "/testing/*/: ",
              len(test_files_depth_path), sep='')
        print("Num of colored images found in '", os.path.split(dataset_path)[1], "/training/*/: ",
              len(train_files_colors_path), sep='')
        print("Num of   train_depth_filepath images found in '", os.path.split(dataset_path)[1], "/training/*/: ",
              len(train_files_depth_path), sep='')
        print()

        # Gets only the filename from the complete file path
        print("[monodeep/Dataloader] Getting the filename files list...")
        test_files_colors_filename = getFilesFilename(test_files_colors_path)
        test_files_depth_filename = getFilesFilename(test_files_depth_path)
        train_files_colors_filename = getFilesFilename(train_files_colors_path)
        train_files_depth_filename = getFilesFilename(train_files_depth_path)

        # Checks which files have its train_depth_filepath/disparity correspondent
        print("\n[monodeep/Dataloader] Checking which files have its train_depth_filepath/disparity correspondent...")
        test_valid_pairs_idx = getValidPairFiles(test_files_colors_filename, test_files_depth_filename, datasetObj)
        train_valid_pairs_idx = getValidPairFiles(train_files_colors_filename, train_files_depth_filename, datasetObj)

        # Original Images
        # test_colors_crop, test_depth_crop = [], []
        # train_colors_crop, train_depth_crop = [], []

        # Processed Images for Network Training
        # test_dataset, test_labels = [], []
        # train_dataset, train_labels = [], []

        """Testing"""
        test_colors_filepath, test_depth_filepath = [], []
        if len(test_valid_pairs_idx):  # Some test data doesn't have depth images
            for i, val in enumerate(test_valid_pairs_idx):
                test_colors_filepath.append(test_files_colors_path[val[0]])
                test_depth_filepath.append(test_files_depth_path[val[1]])

                # print('i:', i, 'idx:', val, 'colors:', test_colors_filepath[i], '\n\t\t\t\tdepth:', test_depth_filepath[i])
        else:
            test_colors_filepath = test_files_colors_path
            test_depth_filepath = []

        self.test_dataset = test_colors_filepath
        self.test_labels = test_depth_filepath

        # Divides the Processed train data into training set and validation set
        print('\n[monodeep/Dataloader] Dividing available data into training, validation and test sets...')
        trainSize = len(train_valid_pairs_idx)
        divider = int(TRAIN_VALID_RATIO * trainSize)

        """Training"""
        train_colors_filepath, train_depth_filepath = [], []
        for i, val in enumerate(train_valid_pairs_idx):
            train_colors_filepath.append(train_files_colors_path[val[0]])
            train_depth_filepath.append(train_files_depth_path[val[1]])

            # print('i:', i, 'idx:', val, 'train_colors_filepath:', train_colors_filepath[i], '\n\t\t\ttrain_depth_filepath:', train_depth_filepath[i])

        self.train_dataset = train_colors_filepath[:divider]
        self.train_labels = train_depth_filepath[:divider]

        """Validation"""
        self.valid_dataset = train_colors_filepath[divider:]
        self.valid_labels = train_depth_filepath[divider:]

        """Final"""
        print("\nSummary")
        print("train_dataset shape:", len(self.train_dataset))
        print("train_labels shape:", len(self.train_labels))
        print("valid_dataset shape:", len(self.valid_dataset))
        print("valid_labels shape:", len(self.valid_labels))
        print("test_dataset shape:", len(self.test_dataset))
        print("test_labels shape:", len(self.test_labels))

        # TODO: Aprimorar
        f = open('kitti_train_filenames.txt', 'w')
        for i in range(len(self.train_dataset)):
            f.write("%s %s\n" % (self.train_dataset[i], self.train_labels[i]))
        f.close()

        f = open('kitti_valid_filenames.txt', 'w')
        for i in range(len(self.valid_dataset)):
            f.write("%s %s\n" % (self.valid_dataset[i], self.valid_labels[i]))
        f.close()

        f = open('kitti_test_filenames.txt', 'w')
        for i in range(len(self.test_dataset)):
            if len(self.test_labels):
                f.write("%s %s\n" % (self.test_dataset[i], self.test_labels[i]))
            else:
                f.write("%s\n" % (self.test_dataset[i]))
        f.close()

        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.image_batch = None

        input("oi")

        ################################
        input_queue = tf.train.string_input_producer(['kitti_train_filenames.txt'], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        a = imread(self.train_dataset[0])
        imshow(a)
        input("oi")

        image_path = split_line[0]
        image_o = self.read_image(image_path)
        depth_path = split_line[1]
        depth_o = self.read_image(depth_path)

        print(input_queue)
        print(line_reader)
        print(line)
        print(split_line)
        print(image_path)
        print(len(image_path))
        input("Continue3...")

        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: image_o)

            sess = tf.Session()
            # print(do_flip.eval(session=sess))
            input("Continue2...")

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(image), lambda: (image))

            image.set_shape( [None, None, 3])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.image_batch = tf.train.shuffle_batch([image],
                                                      params.batch_size, capacity, min_after_dequeue, params.num_threads)

        elif mode == 'test':
            self.image_batch = tf.stack([image_o, tf.image.flip_left_right(image_o)], 0)
            self.image_batch.set_shape([2, None, None, 3])

    def augment_image_pair(self, image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        image_aug  = image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_aug  =  image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        image_aug  *= color_image

        # saturate
        image_aug  = tf.clip_by_value(image_aug,  0, 1)

        return image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]

        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))


        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]



        image  = tf.image.convert_image_dtype(image,  tf.float32)
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        input("continue5...")


        return image
