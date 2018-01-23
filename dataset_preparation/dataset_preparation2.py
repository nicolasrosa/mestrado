#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =====================
#  Dataset Preparation
# =====================
# TODO: Verificar se o código ainda funciona para nyu e kittiraw_<scenes>

# ===========
#  Libraries
# ===========
import argparse
import os
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import misc as scp

# ==================
#  Global Variables
# ==================
# DATASET_PATH_ROOT = '/media/nicolas/Documentos/workspace/datasets'
DATASET_PATH_ROOT = '/media/olorin/Documentos/datasets'

SHOW_IMAGES = False
KITTI_OCCLUSION = True  # According to Kitti Description, occluded == All Pixels
TRAIN_VALID_RATIO = 0.8  # 80% for Training and 20% for Validation


class datasetKitti(object):
    def __init__(self, args, dataset_path):
        self.path = dataset_path
        self.name = args.dataset
        self.type = 'kitti'

        self.imageInputSize = [376, 1241]
        self.depthInputSize = [376, 1226]

        self.imageOutputSize = [172, 576]
        self.depthOutputSize = [43, 144]

        print("[App] datasetKitti object created.")


class datasetNyuDepth(object):
    def __init__(self, args, dataset_path):
        self.path = dataset_path
        self.name = args.dataset
        self.type = 'nyudepth'

        self.imageInputSize = [480, 640]
        self.depthInputSize = [480, 640]

        self.imageOutputSize = [228, 304]
        self.depthOutputSize = [57, 76]

        print("[App] datasetNyuDepth object created.")


# ===========
#  Functions
# ===========
def createArgsHandler():
    print("[App] Creating Argument Parser...")

    parser = argparse.ArgumentParser("Dumps dataset images into a dataset.pickle file.")
    parser.add_argument('-s', '--dataset', action='store', dest='dataset',
                        help="Selects the dataset ['kitti2012','kitti2015','nyudepth',kittiraw]")
    parser.add_argument('-i', '--showImageRGB', action="store_true", dest="showImageRGB",
                        help="Plots the Colored Input Images ", default=False)
    parser.add_argument('-d', '--showImageDepth', action="store_true", dest="showImageDepth",
                        help="Plots the Depth Input Images", default=False)
    parser.add_argument('-n', '--showNormed', action="store_true", dest="showNormedPlot",
                        help="Plots the Normed Input Images", default=False)
    parser.add_argument('-c', '--showCrop', action="store_true", dest="showCrop",
                        help="Plots the Cropped Input Images", default=False)
    parser.add_argument('-r', '--showDepthResized', action="store_true", dest="showDepthResized",
                        help="Plots the Resized Depth Images", default=False)

    args = parser.parse_args()

    return args


def checkArgumentsIntegrity(args):
    print("[App] Selected Options:")
    print(args)

    try:
        if args.dataset != 'nyudepth' and args.dataset[0:5] != 'kitti':
            raise ValueError

    except ValueError as e:
        print(e)
        print("[Error] ValueError: '", args.dataset,
              "' is not a valid name! Please select one of the following datasets: "
              "'kitti<dataset_identification>' or 'nyudepth'", sep='')
        print("e.g: python3 ", os.path.splitext(sys.argv[0])[0], ".py -s kitti2012", sep='')
        raise SystemExit


def selectedDataset(args):
    dataset_path = None

    if args.dataset[0:5] == 'kitti':  # If the first five letters are equal to 'kitti'
        if args.dataset == 'kitti2012':
            dataset_path = DATASET_PATH_ROOT + '/kitti/data_stereo_flow'

        elif args.dataset == 'kitti2015':
            dataset_path = DATASET_PATH_ROOT + '/kitti/data_scene_flow'

        elif args.dataset == 'kittiraw_city':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/city/2011_09_29_drive_0071'

        elif args.dataset == 'kittiraw_road':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/road/2011_10_03_drive_0042'

        elif args.dataset == 'kittiraw_residential':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/residential/2011_09_30_drive_0028'

        elif args.dataset == 'kittiraw_campus':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/campus/2011_09_28_drive_0039'
            
        elif args.dataset == 'kittiraw_residential_continuous':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/residential_continuous'

        # print(dataset_path)
        # input()
        kitti = datasetKitti(args, dataset_path)

        return kitti, dataset_path

    elif args.dataset == 'nyudepth':
        dataset_path = DATASET_PATH_ROOT + '/nyu-depth-v2/images'

        nyudepth = datasetNyuDepth(args, dataset_path)

        return nyudepth, dataset_path


def getListFolders(path):
    list = next(os.walk(path))[1]
    # print(list)
    return list


def removeUnusedFolders(test_folders, train_folders, datasetObj):
    print("[App] Removing unused folders for Kitti datasets...")
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
    print("Testing")
    print("colors:", colors)
    print(len(colors))
    print("depth:", depth)
    print(len(depth))
    print()

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
    print("Training")
    print("colors:", colors)
    print(len(colors))
    print("depth:", depth)
    print(len(depth))
    print()

    return colors, depth


def getFilesFilename(file_path_list):
    filename_list = []

    for i in range(0, len(file_path_list)):
        filename_list.append(os.path.split(file_path_list[i])[1])

    print(filename_list)
    print(len(filename_list))

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
    print("valid_pairs_idx:", valid_pairs_idx)
    print(len(valid_pairs_idx))

    return valid_pairs_idx


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()


def cropImage(img, x_min=None, x_max=None, y_min=None, y_max=None, size=None):
    try:
        if size is None:
            raise ValueError
    except ValueError:
        print("[ValueError] Oops! Empty cropSize list. Please sets the desired cropSize.\n")

    if len(img.shape) == 3:
        lx, ly, _ = img.shape
    else:
        lx, ly = img.shape

    # Debug
    # print("img.shape:", img.shape)
    # print("lx:",lx,"ly:",ly)

    if (x_min is None) and (x_max is None) and (y_min is None) and (y_max is None):
        # Crop
        # (y_min,x_min)----------(y_max,x_min)
        #       |                      |
        #       |                      |
        #       |                      |
        # (y_min,x_max)----------(y_max,x_max)
        x_min = round((lx - size[0]) / 2)
        x_max = round((lx + size[0]) / 2)
        y_min = round((ly - size[1]) / 2)
        y_max = round((ly + size[1]) / 2)

        crop = img[x_min: x_max, y_min: y_max]

        # Debug
        # print("x_min:",x_min,"x_max:", x_max, "y_min:",y_min,"y_max:", y_max)
        # print("crop.shape:",crop.shape)

        # TODO: Draw cropping Rectangle

    else:
        crop = img[x_min: x_max, y_min: y_max]

    return crop


def downsampleImage(img, size):
    try:
        if size is None:
            raise ValueError
    except ValueError:
        print("[ValueError] Oops! Empty resizeSize list. Please sets the desired resizeSize.\n")

    resized = scp.imresize(img, size, interp='bilinear')  # FIXME: ESSE METODO NAO MANTEM A ESCALA DE PROFUNDIDADE!!!

    return resized


def normalizeImage(img):
    pixel_depth = 255

    normed = (img - pixel_depth / 2) / pixel_depth

    # Debug
    # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
    # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

    return normed


def openImage(colors_path, depth_path, datasetObj):
    # Kitti RGB Image: (375, 1242, 3) uint8     #   NyuDepth RGB Image: (480, 640, 3) uint8 #
    #     Depth Image: (375, 1242)    int32     #          Depth Image: (480, 640)    int32 #

    img_colors = scp.imread(os.path.join(colors_path))
    img_depth = scp.imread(os.path.join(depth_path))

    # Debug
    # print(img_colors.shape, img_colors.dtype)
    # print(img_depth.shape, img_depth.dtype)
    # input()

    # Crops Image
    img_colors_crop = cropImage(img_colors, size=datasetObj.imageOutputSize)
    img_depth_crop = cropImage(img_depth, size=datasetObj.imageOutputSize)  # Same cropSize as the colors image

    # Normalizes RGB Image and Downsizes Depth Image
    img_colors_normed = normalizeImage(img_colors_crop)
    img_depth_downsized = downsampleImage(img_depth_crop, size=datasetObj.depthOutputSize)

    # Results
    if SHOW_IMAGES:
        def plot1():
            imshow(img_colors)
            imshow(img_depth)
            imshow(img_colors_crop)
            imshow(img_depth_crop)
            imshow(img_colors_normed)
            imshow(img_depth_downsized)

        def plot2():
            fig, axarr = plt.subplots(3, 2)
            ax1 = axarr[0, 0].imshow(img_colors)
            ax2 = axarr[0, 1].imshow(img_depth)
            ax3 = axarr[1, 0].imshow(img_colors_crop)
            ax4 = axarr[1, 1].imshow(img_depth_crop)
            ax5 = axarr[2, 0].imshow(img_colors_normed)
            ax6 = axarr[2, 1].imshow(img_depth_downsized)

        # plot1()
        plot2()

        plt.show()  # Display it

    return img_colors_crop, img_depth_crop, img_colors_normed, img_depth_downsized


# ======
#  Main
# ======
def main():
    # Creates Arguments Parser
    args = createArgsHandler()
    checkArgumentsIntegrity(args)

    print("\n[App] Description: This script prepares the ", args.dataset, ".pkl file for posterior Networks training.",
          sep='')

    # Creates Dataset Handler
    datasetObj, dataset_path = selectedDataset(args)

    # Gets the list of folders inside the 'DATASET_PATH' folder
    print('\n[App] Getting list of folders...')
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
    print("Num of   depth images found in '", os.path.split(dataset_path)[1], "/testing/*/: ",
          len(test_files_depth_path), sep='')
    print("Num of colored images found in '", os.path.split(dataset_path)[1], "/training/*/: ",
          len(train_files_colors_path), sep='')
    print("Num of   depth images found in '", os.path.split(dataset_path)[1], "/training/*/: ",
          len(train_files_depth_path), sep='')
    print()

    # Gets only the filename from the complete file path
    print("[App] Getting the filename files list...")
    test_files_colors_filename = getFilesFilename(test_files_colors_path)
    test_files_depth_filename = getFilesFilename(test_files_depth_path)
    train_files_colors_filename = getFilesFilename(train_files_colors_path)
    train_files_depth_filename = getFilesFilename(train_files_depth_path)

    # Checks which files have its depth/disparity correspondent
    print("\n[App] Checking which files have its depth/disparity correspondent...")
    test_valid_pairs_idx = getValidPairFiles(test_files_colors_filename, test_files_depth_filename, datasetObj)
    train_valid_pairs_idx = getValidPairFiles(train_files_colors_filename, train_files_depth_filename, datasetObj)

    # Creates the vectors what will be dumped as dict words in the pickle file.
    pkl_filename = args.dataset + '.pkl'

    # Original Images
    test_colors_crop, test_depth_crop = [], []
    train_colors_crop, train_depth_crop = [], []

    # Processed Images for Network Training
    test_dataset, test_labels = [], []
    train_dataset, train_labels = [], []

    # Generates the Network Inputs Images based on the original raw images
    if os.path.isfile(False):
    # if os.path.exists('output/' + pkl_filename):
        print("\n[App]", pkl_filename, 'already exists!')
    else:
        """Testing"""
        if len(test_valid_pairs_idx):  # Some test data doesn't have depth images
            for i, val in enumerate(test_valid_pairs_idx):
                colors_temp = test_files_colors_path[val[0]]
                depth_temp = test_files_depth_path[val[1]]

                print('i:', i, 'idx:', val, 'colors:', colors_temp, '\n\t\t\t\tdepth:', depth_temp)
                test_colors_crop_temp, test_depth_crop_temp, test_colors_normed_temp, test_depth_downsized_temp = openImage(
                    colors_temp, depth_temp, datasetObj)

                test_colors_crop.append(test_colors_crop_temp)
                test_depth_crop.append(test_depth_crop_temp)
                test_dataset.append(test_colors_normed_temp)
                test_labels.append(test_depth_downsized_temp)

        """Training"""
        for i, val in enumerate(train_valid_pairs_idx):
            colors_temp = train_files_colors_path[val[0]]
            depth_temp = train_files_depth_path[val[1]]

            # Test
            # if i == 195:
            #     global SHOW_IMAGES
            #     SHOW_IMAGES = True

            print('i:', i, 'idx:', val, 'colors:', colors_temp, '\n\t\t\t\tdepth:', depth_temp)
            train_colors_crop_temp, train_depth_crop_temp, train_colors_normed_temp, train_depth_downsized_temp = openImage(
                colors_temp,
                depth_temp,
                datasetObj)

            train_colors_crop.append(train_colors_crop_temp)
            train_depth_crop.append(train_depth_crop_temp)
            train_dataset.append(train_colors_normed_temp)
            train_labels.append(train_depth_downsized_temp)

        """Validation"""
        # Divides the Processed train data into training set and validation set
        print('\n[App] Dividing available data into training, validation and test sets...')
        trainSize = len(train_valid_pairs_idx)
        divider = int(TRAIN_VALID_RATIO * trainSize)

        valid_files_colors_filename = train_files_colors_filename[divider:]
        valid_files_depth_filename = train_files_depth_filename[divider:]
        valid_colors_crop = train_colors_crop[divider:]
        valid_depth_crop = train_depth_crop[divider:]
        valid_dataset = train_dataset[divider:]
        valid_labels = train_labels[divider:]

        train_files_colors_filename = train_files_colors_filename[:divider]
        train_files_depth_filename = train_files_depth_filename[:divider]
        train_colors_crop = train_colors_crop[:divider]
        train_depth_crop = train_depth_crop[:divider]
        train_dataset = train_dataset[:divider]
        train_labels = train_labels[:divider]

        """Final"""
        # TODO: Preciso realmente fazer isso?
        # Convert lists variables to np.array
        test_colors_crop = np.asarray(test_colors_crop)
        test_depth_crop = np.asarray(test_depth_crop)
        test_dataset = np.asarray(test_dataset)
        test_labels = np.asarray(test_labels)

        train_colors_crop = np.asarray(train_colors_crop)
        train_depth_crop = np.asarray(train_depth_crop)
        train_dataset = np.asarray(train_dataset)
        train_labels = np.asarray(train_labels)

        valid_colors_crop = np.asarray(valid_colors_crop)
        valid_depth_crop = np.asarray(valid_depth_crop)
        valid_dataset = np.asarray(valid_dataset)
        valid_labels = np.asarray(valid_labels)

        print("\nSummary")
        print("test_colors_crop len:", test_colors_crop.shape)
        print("test_depth_crop len:", test_depth_crop.shape)
        print("test_dataset shape:", test_dataset.shape)
        print("test_labels shape:", test_labels.shape)

        print("train_colors_crop len:", train_colors_crop.shape)
        print("train_depth_crop len:", train_depth_crop.shape)
        print("train_dataset shape:", train_dataset.shape)
        print("train_labels shape:", train_labels.shape)

        print("valid_colors_crop len:", valid_colors_crop.shape)
        print("valid_depth_crop len:", valid_depth_crop.shape)
        print("valid_dataset shape:", valid_dataset.shape)
        print("valid_labels shape:", valid_labels.shape)

        # Dumps data inside pickle file.
        print("\n[App] Dumping the processed images into the ", pkl_filename, " file...", sep='')
        # pkl_folder_path = 'output/'+appName+'/'
        pkl_folder_path = 'output/'

        if not os.path.exists(pkl_folder_path):
            os.makedirs(pkl_folder_path)

        pkl_file = open(pkl_folder_path + pkl_filename, 'wb')

        data = {'test_dataset_crop': test_colors_crop, 'test_labels_crop': test_depth_crop,
                'train_dataset_crop': train_colors_crop, 'train_labels_crop': train_depth_crop,
                'valid_dataset_crop': valid_colors_crop, 'valid_labels_crop': valid_depth_crop,

                'test_dataset': test_dataset, 'test_labels': test_labels,
                'train_dataset': train_dataset, 'train_labels': train_labels,
                'valid_dataset': valid_dataset, 'valid_labels': valid_labels,

                'train_colors_files_filename': train_files_colors_filename,
                'train_depth_files_filename': train_files_depth_filename,
                'valid_colors_files_filename': valid_files_colors_filename,
                'valid_depth_files_filename': valid_files_depth_filename,
                'test_colors_files_filename': test_files_colors_filename,
                'test_depth_files_filename': test_files_depth_filename}

        pkl.dump(data, pkl_file, pkl.HIGHEST_PROTOCOL)
        pkl_file.close()

    print("[App] Done.")


# ======
#  Main
# ======
if __name__ == '__main__':
    main()
