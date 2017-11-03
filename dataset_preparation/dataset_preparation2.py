#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =====================
#  Dataset Preparation
# =====================
# TODO: Verificar se disp_noc_0 e disp_noc_1 estao sincronizados com image_2 e image_3, respectivamente!
# TODO: Verificar se o código ainda funciona para nyu e kittiRaw_<scenes>

# ===========
#  Libraries
# ===========
import argparse
import os
import sys
import numpy as np
import glob

# ==================
#  Global Variables
# ==================
DATASET_PATH_ROOT = '/media/nicolas/Documentos/workspace/datasets'
# DATASET_PATH_ROOT = '/media/olorin/Documentos/datasets'

kitti_occlusion = True  # According to Kitti Description, occluded == All Pixels


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
              "'kitti<dataset_identification>' or 'nyudepth'",
              sep='')
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

        if kitti_occlusion:
            unused_train_folders_idx = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        else:
            unused_train_folders_idx = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11]

    if datasetObj.name == 'kitti2015':
        unused_test_folders_idx = []
        if kitti_occlusion:
            unused_train_folders_idx = [0, 1, 4, 5, 8, 9, 10]
        else:
            unused_train_folders_idx = [2, 3, 4, 5, 8, 9, 10]

    # TODO: Posso remover?
    # # if datasetObj.name[0:8] == 'kittiRaw':
    # #     unused_test_folders_idx = []
    # #     unused_train_folders_idx = []

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

        elif datasetObj.name[0:8] == 'kittiRaw':
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
            if i == 2 or i == 3:
                colors = colors + glob.glob(os.path.join(datasetObj.path, 'training', folders[i], '*.png'))
            if i == 0 or i == 1:
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
    valid_pairs = []

    if datasetObj.name == 'nyudepth':
        for i in range(0, len(colors_filename)):
            valid_pairs.append(depth_filename[i] in depth_filename)

    if datasetObj.name[0:5] == 'kitti':
        for i in range(len(colors_filename)):
            valid_pairs.append(colors_filename[i] in depth_filename)

    # Debug
    print(valid_pairs)
    print(len(valid_pairs))

    return valid_pairs

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
    # TODO: Posso remover o path da nomenclatura da variavel?
    # TODO: Posso tirar o files_ da nomenclatura da variavel?
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
    test_valid_pairs = getValidPairFiles(test_files_colors_filename, test_files_depth_filename, datasetObj)
    train_valid_pairs = getValidPairFiles(train_files_colors_filename, train_files_depth_filename, datasetObj)

    idx = [i for i, x in enumerate(test_valid_pairs) if x]
    idx2 = [i for i, x in enumerate(train_valid_pairs) if x]

    print('idx:', idx)
    print('idx_size:', len(idx), '\n')
    print('idx2:', idx2)
    print('idx2_size:', len(idx2), '\n')

# ======
#  Main
# ======
if __name__ == '__main__':
    main()
