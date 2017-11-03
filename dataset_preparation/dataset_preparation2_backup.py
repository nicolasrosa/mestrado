#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =====================
#  Dataset Preparation
# =====================
# TODO: Arrumar o bug de MemoryError que aconteceu para o kittiRaw_residential 

# ===========
#  Libraries
# ===========
from scipy import misc as scp

import argparse
import glob
import os
import os.path
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import six.moves.cPickle as pickle

from skimage.measure import block_reduce
from sys import getsizeof

np.set_printoptions(threshold=np.nan)

# ==================
#  Global Variables
# ==================
appName = os.path.splitext(sys.argv[0])[0]

DATASET_PATH_ROOT = '/media/nicolas/Documentos/workspace/datasets'
# DATASET_PATH_ROOT = '/media/olorin/Documentos/datasets'

SHOW_FILENAMES_LISTS = True
TRAIN_VALID_RATIO = 0.8  # 80% for Training and 20% for Validation

# ===========
#  Functions
# ===========
def plotImageRGB(imgRGB):
    plt.figure()
    gs = gridspec.GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.title("Image")
    plt.imshow(imgRGB)
    plt.colorbar()
    plt.subplot(gs[1, 0])
    plt.imshow(imgRGB[:, :, 0])
    plt.title("Channel 0")
    plt.subplot(gs[1, 1])
    plt.imshow(imgRGB[:, :, 1])
    plt.title("Channel 1")
    plt.subplot(gs[1, 2])
    plt.imshow(imgRGB[:, :, 2])
    plt.title("Channel 2")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.draw()
    # plt.pause(1)
    # plt.close()

    # plt.show()


def plotImageDepth(imgDepth):
    plt.figure()
    plt.imshow(imgDepth, cmap='gray')
    plt.colorbar()
    plt.title("Depth")

    # plt.show()

def plotResizedImageDepth(depth,resized):
    plt.figure()
    plt.subplot(211)
    plt.imshow(depth, cmap='gray')
    plt.colorbar()
    plt.title("Cropped Depth")

    plt.subplot(212)
    plt.imshow(resized, cmap='gray')
    plt.colorbar()
    plt.title("Resized")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()

def plotResizedImageDepth(depth, resized):
    plt.figure()
    plt.subplot(211)
    plt.imshow(depth, cmap='gray')
    plt.colorbar()
    plt.title("Depth")

    plt.subplot(212)
    plt.imshow(resized, cmap='gray')
    plt.colorbar()
    plt.title("Resized")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()


def plotImageCrop(img, crop):
    plt.figure()
    plt.subplot(211)
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title("Raw Image")
    plt.subplot(212)
    if len(img.shape) == 3:
        plt.imshow(crop)
    else:
        plt.imshow(crop, cmap='gray')
    plt.title("Cropped Image")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()


def plotHist(img):
    plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.draw()
    # plt.pause(1)
    # plt.close()

    # plt.show()


def normalizeImage(img):
    pixel_depth = 255

    normed = (img - pixel_depth / 2) / pixel_depth

    # Debug
    # print("img[0,0,0]:", img[0, 0, 0], "img[0,0,1]:", img[0, 0, 1], "img[0,0,2]:", img[0, 0, 2])
    # print("normed[0,0,0]:", normed[0, 0, 0], "normed[0,0,1]:", normed[0, 0, 1], "normed[0,0,2]:", normed[0, 0, 2])

    return normed


def openImage(filename, showContent=False, showImagePlot=False, showDepthPlot=False, showNormedPlot=False,
             showHist=False, showCrop=False, showDepthResizedPlot=False, cropSize=None, resizeSize=None):

    normed = None
    resized = None
    
    img = scp.imread(os.path.join(filename)) 

    # Debug
    # Kitti RGB Image: (375, 1242, 3) uint8
    # Depth Image: (375, 1242) int32
    # print(img.shape, img.dtype)
    # input()

    # Crops Image
    # TODO: Dar suporte a data augmentation. Conseguir gerar novas imagens a partir da imagem lida.
    crop = cropImage(img, size=cropSize)

    if len(img.shape) == 2:
        resized = downsampleImage(crop, size=resizeSize)

    # Normalizes Image
    if len(img.shape) == 3:
        normed = normalizeImage(crop)

    # Show Plots
    if showImagePlot and (img.shape[2] if len(img.shape) == 3 else 1 == 3):  # numChannels
        plotImageRGB(imgRGB=img)

    if showDepthPlot:
        plotImageDepth(imgDepth=img)

    if (showImagePlot and showCrop) or (showDepthPlot and showCrop):
        plotImageCrop(img, crop)

    if showImagePlot and (img.shape[2] if len(img.shape) == 3 else 1 == 3) and showNormedPlot:
        plotImageRGB(imgRGB=normed)

    if showHist:
        plotHist(img)

    if showDepthResizedPlot:
        plotResizedImageDepth(crop, resized)

    if showImagePlot or showDepthPlot or showCrop or showHist or showDepthResizedPlot:
        plt.show()

    if showContent:
        print(filename, ': ', img, sep='')

    if len(img.shape) == 3:
        return crop, normed
    else:
        # return crop
        return crop, resized


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

    # TODO: Esse é o melhor método?
    resized = block_reduce(img, block_size=(4, 4), func=np.max)

    # Debug
    # print(img.shape)
    # print(resized.shape)
    
    return resized

# TODO: Acho que é possível criar listar de imagens, sem precisar alocar matrizes com o numpy
def make_arrays(nFiles, imageSize=None, depthImage=None):
    try:
        if imageSize is None or depthImage is None:
            raise ValueError
    except ValueError:
        print("[ValueError] Oops! Empty imageSize list or depthImage. Please sets the desired list size.\n")

    nChannels = 3

    if nFiles:
        dataset = np.ndarray((nFiles, imageSize[0], imageSize[1], nChannels),
                             dtype=np.float32)  # (height,width,nChannels)
        labels = np.ndarray((nFiles, depthImage[0], depthImage[1]), dtype=np.int32)  # (height,width)
    else:
        dataset, labels = None, None

    return dataset, labels

# Prints only the folders
def printListFolders(path):
    print(next(os.walk(path))[1])

# Prints only the files
def printListFiles(path):
    print(next(os.walk(path))[2])

def getListFolders(path):
    return next(os.walk(path))[1]

def getListFiles(path):
    return next(os.walk(path))[2]

def getListAllFiles(path):
    if SHOW_FILENAMES_LISTS:
        print("There are ", len(os.listdir(path)), " available folders/files in '", path, "':\n\n", list, "\n", sep='')

    # printListFolders(path)
    # printListFiles(path)

    return os.listdir(path)

def checkArgumentsIntegrity(args):
    print("Selected Options:")
    print(args)

    try:
        if args.dataset != 'nyuDepth' and args.dataset[0:5] != 'kitti' :
            raise ValueError

        return False
    except ValueError as e:
        print(e)
        print("ValueError: '", args.dataset,
              "' is not a valid name! Please select one of the following datasets: 'kitti2012', "
              "'kitti2015' or 'nyuDepth'", sep='')
        print("e.g: python3 ", os.path.splitext(sys.argv[0])[0], ".py -s kitti2012", sep='')
        return True


def createArgsHandler():
    parser = argparse.ArgumentParser("Dumps datasets images into a dataset.pickle file.")
    parser.add_argument('-s', '--dataset', action='store', dest='dataset',
                        help="Selects the dataset ['kitti2012','kitti2015','nyuDepth',kittiRaw]")
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

def selectedDataset(args):
    if (args.dataset[0:5] == 'kitti'):  # If the first five letters are equal to 'kitti'
        # TODO: Tamanhos utilizados para conseguir as raw images
        imageInputSize = [376, 1241]
        depthInputSize = [376, 1226]

        imageOutputSize = [172, 576]
        depthOutputSize = [43, 144]

        if args.dataset == 'kitti2012':
            dataset_path = DATASET_PATH_ROOT + '/kitti/data_stereo_flow'

        elif args.dataset == 'kitti2015':
            dataset_path = DATASET_PATH_ROOT + '/kitti/data_scene_flow'

        elif args.dataset == 'kittiRaw_city':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/city/2011_09_29_drive_0071'

        elif args.dataset == 'kittiRaw_road':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/road/2011_10_03_drive_0042'

        elif args.dataset == 'kittiRaw_residential':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/residential/2011_09_30_drive_0028'

        elif args.dataset == 'kittiRaw_campus':
            dataset_path = DATASET_PATH_ROOT + '/nicolas_kitti/dataset1/campus/2011_09_28_drive_0039'

        elif args.dataset == 'nyuDepth':
            DATASET_PATH = DATASET_PATH_ROOT + '/nyu-depth-v2/images'

            # TODO: Tamanhos utilizados para conseguir as raw images
            imageInputSize = []
            depthInputSize = []

            imageOutputSize = [228, 304]
            depthOutputSize = [57, 76]

        return imageOutputSize, depthOutputSize, dataset_path

# ======
#  Main
# ======
def main():
    # Local Variables
    train_valid_pair_files = []
    test_valid_pair_files = []

    # Creating Arguments Parser
    args = createArgsHandler()

    if checkArgumentsIntegrity(args):
        return 0

    print("This script prepares the ", args.dataset, ".pkl file for posterior Networks training.\n", sep='')

    # Creates Dataset Handler
    imageOutputSize, depthOutputSize, DATASET_PATH = selectedDataset(args)

    # Gets the list of folders inside the 'DATASET_PATH' folder
    list_test_folders = getListFolders(os.path.join(DATASET_PATH, 'testing'))
    list_train_folders = getListFolders(os.path.join(DATASET_PATH, 'training'))

    print("Summary")
    print("Num of folders inside '", os.path.split(DATASET_PATH)[1], "/testing': ", len(list_test_folders), sep='')
    print("Num of folders inside '", os.path.split(DATASET_PATH)[1], "/training': ", len(list_train_folders), sep='')

    # Debug
    print()
    print(list_test_folders)
    print()
    print(list_train_folders)
    print()

    # Removing unused folders for Kitti datasets
    print("Removing unused folders for Kitti datasets...")
    if  args.dataset[0:5] == 'kitti':
        if args.dataset == 'kitti2012':
            unused_test_folders_idx = [0, 3, 4]
            unused_train_folders_idx = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]

        if args.dataset == 'kitti2015':
            unused_test_folders_idx = []
            unused_train_folders_idx = [0, 1, 4, 5, 8, 9, 10]

        if args.dataset[0:8] == 'kittiRaw':
            unused_test_folders_idx = []
            unused_train_folders_idx = []


        list_test_folders = np.delete(list_test_folders, unused_test_folders_idx).tolist()
        list_train_folders = np.delete(list_train_folders, unused_train_folders_idx).tolist()   

    # Debug
    print()
    print(list_test_folders)
    print()
    print(list_train_folders)
    print()

    # Gets the list of files inside each folder grouping *_colors.png and *_depth.png files.
    list_test_colors_files_path = []
    list_test_depth_files_path = []
    list_train_colors_files_path = []
    list_train_depth_files_path = []

    for i in range(0, len(list_test_folders)):
        if args.dataset == 'nyuDepth':
            list_test_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'testing', list_test_folders[i],'*_colors.png')) + list_test_colors_files_path
            list_test_depth_files_path = glob.glob(os.path.join(DATASET_PATH, 'testing', list_test_folders[i], '*_depth.png')) + list_test_depth_files_path
        
        elif args.dataset == 'kitti2012' or args.dataset == 'kitti2015':
            list_test_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'testing', list_test_folders[i], '*.png')) + list_test_colors_files_path
            list_test_depth_files_path = []

        elif args.dataset[0:8] == 'kittiRaw':
            if i==1:
                list_test_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'testing', list_test_folders[i], '*.png')) + list_test_colors_files_path
            if i==0:
                list_test_depth_files_path = glob.glob(os.path.join(DATASET_PATH, 'testing', list_test_folders[0], '*.png')) + list_test_colors_files_path

    # Debug
    # print(list_test_colors_files_path)
    # print(len(list_test_colors_files_path))
    # print()
    # print(list_test_depth_files_path)
    # print(len(list_test_depth_files_path))
    # print()
    # input("Press")

    for i in range(0, len(list_train_folders)):
        if args.dataset == 'nyuDepth':
            list_train_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*_colors.png')) + list_train_colors_files_path
            list_train_depth_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*_depth.png')) + list_train_depth_files_path

        elif args.dataset == 'kitti2012':
            if i == 0:
                list_train_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*.png')) + list_train_colors_files_path
            if i == 1:
                list_train_depth_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i], '*.png')) + list_train_depth_files_path

        elif args.dataset == 'kitti2015':
            if i == 2 or i == 3:
                list_train_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*.png')) + list_train_colors_files_path
            if i == 0 or i == 1:
                list_train_depth_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*.png')) + list_train_depth_files_path

        elif args.dataset[0:8] == 'kittiRaw':
            if i == 1:
                list_train_colors_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*.png')) + list_train_colors_files_path
            if i == 0:
                list_train_depth_files_path = glob.glob(os.path.join(DATASET_PATH, 'training', list_train_folders[i],'*.png')) + list_train_depth_files_path


    # print(list_train_files_colors)
    # print(len(list_train_files_colors))
    # print()
    # print(list_train_files_depth)
    # print(len(list_train_files_depth))
    # print()

    print("Summary")
    print("Num of *_colors.png images found in '", os.path.split(DATASET_PATH)[1], "/testing/*/: ",
          len(list_test_colors_files_path), sep='')
    print("Num of *_depth.png images found in '", os.path.split(DATASET_PATH)[1], "/testing/*/: ",
          len(list_test_depth_files_path), sep='')
    print("Num of *_colors.png images found in '", os.path.split(DATASET_PATH)[1], "training/*/: ",
          len(list_train_colors_files_path), sep='')
    print("Num of *_depth.png images found in '", os.path.split(DATASET_PATH)[1], "/training/*/: ",
          len(list_train_depth_files_path), sep='')
    print()

    # Getting only the filename from the complete file path
    # TODO: Tentar implementar de um jeito que não precise utilizar uma variavel auxiliar
    list_train_colors_files_filename_aux = []
    list_train_depth_files_filename_aux = []

    list_train_colors_files_filename = []
    list_train_depth_files_filename = []
    list_valid_colors_files_filename = []
    list_valid_depth_files_filename = []
    list_test_colors_files_filename = []
    list_test_depth_files_filename = []

    for i in range(0, len(list_train_colors_files_path)):
        list_train_colors_files_filename.append(os.path.split(list_train_colors_files_path[i])[1])
        list_train_colors_files_filename_aux.append(os.path.split(list_train_colors_files_path[i])[1])

    for i in range(0, len(list_train_depth_files_path)):
        list_train_depth_files_filename.append(os.path.split(list_train_depth_files_path[i])[1])
        list_train_depth_files_filename_aux.append(os.path.split(list_train_depth_files_path[i])[1])

    for i in range(0, len(list_test_colors_files_path)):
        list_test_colors_files_filename.append(os.path.split(list_test_colors_files_path[i])[1])

    for i in range(0, len(list_test_depth_files_path)):
        list_test_depth_files_filename.append(os.path.split(list_test_depth_files_path[i])[1])

    print("Checking if all '*_colors.png' files have its correspondent '*_depth.png file'...")
    # TODO: Unificar esse trecho
    if args.dataset == 'nyuDepth':
        for i in range(0, len(list_test_colors_files_path)):
            test_valid_pair_files.append(list_test_depth_files_path[i] in list_test_depth_files_path)
            # test_valid_pair_files.append(list_test_files_colors[i] in list_test_files_depth)

        for i in range(0, len(list_train_colors_files_path)):
            train_valid_pair_files.append(list_train_depth_files_path[i] in list_train_depth_files_path)
            # train_valid_pair_files.append(list_train_files_colors[i] in list_train_files_depth)

    elif args.dataset[0:5] == 'kitti':
        for i in range(0, len(list_test_colors_files_path)):
            test_valid_pair_files.append(os.path.split(list_test_colors_files_path[i])[1] in list_test_depth_files_filename)

            # Debug
            # print(list_test_colors_files_path[i])  # Path of the Colored Image
            # print(os.path.split(list_test_colors_files_path[i])[1])  # Filename of the Colored Image
            # print(list_test_depth_files_filename[i])  # Filename of the Depth Image
            # print(os.path.split(list_test_colors_files_path[i])[1] in list_train_depth_files_filename)  # Comparison

        for i in range(0, len(list_train_colors_files_path)):
            
            train_valid_pair_files.append(os.path.split(list_train_colors_files_path[i])[1] in list_train_depth_files_filename)

            # Debug
            # print(list_train_colors_files_path[i]) # Path of the Colored Image
            # print(os.path.split(list_train_colors_files_path[i])[1])  # Filename of the Colored Image
            # print(list_train_depth_files_filename[i])                 # Filename of the Depth Image
            # print(os.path.split(list_train_colors_files_path[i])[1] in list_train_depth_files_filename) # Comparison

    print("Checked files:", len(test_valid_pair_files) + len(train_valid_pair_files))
    print("Checking Results:\n", test_valid_pair_files, '\n', train_valid_pair_files, '\n')

    print('\nGetting aux index vector that indicates the pair image-depth files...')
    idx = [i for i, x in enumerate(train_valid_pair_files) if x]
    idx2 = [i for i, x in enumerate(test_valid_pair_files) if x]

    print('idx:', idx)
    print('idx_size:', len(idx), '\n')
    print('idx2:', idx2)
    print('idx2_size:', len(idx2), '\n')

    # Creates the vectors what will be dumped as dict words in the pickle file.
    pkl_filename = args.dataset + '.pkl'

    # TODO: Remover?
    # trainSize = round(len(list_train_colors_files_path) * TRAIN_VALID_RATIO)
    # validSize = len(list_train_colors_files_path) - trainSize
    # testSize = len(list_test_colors_files_path)

    trainSize = round(len(idx) * TRAIN_VALID_RATIO)
    validSize = len(idx) - trainSize
    
    if args.dataset == 'nyuDepth':
        testSize = len(idx2)
    elif args.dataset == 'kitti2012' or 'kitti2015':
        testSize = len(list_test_colors_files_path) # Doesn't have Depth Images
        idx2 = np.arange(testSize) # Fill vector for post-enumeration when opening the images
        
    print('trainSize:', trainSize)
    print('validSize:', validSize)
    print('testSize:', testSize)

    list_train_colors_files_filename = list_train_colors_files_filename_aux[:trainSize]
    list_train_depth_files_filename = list_train_depth_files_filename_aux[:trainSize]
    list_valid_colors_files_filename = list_train_colors_files_filename_aux[trainSize:]
    list_valid_depth_files_filename = list_train_depth_files_filename_aux[trainSize:]

    # Debug
    # print(list_train_colors_files_filename)
    # print(len(list_train_colors_files_filename),'\n')
    # print(list_train_depth_files_filename)
    # print(len(list_train_depth_files_filename),'\n')
    # print(list_valid_colors_files_filename)
    # print(len(list_valid_colors_files_filename),'\n')
    # print(list_valid_depth_files_filename)
    # print(len(list_valid_depth_files_filename),'\n')
    # print(list_test_colors_files_filename)
    # print(len(list_test_colors_files_filename),'\n')
    # print(list_test_depth_files_filename)
    # print(len(list_test_depth_files_filename),'\n')
    # input()

    # Original Images
    train_dataset_crop = []
    train_labels_crop = []
    valid_dataset_crop = []
    valid_labels_crop = []
    test_dataset_crop = []
    test_labels_crop = []

    # Processed Images for Network Training
    train_dataset, train_labels = make_arrays(trainSize, imageOutputSize, depthOutputSize)
    valid_dataset, valid_labels = make_arrays(validSize, imageOutputSize, depthOutputSize)
    test_dataset, test_labels = make_arrays(testSize, imageOutputSize, depthOutputSize)

    if os.path.isfile(False):  # TODO: Remover
        # if os.path.isfile(pkl_filename):
        print(pkl_filename, 'already exists!')
    else:
        print('\nDividing available data into training, validation and test sets...')

        for i, val in enumerate(idx):
            # for i in range(0, 10):
            print('i:', i, 'idx:', idx[i], 'image:', list_train_colors_files_path[idx[i]], '\n\t\tdepth:', list_train_depth_files_path[i])
            print(sys.getsizeof(train_dataset_crop), "bytes")


            if i < trainSize:
                crop, train_dataset[i] = openImage(list_train_colors_files_path[idx[i]], showContent=False,
                                             showImagePlot=args.showImageRGB, showCrop=args.showCrop,
                                             showNormedPlot=args.showNormedPlot, cropSize=imageOutputSize)
                train_dataset_crop.append(crop)
                
 
                crop, train_labels[i] = openImage(list_train_depth_files_path[i], showContent=False,
                                            showDepthPlot=args.showImageDepth, showCrop=args.showCrop,
                                            showNormedPlot=args.showNormedPlot,
                                            showDepthResizedPlot=args.showDepthResized, cropSize=imageOutputSize,
                                            resizeSize=depthOutputSize)
                train_labels_crop.append(crop)
            
            else:
                crop, valid_dataset[i - trainSize] = openImage(list_train_colors_files_path[idx[i]], showContent=False,
                                                         showImagePlot=args.showImageRGB, showCrop=args.showCrop,
                                                         showNormedPlot=args.showNormedPlot, cropSize=imageOutputSize)
                
                valid_dataset_crop.append(crop)


                crop, valid_labels[i - trainSize] = openImage(list_train_depth_files_path[i], showContent=False,
                                                        showDepthPlot=args.showImageDepth, showCrop=args.showCrop,
                                                        showNormedPlot=args.showNormedPlot,
                                                        showDepthResizedPlot=args.showDepthResized,
                                                        cropSize=imageOutputSize, resizeSize=depthOutputSize)
                valid_labels_crop.append(crop)

        # print("len(idx2):",len(idx2))
        for i, val in enumerate(idx2):
            if len(list_test_depth_files_path) != 0:  # Has Depth Images
                print('i:', i, 'idx2:', idx2[i], 'image:', list_test_colors_files_path[idx2[i]], 'depth:',
                      list_test_depth_files_path[i])
                crop, test_dataset[i] = openImage(list_test_colors_files_path[i], showContent=False,
                                        showImagePlot=args.showImageRGB, showCrop=args.showCrop,
                                        showNormedPlot=args.showNormedPlot, cropSize=imageOutputSize)

                test_dataset_crop.append(crop)
                
                crop, test_labels[i] = openImage(list_test_depth_files_path[i], showContent=False,
                                           showImagePlot=args.showImageRGB, showCrop=args.showCrop,
                                           showNormedPlot=args.showNormedPlot, 
                                           showDepthResizedPlot=args.showDepthResized,cropSize=imageOutputSize,resizeSize=depthOutputSize)

                test_labels_crop.append(crop)

            else:
                print('i:', i, 'image:', list_test_colors_files_path[i], 'depth:', 'N/A')
                crop, test_dataset[i] = openImage(list_test_colors_files_path[i], showContent=False,
                                        showImagePlot=args.showImageRGB, showCrop=args.showCrop,
                                        showNormedPlot=args.showNormedPlot, cropSize=imageOutputSize)

                test_dataset_crop.append(crop)


        print("\nSummary")
        print("train_dataset shape:", train_dataset.shape)
        print("train_labels shape:", train_labels.shape)
        print("valid_dataset shape:", valid_dataset.shape)
        print("valid_labels shape:", valid_labels.shape)
        print("test_dataset shape:", test_dataset.shape)
        print("test_labels shape:", test_labels.shape)
        print()
        print("train_dataset_crop len:", len(train_dataset_crop))
        print("train_labels_crop len:", len(train_labels_crop))
        print("valid_dataset_crop len:", len(valid_dataset_crop))
        print("valid_labels_crop len:", len(valid_labels_crop))
        print("test_dataset_crop len:", len(test_dataset_crop))
        print("test_labels_crop len:", len(test_labels_crop))

        # Convert lists variables to np.array
        train_dataset_crop = np.asarray(train_dataset_crop)
        train_labels_crop = np.asarray(train_labels_crop)
        valid_dataset_crop = np.asarray(valid_dataset_crop)
        valid_labels_crop = np.asarray(valid_labels_crop)
        test_dataset_crop = np.asarray(test_dataset_crop)
        test_labels_crop = np.asarray(test_labels_crop)

        # Free Memory


        # Dumps data inside pickle file.
        # Saves the Normed images that have correspondent depth images in a pickle file
        print("\nDumping the processed images into the ", pkl_filename, " file...", sep='')
        # pkl_folder_path = 'output/'+appName+'/'
        pkl_folder_path = 'output/'

        if not os.path.exists(pkl_folder_path):
                os.makedirs(pkl_folder_path)

        pkl_file = open(pkl_folder_path+pkl_filename, 'wb')

        data = {'train_dataset_crop': train_dataset_crop, 'train_labels_crop': train_labels_crop, 'valid_dataset_crop': valid_dataset_crop,
                'valid_labels_crop': valid_labels_crop, 'test_dataset_crop': test_dataset_crop, 'test_labels_crop': test_labels_crop,
                'train_dataset': train_dataset, 'train_labels': train_labels, 'valid_dataset': valid_dataset,
                'valid_labels': valid_labels, 'test_dataset': test_dataset, 'test_labels': test_labels,
                'list_train_colors_files_filename': list_train_colors_files_filename,
                'list_train_depth_files_filename': list_train_depth_files_filename,
                'list_valid_colors_files_filename': list_valid_colors_files_filename,
                'list_valid_depth_files_filename': list_valid_depth_files_filename,
                'list_test_colors_files_filename': list_test_colors_files_filename,
                'list_test_depth_files_filename': list_test_depth_files_filename}

        # pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL) # TODO: DESCOMENTAR!!!
        pkl_file.close()


# ======
#  Main
# ======
if __name__ == '__main__':
    main()
