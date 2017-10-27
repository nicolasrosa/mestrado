#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  MonoDeep
# ===========


# ===========
#  Libraries
# ===========
import os
import argparse
import tensorflow as tf
import numpy as np
import sys

from monodeep_model import *
from monodeep_dataloader import *

# TODO: Remover
# import matplotlib.pyplot as plt
# from scipy import misc as scp
# from PIL import Image
# import pstats


    # ==================
    #  Global Variables
    # ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ===========
#  Functions
# ===========
def argumentHandler():
    # Creating Arguments Parser
    parser = argparse.ArgumentParser(description='Monodeep TensorFlow implementation.')

    # Input
    parser.add_argument('-m', '--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument(      '--model_name',                type=str,   help='model name', default='monodeep')
    # parser.add_argument(    '--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    # parser.add_argument(      '--dataset',                   type=str,   help='dataset to train on, kitti, or nyuDepth', default='kitti')
    parser.add_argument('-i', '--data_path',                 type=str,   help="set relative path to the dataset <filename>.pkl file", required=True)
    # parser.add_argument(    '--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument(      '--input_height',              type=int,   help='input height', default=256)
    parser.add_argument(      '--input_width',               type=int,   help='input width', default=512)
    parser.add_argument(      '--batch_size',                type=int,   help='batch size', default=16)
    parser.add_argument('-e', '--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('-l', '--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('-d', '--dropout',                   type=float,  help="enable dropout in the model during training", default=0.5)
    parser.add_argument(      '--ldecay',                    type=bool,  help="enable learning decay", default=False) 
    parser.add_argument('-n', '--l2norm',                    type=bool,  help="Enable L2 Normalization", default=False)
    
    # parser.add_argument(      '--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
    # parser.add_argument(      '--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    # parser.add_argument(      '--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    # parser.add_argument(      '--do_stereo',                             help='if set, will train the stereo model', action='store_true')
    # parser.add_argument(      '--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
    # parser.add_argument(      '--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
    parser.add_argument(      '--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
    # parser.add_argument(      '--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('-o', '--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
    parser.add_argument(      '--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument(      '--restore_path',              type=str,   help='path to a specific restore to load', default='')
    parser.add_argument(      '--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument(      '--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

    # TODO: Adicionar acima
    # parser.add_argument('-t','--showTrainingErrorProgress', action='store_true', dest='showTrainingErrorProgress', help="Show the first batch label, the correspondent Network predictions and the MSE evaluations.", default=False)
    # parser.add_argument('-v','--showValidationErrorProgress', action='store_true', dest='showValidationErrorProgress', help="Show the first validation example label, the Network predictions, and the MSE evaluations", default=False)
    # parser.add_argument('-u', '--showTestingProgress', action='store_true', dest='showTestingProgress', help="Show the first batch testing Network prediction img", default=False)
    # parser.add_argument('-p', '--showPlots', action='store_true', dest='enablePlots', help="Allow the plots being displayed", default=False) # TODO: Correto seria falar o nome dos plots habilitados
    # parser.add_argument('-s', '--save', action='store_true', dest='enableSave', help="Save the trained model for later restoration.", default=False)

    # parser.add_argument('--saveValidFigs', action='store_true', dest='saveValidFigs', help="Save the figures from Validation Predictions.", default=False) # TODO: Add prefix char '-X'
    # parser.add_argument('--saveTestPlots', action='store_true', dest='saveTestPlots', help="Save the Plots from Testing Predictions.", default=False)   # TODO: Add prefix char '-X'
    # parser.add_argument('--saveTestFigs', action='store_true', dest='saveTestFigs', help="Save the figures from Testing Predictions", default=False)    # TODO: Add prefix char '-X'

    return parser.parse_args()

def train(params, args):
    """Training loop."""
    print('[App] Selected mode: Train')

    # =========================================
    #  Network Training Model - Building Graph
    # =========================================
    print("\n[Network] Constructing Graph...")

    graph = tf.Graph()
    with graph.as_default():
        # Optimizer
        dataloader = MonoDeepDataloader(params, args.mode, args.data_path)
        model = MonoDeepModel(params, args.mode, dataloader.inputSize, dataloader.outputSize)

        
        # Session

        # Saver

        # Count Params

        # Init

        # Load checkpoint if set

        # GO!





    
def test(params, args):
    print('[App] Selected mode: Test')

# ======
#  Main
# ======
def main(_):
    print("[App] Running...")

    args = argumentHandler();
    params = monodeep_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        dropout=args.dropout,
        full_summary=args.full_summary)

    
    if args.mode == 'train':
        train(params, args)
    elif args.mode == 'test':
        test(params, args)

    print("\n[App] DONE!")
    sys.exit()


# ======
#  Main 
# ======
if __name__ == '__main__':
    tf.app.run()
