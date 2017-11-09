#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ===========
#  MonoDeep
# ===========
# This Coarse-to-Fine Network Architecture predicts the log depth (log y).

# TODO: Adaptar o código para tambem funcionar com o tamanho do nyuDepth
# TODO: Adicionar metricas
# TODO: Adicionar funcao de custo do Eigen, pegar parte do calculo de gradientes da funcao de custo do monodepth
# FIXME: Arrumar dataset_preparation.py, kitti2012.pkl nao possui imagens de teste
# FIXME: Apos uma conversa com o vitor, aparentemente tanto a saida do coarse/fine devem ser lineares, nao eh necessario apresentar o otimizar da Coarse e a rede deve prever log(depth), para isso devo converter os labels para log(y_)

# ===========
#  Libraries
# ===========
import os
import argparse
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import pprint

from scipy.misc import imshow
from monodeep_model import *
from monodeep_dataloader import *

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def weight_variable(shape, variableName):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.01, dtype=tf.float32)         # Recommend by Vitor Guzilini
    # initial = tf.truncated_normal(shape, mean=0.00005, stddev=0.0001, dtype=tf.float32) # Nick, try to avoid generate negative values
    # initial = tf.truncated_normal(shape, stddev=10.0)                                   # Test
    
    return tf.Variable(initial, name=variableName)

def bias_variable(shape, variableName):
    initial = tf.constant(0.0, dtype=tf.float32, shape=shape) # Recommend by Vitor Guizilini
    # initial = tf.constant(0.1, dtype=tf.float32, shape=shape) # Nick
    # initial = tf.constant(10.0, dtype=tf.float32, shape=shape) # Test

    return tf.Variable(initial, name=variableName)

graph = tf.Graph()
with graph.as_default():
    Wh = weight_variable([10, 10], "c_Wh7")
    bh = bias_variable([10], "c_bh7")


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
        
    # ----- Session Run! ----- #
    Wh, bh = session.run([Wh, bh], feed_dict=[])

    print(Wh)
    print(bh)