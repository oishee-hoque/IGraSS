import numpy as np
import matplotlib.pyplot as plt
import cv2
from patchify import patchify, unpatchify
import sys
from network_utils import *
from numba import jit   
import pickle
import networkx as nx
import csv
import io
from contextlib import redirect_stdout
import os
import gc
from ResUNet import ResUNet

sys.path.insert(0, './utils')

from deeplabV3Model import *
from network_utils import *




years = [2020,2021,2022,2023]
IMAGE_SIZE = 512
p_s = IMAGE_SIZE
BATCH_SIZE = 4
NUM_CLASSES = 1
large_val = 999999999999
patch_size = 512


outputh_path = '/scratch/gza5dr/Current_Canal_Experiments/Canal_Detection_Experiments/Proposed_Model_Pipeline/implementation/framework/R20/resunet_final_outputs_v2'
sattelite_imagery_path = '/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/processed_images'
gt_path = '/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/'
weights = '/scratch/gza5dr/Current_Canal_Experiments/Canal_Detection_Experiments/ResUNet/models/resunet_e.weights.h5'
img_prefix = 'common_normalized_'
gt_prefix = 'common_satellite'
full_h_prefix = 'full_h'
prediction_prefix = 'predictions'
curr_unreachable_nodes_prefix = 'curr_unreachable_nodes'
unreachable_terminals_prefix = 'unreachable_terminals'
water_edge_points_prefix = 'water_edge_points'
source_terminal_pair_prefix = 'source_terminal_pair'


    
water_index_path = f"{outputh_path}/water_index.npy"
reachable_nodes_path = f"{outputh_path}/reachable_nodes.npy"
unreachable_nodes_path = f"{outputh_path}/unreachable_nodes.npy"
reachable_nodes_H_path = f"{outputh_path}/reachable_nodes_H.npy"
unreachable_nodes_H_path = f"{outputh_path}/unreachable_nodes_H.npy"
curr_unreachable_nodes_path = f"{outputh_path}/curr_unreachable_nodes.npy"
common_unreachable_nodes_path = f"{outputh_path}/common_unreachable_nodes.npy"
 

def get_water_index_path(years):
    return f"{outputh_path}/water_index{years}.npy"
def get_reachable_nodes_path(years):
    return f"{outputh_path}/reachable_nodes{years}.npy"
def get_unreachable_nodes_path(years):
    return f"{outputh_path}/unreachable_nodes{years}.npy"
def get_reachable_nodes_H_path(years):
    return f"{outputh_path}/reachable_nodes_H{years}.npy"
def get_unreachable_nodes_H_path(years):
    return f"{outputh_path}/unreachable_nodes_H{years}.npy"
def get_curr_unreachable_nodes_path(years):
    return f"{outputh_path}/curr_unreachable_nodes{years}.npy"
def get_common_unreachable_nodes_path(years):
    return f"{outputh_path}/common_unreachable_nodes{years}.npy"
def get_unreachable_terminals_path(years):
    return f'{outputh_path}/{unreachable_terminals_prefix}{years}.npy'
def get_X_r_path(years):
    return f'{outputh_path}/X_r{years}.npy'
def get_water_edge_points_path(years):
    return f'{outputh_path}/{water_edge_points_prefix}{years}.npy'
def get_source_terminal_pair_path(years):
    return f'{outputh_path}/{source_terminal_pair_prefix}{years}.pkl'
def generated_paths_from_terminals(years):
    return f'{outputh_path}/terminal_paths{years}.pkl'


def get_sattelite_image(years):
    return np.load(f'{sattelite_imagery_path}/{img_prefix}{years}.npy')

def get_gt(years):
    return np.load(f'{gt_path}/{gt_prefix}{years}.npy')

def save_gt(gt,years):
    return np.save(f'{gt_path}/{gt_prefix}{years}.npy',gt)
    
def get_processed_gt(years):
    ground_truth = get_gt(years)
    cut_1,cut_2 = ground_truth.shape[0]%p_s,ground_truth.shape[1]%p_s
    ground_truth = ground_truth[cut_1:,cut_2:]
    
    return ground_truth

def get_processed_img(years):
    sattelite_imagery = get_sattelite_image(years)
    cut_1,cut_2 = sattelite_imagery.shape[0]%p_s,sattelite_imagery.shape[1]%p_s
    img = sattelite_imagery[cut_1:,cut_2:,:]
    return img
    
def get_processed_img_mask(years):
    sattelite_imagery = get_sattelite_image(years)
    ground_truth = get_gt(years)
    cut_1,cut_2 = ground_truth.shape[0]%p_s,ground_truth.shape[1]%p_s
    img = sattelite_imagery[cut_1:,cut_2:,:]
    ground_truth = ground_truth[cut_1:,cut_2:]
    
    return img,ground_truth

def get_gt_patches(years):
    ground_truth = get_processed_gt(years) 
    return patchify(ground_truth,(p_s,p_s),step=p_s)
    
def get_img_patches(years):
    img = get_processed_img(years)
    return patchify(img,(p_s,p_s,img.shape[2]),step=p_s)

def get_predictions(years):
    return np.load(f"{outputh_path}/{prediction_prefix}{years}.npy")

def save_predictions(predictions,years):
    np.save(f"{outputh_path}/{prediction_prefix}{years}.npy",predictions)
    
def get_watermask(years):
    return np.load(f"{outputh_path}/{prediction_prefix}{years}.npy")

def save_watermask(watermask,years):
    np.save(f"{outputh_path}/{'water_mask'}{years}.npy",watermask)
    
def get_full_h(years):
    return np.load(f'{outputh_path}/{full_h_prefix}{years}.npy')

def save_full_h(years,full_h):
     np.save(f"{outputh_path}/{full_h_prefix}{years}.npy",full_h)
        
def get_curr_unreachable_nodes(years=''):
    return np.load(get_curr_unreachable_nodes_path(years))

def get_full_h_patches(years):
    full_h = get_full_h(years)
    return patchify(full_h,(p_s,p_s),step=p_s)

def get_terminals(years):
    return np.load(get_unreachable_terminals_path(years))



    
    
    
    
    
    