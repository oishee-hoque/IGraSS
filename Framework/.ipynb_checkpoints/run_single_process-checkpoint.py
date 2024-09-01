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
import segmentation_models as sm




years = [2020,2021,2022,2023]
IMAGE_SIZE = 512
p_s = IMAGE_SIZE
BATCH_SIZE = 4
NUM_CLASSES = 1
large_val = 999999999999
patch_size = 512


combined_full_h = []
combined_predictions = []
combined_waterMask = []
combined_gt = []
R = None
th = None
r_th = None
outputh_path = None
# img_prefix = 'common_normalized_'
img_prefix = 'train_img_'
gt_prefix = 'common_satellite'
full_h_prefix = 'full_h'
prediction_prefix = 'predictions'
curr_unreachable_nodes_prefix = 'curr_unreachable_nodes'
unreachable_terminals_prefix = 'unreachable_terminals'
water_edge_points_prefix = 'water_edge_points'
source_terminal_pair_prefix = 'source_terminal_pair'


    
water_index_path = None
reachable_nodes_path = None
unreachable_nodes_path = None
reachable_nodes_H_path = None
unreachable_nodes_H_path = None
curr_unreachable_nodes_path = None
common_unreachable_nodes_path = None
 

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
    if gt_path == None:
        # return np.load(f'/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/common_satellite{years}.npy')
        return np.load(f'/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/train_mask_{years}.npy')
    return np.load(gt_path)

def save_gt(gt,years):
    return np.save(f'{outputh_path}/{gt_prefix}{years}.npy',gt)
    
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





def make_predictions(years,model="DeeplabV3+"):

    ## loading the trained model
    if model.lower() =="deeplabv3+": 
        trained_learner = DeeplabV3Plus(image_size=p_s, num_classes=NUM_CLASSES)
        trained_learner.load_weights(weights)
    elif model.lower() == "resunet":
        trained_learner = ResUNet((512, 512, 3))
        trained_learner.load_weights(weights)
    elif model.lower() == "resnet":
        trained_learner = sm.Unet('resnet50', 
                input_shape=(p_s, p_s, 3),
                classes=NUM_CLASSES, activation='sigmoid')
        trained_learner.load_weights(weights)

    image_patches = get_img_patches(years)
    image_patches = np.squeeze(image_patches, axis=2)
    
    predictions = []
    for i in range(image_patches.shape[0]):
            predictions.append(trained_learner.predict(image_patches[i]))

    if not os.path.exists(outputh_path):
        os.makedirs(outputh_path)

    predictions = save_predictions(predictions,years)
    del image_patches,predictions,trained_learner
    gc.collect()
    
    
def genearateH(years):
    predictions = get_predictions(years)
    ground_truth = get_processed_gt(years)

    f = np.array(predictions)
    f = np.squeeze(f,axis=-1)
    canal_mask = ground_truth.copy()
    full_predictions_mask_patches = f.copy()
    
    water_mask = ground_truth.copy()
    water_mask[water_mask!=2] = 0
    
    canal_mask[canal_mask==3] = 1
    canal_mask[canal_mask>1] = 0


    full_predictions_mask_patches[full_predictions_mask_patches>=th] = 1
    full_predictions_mask_patches[full_predictions_mask_patches<th] = 0

    full_predictions_mask = unpatchify(full_predictions_mask_patches, ground_truth.shape) #f

    
    kernel = np.ones((3, 3), np.uint8) 
    new_mask = cv2.dilate(canal_mask,kernel=kernel,iterations=5)
    new_mask = new_mask + full_predictions_mask
    new_mask[new_mask>1] = 1
    H = cv2.erode(new_mask, kernel=kernel,iterations=5)
    del new_mask,full_predictions_mask_patches,full_predictions_mask
    gc.collect()
    return H,predictions,ground_truth,water_mask
    
def find_common_rows(a, b):
    # Convert b to a set of tuples for faster lookup
    b_set = set(map(tuple, b))
    
    # Use a list comprehension to find common rows while preserving order
    common = [row for row in a if tuple(row) in b_set]
    
    return np.array(common)

def generate_all_nodes(years):
    full_h = get_full_h(years)
    full_h_patches = get_full_h_patches(years)
    ground_truth = get_processed_gt(years)
    predictions = get_predictions(years)
    
    # replacing intersection of water mask and canals with just canal pixels
    gt_patches = get_gt_patches(years)
    gt_patches[gt_patches==3]=1

    all_mask = ground_truth.copy()
    all_mask[all_mask==3]=1
    
    canal_mask = ground_truth.copy()
    canal_mask[canal_mask==3] = 1
    canal_mask[canal_mask>1] = 0

    ###Main GT
    water_index = get_water_source_index_patch(all_mask,gt_patches)
    directly_connected_patches,not_connected_patches = directly_connected_1s_patch(all_mask,gt_patches,water_index)
    reachable_nodes, unreachable_nodes = bfs_patch(gt_patches, directly_connected_patches, directly_connected_patches,not_connected_patches)
    print(f"Number of reachable Canal pixels:{len(reachable_nodes)}")

    unreachable_nodes = np.where(unreachable_nodes == 1)
    i,j,d_row_indices, d_col_indices = unreachable_nodes
    unreachable_nodes_index_pairs = list(zip(i,j,d_row_indices, d_col_indices))
    unreachable_nodes_index_pairs = np.array(unreachable_nodes_index_pairs)
    print(f"Number of unreachable Canal pixels in ground truth: {len(unreachable_nodes_index_pairs)}")


    np.save(get_water_index_path(years),list(water_index))
    np.save(get_reachable_nodes_path(years),list(reachable_nodes))
    np.save(get_unreachable_nodes_path(years),list(unreachable_nodes_index_pairs))
    ####Main GT End
    
    
    ## For H
    directly_connected_preds,not_connected_preds = directly_connected_1s_patch(full_h,full_h_patches,water_index)
    reachable_nodes_H, unreachable_nodes_H = bfs_patch(full_h_patches, directly_connected_preds, directly_connected_preds,not_connected_preds)
    print(f"Number of reachable Canal pixels:{len(reachable_nodes_H)}")
    
    
    unreachable_nodes_H = np.where(unreachable_nodes_H == 1)
    i,j,d_row_indices, d_col_indices = unreachable_nodes_H
    unreachable_nodes_index_pairs_H = list(zip(i,j,d_row_indices, d_col_indices))
    unreachable_nodes_index_pairs_H = np.array(unreachable_nodes_index_pairs_H)
    print(f"Number of unreachable Canal pixels in H: {len(unreachable_nodes_index_pairs_H)}")
    
    np.save(get_reachable_nodes_H_path(years),list(reachable_nodes_H))
    np.save(get_unreachable_nodes_H_path(years),list(unreachable_nodes_index_pairs_H))
    ## End H
    
    a = np.array(unreachable_nodes_index_pairs)
    b = np.array(unreachable_nodes_index_pairs_H)

    result = find_common_rows(a, b)
    print("Number of common canal pixels in GT and H",len(result))
    
    curr_unreachable_nodes = unreachable_nodes_index_pairs
    common_unreachable_nodes = result
    
    np.save(get_curr_unreachable_nodes_path(years),curr_unreachable_nodes)
    np.save(get_common_unreachable_nodes_path(years),common_unreachable_nodes)
    
    
def find_isolated_points(data,full_h_patches):
    data_set = set(tuple(point) for point in data)  # Convert array to a set of tuples for fast lookup
    visited = set()  # To track visited points
    endpoints = []  # List to store endpoints
    p_i,p_j,p_rows,p_columns = full_h_patches.shape

    # Directions corresponding to N, NE, E, SE, S, SW, W, NW
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    for point in data:
        neighbors = []
        point_tuple = tuple(point)
        i,j,x,y = point_tuple
        if point_tuple not in visited:
            # Check all 8 directions and count neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                n_i,n_j= i,j

                if 0>nx or nx>=p_rows:
                    n_i = i+dx
                    nx = 511 if nx == -1 else 0 if nx == 512 else nx

                if 0>ny or ny>=p_columns:
                    n_j = j+dy
                    ny = 511 if ny == -1 else 0 if ny == 512 else ny
                    
                if (n_i,n_j,nx, ny) in data_set:
                    neighbors.append((n_i,n_j,nx, ny))

            num_neighbors = len(neighbors)
            
            if num_neighbors <= 1:
                endpoints.append(point)  # Zero or one neighbor, so it's an endpoint
                
            # Mark point and its neighbors as visited
            visited.add(point_tuple)
            # visited.update(neighbors)

    return np.array(endpoints),visited

def find_terminals(years):
    curr_unreachable_nodes = np.load(get_curr_unreachable_nodes_path(years))
    full_h = get_full_h(years) ## previously saved
    full_h_patches = patchify(full_h,(p_s,p_s),step=p_s)
    ground_truth = get_processed_gt(years)
    gt_patches = get_gt_patches(years)
    gt_patches[gt_patches==3]=1

    data = np.array(curr_unreachable_nodes)

    terminals,visited = find_isolated_points(data,gt_patches)
    print(len(terminals))


    unique_patch_nodes = set()

    for i,j,_,_ in curr_unreachable_nodes:
        unique_patch_nodes.add((i,j))

    print("Number of unique patches - for unreachable nodes",len(unique_patch_nodes))

    unique_patch_terminals = set()

    for i,j,_,_ in terminals:
        unique_patch_terminals.add((i,j))

    print("Number of unique patches - for terminal points",len(unique_patch_terminals))
    np.save(get_unreachable_terminals_path(years),terminals)
    del curr_unreachable_nodes, full_h, full_h_patches, ground_truth, gt_patches, data,  terminals, visited, unique_patch_nodes
    
    
    
@jit(nopython=True)
def get_neighbors_fast(i, j, x, y, R, shape, patch_size):
    n, m = shape
    X_L = x + (i * patch_size)
    Y_L = y + (j * patch_size)
    neighbors = []
    for dx in range(-R, R+1):
        for dy in range(-R, R+1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = X_L + dx, Y_L + dy
            if 0 <= nx < n and 0 <= ny < m:
                p_i, p_j = nx // patch_size, ny // patch_size
                p_x, p_y = nx % patch_size, ny % patch_size
                neighbors.append((p_i, p_j, p_x, p_y))
    return neighbors

@jit(nopython=True)
def process_edge_points(u, w, X_r, R, shape, patch_size, th):
    new_points = []
    for edge_point in u:
        i, j, x, y = edge_point
        neighbors = get_neighbors_fast(i, j, x, y, R, shape, patch_size)
        for neighbor in neighbors:
            i, j, x, y = neighbor
            if w[i, j, x, y] > th and X_r[i, j, x, y] == 0:
                X_r[i, j, x, y] = int(1 / w[i, j, x, y])
                new_points.append((i, j, x, y))
    return new_points, X_r

def generate_refined_X(years):
    curr_unreachable_nodes = np.load(get_curr_unreachable_nodes_path(years))
    full_h = get_full_h(years) ## previously saved
    full_h_patches = patchify(full_h,(p_s,p_s),step=p_s)
    ground_truth = get_processed_gt(years)
    gt_patches = get_gt_patches(years)
    gt_patches[gt_patches==3]=1

    terminals = get_terminals(years)
    print(len(terminals))

    predictions = get_predictions(years)
    f = np.array(predictions)
    f = np.squeeze(f,axis=-1)
    w = f.copy()



    X_r = full_h_patches.copy()
    # f = full_predictions_mask.copy()
    # Define the set of edge points u.
    u = terminals.copy()
    X = full_h_patches.copy()
    # Define the sampling radius and threshold.

    new_points = []
    
    new_points, X_r = process_edge_points(u, w, X_r, R, full_h.shape, patch_size, r_th)

    print(X_r.shape)

    # X_r[X_r>=1] = int(1/.9)
    X_r[X_r==0] = large_val
    
    np.save(get_X_r_path(years),X_r)
    
    
def add_vector_to_point(point, vector):
    return (point[0] + vector[0], point[1] + vector[1])

def points_within_radius(center, radius, points):
    # Convert center to a numpy array
    center = np.array(center)
    
    # Calculate the squared distances from each point in the array to the center
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    
    # Find points where the distance is less than or equal to the radius
    within_radius = points[distances <= radius]
    
    return within_radius

def find_edge_points(full_h_patches,data):
    data_set = set(tuple(point) for point in data)  # Convert array to a set of tuples for fast lookup
    visited = set()  # To track visited points
    edgepoints = []  # List to store endpoints
    p_i,p_j,p_rows,p_columns = full_h_patches.shape

    # Directions corresponding to N, NE, E, SE, S, SW, W, NW
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    for point in data:
        neighbors = []
        point_tuple = tuple(point)
        i,j,x,y = point_tuple
        if point_tuple not in visited:
            # Check all 8 directions and count neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                n_i,n_j= i,j

                if 0>nx or nx>=p_rows:
                    n_i = i+dx
                    nx = 511 if nx == -1 else 0 if nx == 512 else nx

                if 0>ny or ny>=p_columns:
                    n_j = j+dy
                    ny = 511 if ny == -1 else 0 if ny == 512 else ny
                    
                if (n_i,n_j,nx, ny) in data_set:
                    neighbors.append((n_i,n_j,nx, ny))

            num_neighbors = len(neighbors)
            
            if num_neighbors < 8:
                edgepoints.append(point)  # Zero or one neighbor, so it's an endpoint
                
            # Mark point and its neighbors as visited
            visited.add(point_tuple)
            # visited.update(neighbors)

    return np.array(edgepoints),visited

def find_source_terminal_pair(source_terms,X_L,Y_L,i,j,radius):
    n_reachable_terms = []
    directions = [(0,0),(0, 1), (0, -1), (1, 0), (-1, 0),
     (-1, -1), (1, -1), (1, 1), (-1, 1)]
    
    resulting_points = np.array([(i, j)]) + directions
    mask = np.isin(source_terms[:, :2], resulting_points).all(axis=1)
    reachable_terms = source_terms[mask, 2:] + source_terms[mask, :2] * 512
    
    if(len(reachable_terms)!=0):
        n_reachable_terms = points_within_radius((X_L,Y_L), radius, reachable_terms)
    return n_reachable_terms

def get_source_terminals(years):
    terminals = get_terminals(years)
    reachable_nodes = np.load(get_reachable_nodes_path(years))
    full_h = get_full_h(years) ## previously saved
    full_h_patches = get_full_h_patches(years)
    ground_truth = get_processed_gt(years)
    gt_patches = get_gt_patches(years)
    water_index = np.load(get_water_index_path(years))
    data = np.array(water_index)

    water_edges,visited = find_edge_points(gt_patches,data)
    print("water_edges",len(water_edges))
    np.save(get_water_edge_points_path(years),water_edges)

    water_edges = np.load(get_water_edge_points_path(years))
    a = reachable_nodes.copy()
    a = np.concatenate((a,water_edges))
    np.save(f'{outputh_path}/all_source_terminals{years}.npy',a)
    source_terms = np.load(f'{outputh_path}/all_source_terminals{years}.npy')
    
    
    source_terminal_pair = {}
    tot_sp = 0
    for terminal in terminals:
        i,j,x,y = terminal
        X_L = (x) + (i*512) 
        Y_L = (y) + (j*512)
        pair_val = find_source_terminal_pair(source_terms,X_L,Y_L,i,j,R)
        tot_sp += len(pair_val)
        source_terminal_pair[f'{(X_L,Y_L)}'] = (pair_val)




    # Save using Pickle
    file_path = get_source_terminal_pair_path(years)
    with open(file_path, 'wb') as file:
        pickle.dump(source_terminal_pair, file)

    # Load the dictionary back
    with open(file_path, 'rb') as file:
        source_terminal_pair = pickle.load(file)
        print(len(source_terminal_pair))

    print("Number of Source Terminal Pair", tot_sp)
    
    
def create_subgraph_around_terminal(matrix, terminal, radius):
    rows, cols = matrix.shape
    G = nx.DiGraph()
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    t_x, t_y = terminal
    for i in range(max(0, t_x - radius), min(rows, t_x + radius + 1)):
        for j in range(max(0, t_y - radius), min(cols, t_y + radius + 1)):
             if matrix[i][j] < large_val:
                    G.add_edge((i, j, 1), (i, j, 2), weight=matrix[i][j])
    
    for i in range(max(0, t_x - radius), min(rows, t_x + radius + 1)):
        for j in range(max(0, t_y - radius), min(cols, t_y + radius + 1)):
            if matrix[i][j] < large_val:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and matrix[ni][nj] < large_val:
                        G.add_edge((i, j, 2), (ni, nj, 1), weight=0)
                        G.add_edge((ni, nj, 2), (i, j, 1), weight=0)
    
    return G


def find_shortest_path(G, terminal, sources):
    # G = create_transformed_graph(matrix)
    terminal_node = (terminal[0], terminal[1], 2)
    shortest_distance = 99999999
    best_source = None
    best_path = None
    source_nodes = []
    try:
        length, paths = nx.single_source_dijkstra(G, source=terminal_node)
    except nx.NetworkXNoPath:
        print("None")

    path = []
    for source in sources:
        s = (source[0], source[1], 1)
        if s in length and length[s]<shortest_distance:
            shortest_distance = length[s]
            path = paths[s]
            best_source = s
    return best_source, length, path
    
def do_reachability_computation(years):
    print(years)
    # image_patches = get_img_patches(years)
    gt_patches = get_gt_patches(years)
    predictions = get_predictions(years)
    X_r = np.load(get_X_r_path(years))
    full_h = get_full_h(years)
    full_h_patches = get_full_h_patches(years)
    terminals = get_terminals(years)
    netwk = unpatchify(X_r, full_h.shape)
    file_path = get_source_terminal_pair_path(years)
    with open(file_path, 'rb') as file:
        source_terminal_pair = pickle.load(file)
    paths = []
    terminal_paths = {}
    cnt = 0
    for terminal in terminals:
        i,j,x,y = terminal
        X_L = (x) + (i*512) 
        Y_L = (y) + (j*512)
        t_points = (X_L,Y_L)
        source_points = source_terminal_pair[f'{(X_L,Y_L)}']
        goals = list(map(tuple, source_points))
        if len(goals)!= 0:
            netwk_t = create_subgraph_around_terminal(netwk,t_points, R)
            best_source, shortest_distance, best_path = find_shortest_path(netwk_t,t_points,goals)
            if best_source != None:
                cnt = cnt+1
                terminal_paths[f'{t_points}'] = best_path
                paths.extend(best_path)

    print('terminals got connected',cnt)
    file_path = generated_paths_from_terminals(years)
    with open(file_path, 'wb') as file:
        pickle.dump(terminal_paths, file)  
def find_common_rows_2(a_s, b_s):
    # Convert b to a set of tuples for faster lookup
    b_set = set(map(tuple, b_s))

    # Use a list comprehension to find common rows while preserving order
    common = [[a,b,c,d] for a,b,c,d in a_s if tuple([c+(a*512),d+(b*512)]) in b_set]

    return np.array(common)   

def run_process(model_t,it):
    # Set global variables
    # global R, th, r_th, outputh_path
    global water_index_path,reachable_nodes_path,unreachable_nodes_path,reachable_nodes_H_path,curr_unreachable_nodes_path,common_unreachable_nodes_path,gt_path
    
    
    water_index_path = f"{outputh_path}/water_index.npy"
    reachable_nodes_path = f"{outputh_path}/{it}_reachable_nodes.npy"
    unreachable_nodes_path = f"{outputh_path}/{it}_unreachable_nodes.npy"
    reachable_nodes_H_path = f"{outputh_path}/{it}_reachable_nodes_H.npy"
    unreachable_nodes_H_path = f"{outputh_path}/{it}_unreachable_nodes_H.npy"
    curr_unreachable_nodes_path = f"{outputh_path}/{it}_curr_unreachable_nodes.npy"
    common_unreachable_nodes_path = f"{outputh_path}/{it}_common_unreachable_nodes.npy"
    print(R,th,r_th,outputh_path)
    
    for i in range(4):
        print(years[i])
        print("It: Step 1")
        make_predictions(years[i],model_t)
        print("It: Step 2")  
        H,predictions,ground_truth,water_mask = genearateH(years[i])
        combined_full_h.append(H)
        combined_predictions.append(predictions)
        combined_waterMask.append(water_mask)
        combined_gt.append(ground_truth)

    H = np.array(combined_full_h)
    H = np.max(H, axis=0)
    predictions = np.max(np.array(combined_predictions), axis=0)
    water_mask = np.max(np.array(combined_waterMask), axis=0)
    gt = np.max(np.array(combined_gt), axis=0)

    full_h = water_mask+H
    full_h[full_h==3]=1
    pref_name = f"_it{it}_{R}_{th}_{r_th}_combined"
    save_full_h(pref_name,full_h)
    save_predictions(predictions,pref_name)
    save_watermask(water_mask,pref_name)
    save_gt(gt,pref_name)
    gt_path = f'{outputh_path}/{gt_prefix}{pref_name}.npy'
    
    print("It: Step 3")
    generate_all_nodes(pref_name)
    print("It: Step 4")
    find_terminals(pref_name)
    print("It: Step 5")
    generate_refined_X(pref_name)
    print("It: Step 6")
    get_source_terminals(pref_name)
    print("It: Step 7")
    do_reachability_computation(pref_name)

    def find_isolated_points(data):
        data_set = set(tuple(point) for point in data)  # Convert array to a set of tuples for fast lookup
        visited = set()  # To track visited points
        endpoints = []  # List to store endpoints
        p_i,p_j,p_rows,p_columns = full_h_patches.shape

        # Directions corresponding to N, NE, E, SE, S, SW, W, NW
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        for point in data:
            neighbors = []
            point_tuple = tuple(point)
            i,j,x,y = point_tuple
            if point_tuple not in visited:
                # Check all 8 directions and count neighbors
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    n_i,n_j= i,j

                    if 0>nx or nx>=p_rows:
                        n_i = i+dx
                        nx = 511 if nx == -1 else 0 if nx == 512 else nx

                    if 0>ny or ny>=p_columns:
                        n_j = j+dy
                        ny = 511 if ny == -1 else 0 if ny == 512 else ny

                    if (n_i,n_j,nx, ny) in data_set:
                        neighbors.append((n_i,n_j,nx, ny))

                num_neighbors = len(neighbors)

                if num_neighbors <= 1:
                    endpoints.append(point)  # Zero or one neighbor, so it's an endpoint

                # Mark point and its neighbors as visited
                visited.add(point_tuple)
                # visited.update(neighbors)

        return np.array(endpoints),visited




    year = pref_name
    # image_patches = get_img_patches(year)
    gt_patches = get_gt_patches(year)
    predictions = get_predictions(year)
    X_r = np.load(get_X_r_path(year))
    full_h = get_full_h(year)
    full_h_patches = get_full_h_patches(year)
    terminals = get_terminals(year)
    ground_truth = get_processed_gt(year)
    n_all_mask = ground_truth.copy()
    added_nodes = 0
    file_path = generated_paths_from_terminals(year)
    with open(file_path, 'rb') as file:
        terminal_paths = pickle.load(file)
    paths = []
    for terms in list(terminal_paths.keys()):
        paths.extend(terminal_paths[terms])
    for x,y,_ in paths:
        if n_all_mask[x,y]!=1:
            added_nodes += 1
        n_all_mask[x,y] = 1
    # print(added_nodes)
    water_index = get_water_source_index(n_all_mask)
    directly_connected_full,not_connected_full = directly_connected_1s(n_all_mask,water_index)
    reachable_nodes, unreachable_nodes = bfs(n_all_mask, directly_connected_full, directly_connected_full,not_connected_full)
    print(f"Number of reachable Canal pixels:{len(reachable_nodes)}")


    unreachable_nodes_N = np.where(unreachable_nodes == 1)
    d_row_indices, d_col_indices = unreachable_nodes_N
    unreachable_nodes_index_pairs = list(zip(d_row_indices, d_col_indices))
    unreachable_nodes_index_pairs = np.array(unreachable_nodes_index_pairs)
    # np.save(f"{outputh_path}/it_1{y}.npy",n_all_mask)
    curr_unreachable_nodes = np.load(get_curr_unreachable_nodes_path(year))
    print(f"Current Number of unreachable Canal pixels:{len(unreachable_nodes_index_pairs)}")




    a = np.array(curr_unreachable_nodes)  ## from gt
    b = np.array(unreachable_nodes_index_pairs)  # after completion (new)
    c = np.load(get_unreachable_nodes_path(year)) #(main GT)
    d= np.load(get_common_unreachable_nodes_path(year)) #(main GT)

    print(f"Prev Number of unreachable Canal pixels from H: {len(curr_unreachable_nodes)}")

    result = find_common_rows_2(a, b)
    result2 = find_common_rows_2(c, b)
    result3 = find_common_rows_2(d, b)
    print(f"Unreachable Canal pixels Left in main H:{len(result)}")
    print(f"Unreachable Canal pixels Left from GT:{len(result2)}")
    print(f"Unreachable Canal pixels Left from common:{len(result3)}")
    refined_gt_path =f"{outputh_path}{year}_refined_mask.npy"
    np.save(refined_gt_path,n_all_mask)
   


    data = np.array(result3)
    terminals_r,visited = find_isolated_points(data)
    # print("Endpoints:", terminals)
    print("Terminals",len(terminals_r))

    unique_patch_nodes = set()

    for i,j,_,_ in result3:
        unique_patch_nodes.add((i,j))

    print("Unique patches - Nodes",len(unique_patch_nodes))

    unique_patch_terminals = set()

    for i,j,_,_ in terminals_r:
        unique_patch_terminals.add((i,j))

    print("Unique patches - Terminals",len(unique_patch_terminals))
    print(f"Completed run with R={R}, th={th}, r_th={r_th}")
    
    print(refined_gt_path,len(reachable_nodes),len(curr_unreachable_nodes),len(unreachable_nodes_index_pairs),len(result2),len(terminals),len(terminals_r))

    return refined_gt_path,len(reachable_nodes),len(curr_unreachable_nodes),len(unreachable_nodes_index_pairs),len(result2),len(terminals),len(terminals_r)
    
    

    
    
def gen_refine_gt(f_R, f_th, f_r_th, f_outputh_path, f_sattelite_imagery_path, f_gt_path, f_weights,model_t,it):
    
    global R, th, r_th, outputh_path, sattelite_imagery_path, gt_path, weights
    
    R = f_R
    th = f_th
    r_th = f_r_th
     
    outputh_path = f_outputh_path
    sattelite_imagery_path = f_sattelite_imagery_path
    gt_path = f_gt_path
    weights = f_weights
    
    if not os.path.exists(outputh_path):
        os.makedirs(outputh_path)
    
    return run_process(model_t,it)