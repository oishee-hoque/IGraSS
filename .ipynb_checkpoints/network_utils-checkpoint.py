from patchify import patchify, unpatchify
import time
from zipfile import ZipFile
import re
import io
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import numpy as np
import earthpy.plot as ep
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from rasterio.features import rasterize
import warnings
from rasterio.errors import ShapeSkipWarning
import cv2
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from scipy.signal import convolve2d
from collections import deque

    

def get_water_source_index(full_mask):
    f_indices = np.where(full_mask == 2)
    row_indices, col_indices = f_indices
    f_index_pairs = list(zip(row_indices, col_indices))
    print(f'Number of Indices of 2s in full image: {len(f_index_pairs)}')
    return f_index_pairs

def get_water_source_index_patch(full_mask,patches,patch_size=512):
    print("Step_1,get_water_source_index_patch")
    start_time = time.time()
    f_index_pairs = get_water_source_index(full_mask)
    ## Let's do it in code. We're projeceting the indices of 2s in smaller patches from
## the larger patch calculated earlier
    patch_val = []
    patch_size = 512
    for i,j in f_index_pairs:
        p_i,p_j,p_x,p_y = (int(i/patch_size),int(j/patch_size),i%patch_size,j%patch_size)
        assert(full_mask[i,j] == patches[p_i,p_j,p_x,p_y])
        patch_val.append((p_i,p_j,p_x,p_y))
    print(time.time()-start_time)
    
    return patch_val


def directly_connected_1s(full_Mask, f_index_pairs):
    print("Step_2,directly_connected_1s")
    start = time.time()

    rows, columns = full_Mask.shape
    not_connected_full = full_Mask.copy()

    # Create a mask for the f_index_pairs
    mask = np.zeros_like(full_Mask, dtype=bool)
    mask[tuple(np.array(list(f_index_pairs)).T)] = True

    # Create kernels for 8-connectivity
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])

    # Convolve the mask with the kernel
    convolved = np.zeros_like(full_Mask, dtype=int)
    from scipy.signal import convolve2d
    convolved = convolve2d(mask, kernel, mode='same', boundary='fill', fillvalue=0)

    # Find directly connected 1s
    directly_connected = (convolved > 0) & (full_Mask == 1)
    directly_connected_full = set(zip(*np.where(directly_connected)))

    # Update not_connected_full
    not_connected_full[directly_connected] = 0

    end = time.time()
    print(end - start)

    return directly_connected_full, not_connected_full



def directly_connected_1s_patch(full_mask, patches, patch_val):
    print("Step_2,directly_connected_1s_patch")
    start = time.time()

    rows, columns = full_mask.shape
    p_i, p_j, p_rows, p_columns = patches.shape

    # Create a mask for the patch_val
    mask = np.zeros_like(full_mask, dtype=bool)
    for i, j, x, y in patch_val:
        mask[x + i*p_rows, y + j*p_columns] = True

    # Create kernel for 8-connectivity
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])

    # Convolve the mask with the kernel
    convolved = convolve2d(mask, kernel, mode='same', boundary='wrap')

    # Find directly connected 1s
    directly_connected = (convolved > 0) & (full_mask == 1)

    # Convert directly_connected to patch format
    directly_connected_patches = set()
    not_connected_patches = patches.copy()

    for i in range(p_i):
        for j in range(p_j):
            patch_connected = directly_connected[i*p_rows:(i+1)*p_rows, j*p_columns:(j+1)*p_columns]
            patch_coords = np.where(patch_connected)
            directly_connected_patches.update((i, j, x, y) for x, y in zip(*patch_coords))
            not_connected_patches[i, j][patch_connected] = 3

    end = time.time()
    print(end - start)

    # Perform assertion check
    for n_i, n_j, nx, ny in directly_connected_patches:
        X_L = nx + (n_i * p_rows)
        Y_L = ny + (n_j * p_columns)
        assert full_mask[X_L, Y_L] == patches[n_i, n_j, nx, ny]

    return directly_connected_patches, not_connected_patches


def bfs(array, start, directly_connected, not_connected_full):
    print("Step_3,bfs")
    rows, columns = array.shape
    directions = np.array([(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(1,-1),(1,1),(-1,1)])
    
    visited = np.zeros_like(array, dtype=bool)
    visited[tuple(np.array(list(start)).T)] = True
    
    queue = deque(start)
    while queue:
        x, y = queue.popleft()
        neighbors = np.array([x, y]) + directions
        valid = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < rows) & \
                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < columns)
        neighbors = neighbors[valid]
        
        for nx, ny in neighbors:
            if not visited[nx, ny] and array[nx, ny] == 1:
                visited[nx, ny] = True
                queue.append((nx, ny))
                not_connected_full[nx, ny] = 0
    
    return set(zip(*np.where(visited))), not_connected_full

def bfs_patch(array, start, directly_connected, not_connected_patches):
    print("Step_3,bfs_patch")
    p_i, p_j, p_rows, p_columns = array.shape
    directions = np.array([(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(1,-1),(1,1),(-1,1)])
    
    visited = set(start)
    queue = deque(start)
    
    while queue:
        i, j, x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            n_i, n_j = i, j
            
            if nx < 0 or nx >= p_rows:
                n_i = i + dx
                nx = p_rows - 1 if nx == -1 else 0 if nx == p_rows else nx
            if ny < 0 or ny >= p_columns:
                n_j = j + dy
                ny = p_columns - 1 if ny == -1 else 0 if ny == p_columns else ny
            
            if 0 <= n_i < p_i and 0 <= n_j < p_j and (n_i, n_j, nx, ny) not in visited and array[n_i, n_j, nx, ny] == 1:
                visited.add((n_i, n_j, nx, ny))
                queue.append((n_i, n_j, nx, ny))
                not_connected_patches[n_i, n_j, nx, ny] = 0
    
    return visited, not_connected_patches













############################ Old #################################
# def directly_connected_1s(full_Mask,f_index_pairs):
#     print("Step_2,directly_connected_1s")
#     not_connected_full = full_Mask.copy()

#     start = time.time()
#     array = full_Mask.copy()

#     rows,columns = array.shape
#     directly_connected_full = set()

#     directions = [(0,1),(0,-1),(1,0),(-1,0),
#                   (-1,-1),(1,-1),(1,1),(-1,1)]

#     for i,j in f_index_pairs:
#         for dx,dy in directions:
#             nx, ny = i+dx, j+dy
#             if 0<=nx<rows and 0<=ny<columns and array[nx,ny]==1:
#                 directly_connected_full.add((nx, ny))
#                 not_connected_full[nx,ny] = 0
#     end = time.time()
#     print(end-start)
    
#     return directly_connected_full,not_connected_full

# def directly_connected_1s_patch(full_mask,patches,patch_val):
#     print("Step_2,directly_connected_1s_old")
#     not_connected_patches = patches.copy()
#     start = time.time()
#     array = patches.copy()

#     p_i,p_j,p_rows,p_columns = array.shape
#     directly_connected_patches = set()

#     directions = [(0,1),(0,-1),(1,0),(-1,0),
#                   (-1,-1),(1,-1),(1,1),(-1,1)]

#     for i,j,x,y in patch_val:
#         for dx,dy in directions:
#             nx, ny = x+dx, y+dy
#             n_i,n_j= i,j

#             if 0>nx or nx>=p_rows:
#                 n_i = i+dx
#                 nx = 511 if nx == -1 else 0 if nx == 512 else nx

#             if 0>ny or ny>=p_columns:
#                 n_j = j+dy
#                 ny = 511 if ny == -1 else 0 if ny == 512 else ny

#             X_L = (nx) + (n_i*512)
#             Y_L = (ny) + (n_j*512)

#             if 0<=n_i<p_i and 0<=n_j<p_j and array[n_i,n_j,nx,ny]==1:
#                 assert(full_mask[X_L,Y_L] == patches[n_i,n_j,nx,ny])
#                 directly_connected_patches.add((n_i,n_j,nx,ny))
#                 not_connected_patches[n_i,n_j,nx,ny] = 3
#     end = time.time()
#     print(end-start)
    
#     return directly_connected_patches,not_connected_patches

# def bfs(array, start, dierectly_connected,not_connected_full):
#     print("Step_3,bfs")
#     rows,columns = array.shape
#     directions = [(0,1),(0,-1),(1,0),(-1,0),
#                   (-1,-1),(1,-1),(1,1),(-1,1)]
#     visited = set(start)
#     queue = list(start)

#     while queue:
#         x,y = queue.pop(0)
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0<=nx <rows and 0<=ny<columns and (nx, ny) not in visited and array[nx, ny] == 1:
#                 visited.add((nx, ny))
#                 queue.append((nx, ny))
#                 not_connected_full[nx,ny] = 0
                
#     return visited,not_connected_full

# def bfs_patch(array, start, dierectly_connected,not_connected_patches):
#     print("Step_3,bfs")
#     p_i,p_j,p_rows,p_columns = array.shape
#     directions = [(0,1),(0,-1),(1,0),(-1,0),
#                   (-1,-1),(1,-1),(1,1),(-1,1)]
#     visited = set(start)
#     queue = list(start)
    
#     while queue:
#         i,j,x,y = queue.pop(0)
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             n_i,n_j= i,j
        
#             if 0>nx or nx>=p_rows:
#                 n_i = i+dx
#                 nx = 511 if nx == -1 else 0 if nx == 512 else nx

#             if 0>ny or ny>=p_columns:
#                 n_j = j+dy
#                 ny = 511 if ny == -1 else 0 if ny == 512 else ny
                
#             if 0<=n_i<p_i and 0<=n_j<p_j and (n_i,n_j,nx, ny) not in visited and array[n_i,n_j,nx,ny]==1:
#                 visited.add((n_i,n_j,nx, ny))
#                 queue.append((n_i,n_j,nx, ny))
#                 not_connected_patches[n_i,n_j,nx, ny] = 0
#     return visited,not_connected_patches
