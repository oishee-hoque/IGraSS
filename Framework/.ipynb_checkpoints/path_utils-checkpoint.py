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
from PIL import Image
from PIL.PngImagePlugin import PngInfo
    
from affine import Affine
import pyproj
from functools import partial
from pyproj import Proj, transform


# Ignore ShapeSkipWarning
warnings.filterwarnings('ignore', category=ShapeSkipWarning)


class Data_Handler():
    def __init__(self, zip_file_path ='/scratch/gza5dr/Canal_Datasets/temp_3A/', 
                       images_folder_path='/scratch/gza5dr/Canal_Datasets/3A_processed/',
                       waterbody_shapefile="/scratch/gza5dr/Canal_Datasets/NHDShape/full_waterbody/", 
                       canal_shapefile = "/scratch/gza5dr/Canal_Datasets/CBP_waterdeliv",
                       merged_image_path = '/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/images/',
                       merged_mask_path = '/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/',
                       crs = "EPSG:32611"):
        
        self.zip_file_path = zip_file_path
        self.images_folder_path = images_folder_path
        self.file_list = [f for f in os.listdir(zip_file_path)]
        self.crs = crs
        self.waterbody_shapefile = gpd.read_file(waterbody_shapefile).to_crs("EPSG:32611")
        self.canal_shapefile = gpd.read_file(canal_shapefile).to_crs(crs)
        self.merged_image_path = merged_image_path
        self.merged_mask_path = merged_mask_path
        self.metadata = None
        
    def get_files_from_folder(self,folder_path,dir_name=None, search_criteria="*.tif"):
        """
        Searches for and returns a list of file paths matching a specified pattern within a given directory.
        Parameters:
        - folder_path (str): The base directory path where the search begins. Can be an absolute or relative path.
        - dir_name (str,optional): subdirectory within `folder_path` to search in.
        - search_criteria (str, optional): The pattern used to match files, defaulting to "*.tif".
        Returns:
        - list of str: A list containing the full paths to all matched files within the specified directory.
          Returns an empty list if no files match the criteria.
        """
        if dir_name != None:
            path = folder_path+'/'+dir_name
        else:
            path = folder_path
        file_paths = os.path.join(path, search_criteria)
        files = glob.glob(file_paths)

        return files
    
    def concat_files(self, files,n_crs='EPSG:32611'):
        """
        Convert all files to same CRS and concate them together
        Parameters:
        - files (str): list of strs containing the full paths to ammend into one source
        - n_crs: crs to convert to
        Returns:
        - a appended list of files converted into one crs, and metadata
        """        
        src_files_to_mosaic = []
        meta = []
        # Iterate over raster files and add them to source -list in 'read mode'
        for fp in files:
            src = rasterio.open(fp)
            if src.meta['crs'] != n_crs:
                src = repro(fp)  ## this function is coming from data utils file, to convert the the image in another crs
            src_files_to_mosaic.append(src)
            meta = src.meta.copy()
            del src

        return src_files_to_mosaic,meta
    
    def generate_masks(self,image_metadata,shapefile, value=1,
                       dialation = False, dialation_kernel=(2,2),iteration=1):
        """
        Generate masks from given shape file and image metadata
        Parameters:
        - image_metadata: contains image height, width and necessary transformation
        - shapefile: containing the geometry
        - value: the value for fill the masks in the array
        - dialation: default False, if dialation is needed on mask then it should be true
        - dialation_kernel: default=(2,2)
        - iteration: default 1, for dialation
        Returns:
        - nxm (image hxw) array of mask
        """        
        # Create a blank array
        array = np.zeros((image_metadata['height'], image_metadata['width']), dtype=np.uint8)
        # Rasterize the line strings
        shapes = ((geom, value) for geom in shapefile.geometry)
        mask = rasterize(shapes=shapes, out_shape=array.shape, transform=image_metadata['transform'])
        
        if(dialation == True):
            mask = perform_dialation(mask, dialation_kernel, iteration)

        return mask
    
    def perform_dialation(self, arr,kernel_width=(2,2),iterations=1):
        """
        Perform dialation on a given mask
        Parameters:
        - arr(nxm): the mask to be dialated
        - kernel_width (int(x,y)): width of the dialation to be performed
        - iterations(single int, optional)
        Returns:
        - a dialted array
        """    
        kernel = np.ones(kernel_width, np.uint8)
        dilated_arr = cv2.dilate(arr, kernel, iterations=1)
        return dilated_arr

    def save_mossaic(self,dir_name,folder_path=None,crs=None,merged_image_path=None,
                          save_file=False,file_name='mosaic'):
        """
        Saving the combined mossaic in a folder
        Parameters:
        - dir_name(str): directory of each year
        - folder_path(str): base folder containing all the folders of the year
        - crs: crs of the image
        - merged_image_path: destination path where the combined image should be saved
        Returns:
        - saves the combined image in the given destination folder and return the combine image 
        and the metadata
        """    
        folder_path = folder_path or self.images_folder_path
        crs = crs or self.n_crs
        merged_image_path = merged_image_path or self.merged_image_path
        
        files = self.get_files_from_folder(folder_path,dir_name)
        src_files_to_mosaic,out_meta = self.concat_files(files)
        
        #     # Merge function returns a single mosaic array and the transformation info
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Update the metadata
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": crs
                        })
        if save_file == True:
            with rasterio.open(f'{merged_image_path}{file_name}.tif', 'w', **out_meta) as dst:        
                dst.write(mosaic)
            
        return mosaic, out_meta
            
    def save_masks_to_folder(self,mask,file_name='mask',path = None):
        """
        Saving a mask to given path
        Parameters:
        - mask: an array to save as a mask
        - file_name(str): file name to be saved with (without extention)
        - crs: crs of the image
        - path: destination path where the mask should be saved
        Returns:
        -
        """   
        np.save(f'{path}{file_name}.npy', mask)
        
    def from_file_save_masks(self,metadata,shapefile,value=1,
                                 save_file=False,file_name=None,merged_mask_path=None):
        """
        Saving a mask to given path
        Parameters:
        - metadata: containing image width,height, transformation
        - shapefile: a shapefile containing the geometry to get the mask from
        - value: the value geometry should be filled with
        - save_file: if the files should be saved in a folder
        - file_name: name of the file without extension
        - merged_mask_path: folder where the file should be saved
        Returns:
        - generated mask
        """ 
        
        masks = self.generate_masks(image_metadata=metadata,shapefile=canal_shapefile,value=value)
        
        if save_file == True:
            assert isinstance(file_name, (str)), "file name should be given as a string"
            self.save_masks_to_folder(masks,file_name,merged_mask_path)
            
        return masks
            
    def from_file_save_wb_cnl_combined_masks(self,metadata,canal_shapefile=None,waterbody_shapefile=None,
                                             value_canal=1,value_water=2,
                                             replace_with=None,intersection_value = None,
                                             save_file=False,file_name=None,merged_mask_path=None):
        """
        Combined water and canal masks and save combine mask to given path (if save_file = True)
        Parameters:
        - metadata: containing image width,height, transformation
        - canal_shapefile,waterbody_shapefile : a shapefile containing the geometry to get the mask from
        - value_canal,value_water: the value geometry should be filled with
        - save_file: if the files should be saved in a folder
        - intersection_value(int): if the canal and waterbody intersection point should be replaces, 
                                     pass an int(which reflects the intersection point, 3 is default)
        - replace_with(int): what the value should be replace with(if None, no replacement occur)
        - file_name: name of the file without extension
        - merged_mask_path: folder where the file should be saved
        Returns:
        - generated canal_masks,waterbody_masks,combined_mask
        """ 
        
        canal_masks = self.generate_masks(image_metadata=metadata,shapefile=canal_shapefile,value=value_canal)
        waterbody_masks = self.generate_masks(image_metadata=metadata,shapefile=waterbody_shapefile,value=value_water)
        
        combined_mask = self.from_masks_save_wb_cnl_combined_masks(canal_masks,waterbody_masks, 
                                              replace_with,intersection_value,
                                              save_file,file_name,merged_mask_path)
            
        
        return canal_masks,waterbody_masks,combined_mask
    
    def from_masks_save_wb_cnl_combined_masks(self,canal_masks,waterbody_masks, 
                                              replace_with=None,intersection_value = None,
                                              save_file=False,file_name=None,merged_mask_path=None):
        """
        given water and canal masks, save combine mask to given path (if save_file = True) and return
        Parameters:
        - canal_masks,waterbody_masks : contains arrays of canal mask and waterbody masks
        - save_file: if the files should be saved in a folder
        - intersection_value(int): if the canal and waterbody intersection point should be replaces, 
                                     pass an int(which reflects the intersection point, 3 is default)
        - replace_with(int): what the value should be replace with(if None, no replacement occur)
        - file_name: name of the file without extension
        - merged_mask_path: folder where the file should be saved
        Returns:
        - combined_mask
        """         
        
        combined_mask = canal_masks + waterbody_masks
        
        if replace_with != None:
            assert isinstance(replace_with, (int)), "replace_with must be an integer value"
            assert isinstance(intersection_value, (int)), "intersection_value must be an integer value"
            combined_mask[combined_mask==intersection_value] = replace_with
            
        if save_file == True:
            assert isinstance(file_name, (int,str)), "file name should be given as a int or str"
            self.save_masks_to_folder(combined_mask,file_name,merged_mask_path)
        
        return combined_mask
#---------------------------------------------------------------------------------------------------------------------#

def get_zipfile(file_path):
    zip_file = ZipFile(file_path)
    return zip_file
    
def get_imgids_paths(zip_file):
    # Getting only the name of the images for training from the folders. For example: 
    # '1160707_2011-07-30_RE2_3A_Analytic_SR_clip.tif' is a trainable image
    image_ids = set()
    image_paths = set()
    for file in zip_file.namelist():
        # if file.startswith("REOrthoTile/"):
        p = file.split("/")
        if(len(p)>1):
            if(p[1].split('.')[1] == 'tif'):
                # image_path_pattern = f"REOrthoTile/{image_id}_3A_Analytic_.*\.tif"
                if(p[0] == 'REOrthoTile'):
                    image_id = file.split("/")[1].split('.')[0].split("_3A")[0]
                    image_ids.add(image_id)
                    image_path_pattern = f"REOrthoTile/{image_id}_3A_Analytic_.*\.tif"
                else: #2020 - 2023 have different folder and file names
                    image_id = file.split("/")[1].split('.')[0].split("_3B")[0]
                    image_ids.add(image_id)
                    image_path_pattern = f"PSScene/{image_id}_3B_Analytic.*\.tif"

                if re.match(image_path_pattern,file):
                    image_paths.add(file)    
    return list(image_ids),list(image_paths)


def get_image_file(zip_file,image_path):
    #getting the images to save them
    
    img = []
    with zip_file.open(image_path) as fp:
        data = io.BytesIO(fp.read())
        with rasterio.open(data) as src:
            # All bands have same metadata and gcps
            metadata = src.meta.copy()
            img = src.read()
        img = np.array(img)
    return img,metadata

def save_images(image_id, metadata, out_path):
    with rasterio.open(f"{out_path}/{image_id}.tif", "w", **metadata) as dest:
        dest.write(img)
    

def plot_bands(img):
    # Iterate and display each band
    for band_number in range(1, 6):  # Assuming there are 5 bands
        plt.figure()
        plt.imshow(img[band_number - 1], cmap='viridis')
        plt.title(f'Band {band_number}')
        plt.colorbar()
        plt.show()      
        

def save_masks(image_metadata,out_path,waterways, k=(0,0)):
    # Create a blank array
    line_array = np.zeros((image_metadata['height'], image_metadata['width']), dtype=np.uint8)

    # Rasterize the line strings
    shapes = ((geom, 1) for geom in waterways.geometry)
    mask = rasterize(shapes=shapes, out_shape=line_array.shape, transform=image_metadata['transform'])
    kernel = np.ones(k, np.uint8)

    # Perform dilation
    # mask = cv2.dilate(burned, kernel, iterations=1)
    
    np.save(out_path, mask)
    
##used
def prep_data(image,channel=3):
    image = np.transpose(image, (1,2,0))
    rgb = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
    if channel == 4:
        rgb = np.dstack((rgb, image[:, :, 3]))
    return rgb 
    
def clip_satellite_image(image, lower_percentile=2, upper_percentile=98):
    min_val = np.percentile(image, lower_percentile)
    max_val = np.percentile(image, upper_percentile)
    clipped_image = np.clip(image, min_val, max_val)
    return clipped_image

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) / (max_val - min_val))*255
    return normalized_image.astype(np.uint8)

def normalize_satellite_image(image, lower_percentile=2, upper_percentile=98):
    # Clip the image
    clipped_image = clip_satellite_image(image, lower_percentile, upper_percentile)

    # Normalize to 0-255
    normalized_image = normalize_image(clipped_image)

    return normalized_image

def repro(src_file):
    dst_file = '/scratch/gza5dr/output_1.tif'

    # Define source and destination CRS
    src_crs = 'EPSG:32610'  # Example: WGS 84
    dst_crs = 'EPSG:32611'  # Example: Web Mercator

    with rasterio.open(src_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        src = rasterio.open(dst_file)
            
        return src



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

def directly_connected_1s(full_mask,patches,patch_val):
    print("Step_2,directly_connected_1s")
    not_connected_patches = patches.copy()
    start = time.time()
    array = patches.copy()

    p_i,p_j,p_rows,p_columns = array.shape
    directly_connected_patches = set()

    directions = [(0,1),(0,-1),(1,0),(-1,0),
                  (-1,-1),(1,-1),(1,1),(-1,1)]

    for i,j,x,y in patch_val:
        for dx,dy in directions:
            nx, ny = x+dx, y+dy
            n_i,n_j= i,j

            if 0>nx or nx>=p_rows:
                n_i = i+dx
                nx = 511 if nx == -1 else 0 if nx == 512 else nx

            if 0>ny or ny>=p_columns:
                n_j = j+dy
                ny = 511 if ny == -1 else 0 if ny == 512 else ny

            X_L = (nx) + (n_i*512)
            Y_L = (ny) + (n_j*512)

            if 0<=n_i<p_i and 0<=n_j<p_j and array[n_i,n_j,nx,ny]==1:
                assert(full_mask[X_L,Y_L] == patches[n_i,n_j,nx,ny])
                directly_connected_patches.add((n_i,n_j,nx,ny))
                not_connected_patches[n_i,n_j,nx,ny] = 3
    end = time.time()
    print(end-start)
    
    return directly_connected_patches,not_connected_patches


def bfs(array, start, dierectly_connected,not_connected_patches):
    print("Step_3,bfs")
    p_i,p_j,p_rows,p_columns = array.shape
    directions = [(0,1),(0,-1),(1,0),(-1,0),
                  (-1,-1),(1,-1),(1,1),(-1,1)]
    visited = set(start)
    queue = list(start)
    
    while queue:
        i,j,x,y = queue.pop(0)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            n_i,n_j= i,j
        
            if 0>nx or nx>=p_rows:
                n_i = i+dx
                nx = 511 if nx == -1 else 0 if nx == 512 else nx

            if 0>ny or ny>=p_columns:
                n_j = j+dy
                ny = 511 if ny == -1 else 0 if ny == 512 else ny
                
            if 0<=n_i<p_i and 0<=n_j<p_j and (n_i,n_j,nx, ny) not in visited and array[n_i,n_j,nx,ny]==1:
                visited.add((n_i,n_j,nx, ny))
                queue.append((n_i,n_j,nx, ny))
                not_connected_patches[n_i,n_j,nx, ny] = 3
    return visited,not_connected_patches



