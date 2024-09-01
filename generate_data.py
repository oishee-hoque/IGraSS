from path_utils import *
import numpy as np
import os
import shutil






def generate_dataset(image_path, gt_path, out_path, subfolder_prefix = "main_gt", p_s = 512, dilation=False, k=0):
    imagePatches = []
    maskPatches = []
    full_masks = []
    patch_vals = []
    directly_connected_patches =[]
    not_connected_patches = []
    reachable_ones_patches = []
    years = [2020,2021,2022,2023]
    dh = Data_Handler()
    masks = []
    p_s = 512

    for i in range(0,4):
        print(f'years: {years[i]}')

        
        img = np.load(f'{image_path}/common_normalized_{years[i]}.npy') 
        # img = np.load(f'{image_path}/train_img_{years[i]}.npy') 
        temp = np.load(f'/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/common_satellite{years[i]}.npy')
        area = int(temp.shape[0]*.20)
        
        if gt_path == None:
            mask = np.load(f'/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/common_satellite{years[i]}.npy')
        else: 
            mask = np.load(gt_path)

        
        print(dilation)
        
        if dilation == True:
            assert k != 0, "k should be greater than 0"
            mask = dh.perform_dialation(mask,kernel_width=(k,k),iterations=1)
        masks.append(mask)

        cut_1,cut_2 = img.shape[0]%p_s,img.shape[1]%p_s
        img = img[cut_1:,cut_2:,:]
        temp = temp[cut_1:,cut_2:]
        
        cut_1,cut_2 = mask.shape[0]%p_s,mask.shape[1]%p_s
        mask = mask[cut_1:,cut_2:]
        mask[:,:area] = temp[:,:area]


        image_patches = patchify(img,(p_s,p_s,img.shape[2]),step=p_s)
        mask_patches = patchify(mask,(p_s,p_s),step=p_s)
        print(img.shape)
        print(f'Img Patch Shape:{image_patches.shape}')
        print(f'Mask Patch Shape: {mask_patches.shape}')        

        reconstructed_image = unpatchify(image_patches, img.shape)
        reconstructed_mask = unpatchify(mask_patches, mask.shape)
        assert (reconstructed_image == img).all()
        assert (reconstructed_mask == mask).all()
        del reconstructed_image,reconstructed_mask

        imagePatches.append(image_patches)
        maskPatches.append(mask_patches)
        full_masks.append(mask)



    print("for",p_s)

    image_path = f'{out_path}{subfolder_prefix}_{k}dil_images_{p_s}'
    mask_path = f'{out_path}{subfolder_prefix}_{k}dil_masks_{p_s}'

    if os.path.exists(image_path):
        # os.rmdir(path)
        shutil.rmtree(image_path)

    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)  

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)



    for i in range(len(maskPatches)):
        print(i)
        a,b,c,d = maskPatches[i].shape
        cut = a/p_s
        cnt = 0
        for p1 in range(a):
            for p2 in range(b):
                # im = normalize_satellite_image(imagePatches[i][p1][p2][0])
                # im= Image.fromarray(im,'RGB')
                im = Image.fromarray(imagePatches[i][p1][p2][0],'RGB')
                msk = np.zeros(((p_s,p_s,1)))
                msk[:,:,0] = maskPatches[i][p1][p2]
                msk[msk==2] = 0 #this mask just has zeros and 1s, making the water sources 0
                msk[msk==3] = 0


                ##Region Check
                Y_L = p2*512
                r1 = int(np.ceil(masks[0].shape[1]*.19))
                r2 = int(np.ceil(masks[0].shape[1]*.01))
                r3 = int(np.ceil(masks[0].shape[1]*.8))

                if Y_L< r1:
                    region = 'test'
                elif Y_L< r1+r2:
                    region = 'test'
                else:
                    region = 'train'

                num_zeros = (np.array(im) == 0).sum()
                c = p_s*p_s*.005
                z = p_s*p_s*.3
                if (np.sum(msk)<c):
                    continue
                if num_zeros>=z:
                    continue

    #             if np.any(np.all(d_1s == [[p1,p2]], axis=1)):
    #                 cnt += 1
    #                 disconnected_patch_filenames.append(f'/scratch/gza5dr/NHDShape/procssed_data/mask_patches/{years[i]}_r{region}_{p1}_{p2}.npy')

                im.save(f"{image_path}/{years[i]}_r{region}_{p1}_{p2}.png")
                np.save(f'{mask_path}/{years[i]}_r{region}_{p1}_{p2}.npy', msk)
    
    return image_path, mask_path


def generate_dataset_set(image_path, gt_path, out_path, subfolder_prefix = "main_gt", p_s = 512, dilation=False, k=0):
    imagePatches = []
    maskPatches = []
    full_masks = []
    patch_vals = []
    directly_connected_patches =[]
    not_connected_patches = []
    reachable_ones_patches = []
    years = [2020,2021,2022,2023]
    dh = Data_Handler()
    masks = []
    p_s = 512

    for i in range(0,4):
        print(f'years: {years[i]}')

        
        # img = np.load(f'{image_path}/common_normalized_{years[i]}.npy') 
        img = np.load(f'{image_path}/train_img_{years[i]}.npy') 
        
        if gt_path == None:
            mask = np.load(f'/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/masks/train_mask_{years[i]}.npy')
        else: 
            mask = np.load(gt_path)

        
        print(dilation)
        
        if dilation == True:
            assert k != 0, "k should be greater than 0"
            mask = dh.perform_dialation(mask,kernel_width=(k,k),iterations=1)
        masks.append(mask)

        cut_1,cut_2 = img.shape[0]%p_s,img.shape[1]%p_s
        img = img[cut_1:,cut_2:,:]
        
        cut_1,cut_2 = mask.shape[0]%p_s,mask.shape[1]%p_s
        mask = mask[cut_1:,cut_2:]


        image_patches = patchify(img,(p_s,p_s,img.shape[2]),step=p_s)
        mask_patches = patchify(mask,(p_s,p_s),step=p_s)
        print(img.shape)
        print(f'Img Patch Shape:{image_patches.shape}')
        print(f'Mask Patch Shape: {mask_patches.shape}')        

        reconstructed_image = unpatchify(image_patches, img.shape)
        reconstructed_mask = unpatchify(mask_patches, mask.shape)
        assert (reconstructed_image == img).all()
        assert (reconstructed_mask == mask).all()
        del reconstructed_image,reconstructed_mask

        imagePatches.append(image_patches)
        maskPatches.append(mask_patches)
        full_masks.append(mask)



    print("for",p_s)

    image_path = f'{out_path}{subfolder_prefix}_{k}dil_images_{p_s}'
    mask_path = f'{out_path}{subfolder_prefix}_{k}dil_masks_{p_s}'

    if os.path.exists(image_path):
        # os.rmdir(path)
        shutil.rmtree(image_path)

    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)  

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)



    for i in range(len(maskPatches)):
        print(i)
        a,b,c,d = maskPatches[i].shape
        cut = a/p_s
        cnt = 0
        for p1 in range(a):
            for p2 in range(b):
                # im = normalize_satellite_image(imagePatches[i][p1][p2][0])
                # im= Image.fromarray(im,'RGB')
                im = Image.fromarray(imagePatches[i][p1][p2][0],'RGB')
                msk = np.zeros(((p_s,p_s,1)))
                msk[:,:,0] = maskPatches[i][p1][p2]
                msk[msk==2] = 0 #this mask just has zeros and 1s, making the water sources 0
                msk[msk==3] = 0


                ##Region Check
                Y_L = p2*512
                r1 = int(np.ceil(masks[0].shape[1]*.19))
                r2 = int(np.ceil(masks[0].shape[1]*.01))
                r3 = int(np.ceil(masks[0].shape[1]*.8))

                if Y_L< r1:
                    region = 'train'
                elif Y_L< r1+r2:
                    region = 'train'
                else:
                    region = 'train'

                num_zeros = (np.array(im) == 0).sum()
                c = p_s*p_s*.005
                z = p_s*p_s*.3
                if (np.sum(msk)<c):
                    continue
                if num_zeros>=z:
                    continue

    #             if np.any(np.all(d_1s == [[p1,p2]], axis=1)):
    #                 cnt += 1
    #                 disconnected_patch_filenames.append(f'/scratch/gza5dr/NHDShape/procssed_data/mask_patches/{years[i]}_r{region}_{p1}_{p2}.npy')

                im.save(f"{image_path}/{years[i]}_r{region}_{p1}_{p2}.png")
                np.save(f'{mask_path}/{years[i]}_r{region}_{p1}_{p2}.npy', msk)
    
    return image_path, mask_path


