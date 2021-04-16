
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from imageio import imread
import cv2
from PIL import Image

def is_img(x):
    if x.endswith('.png') and not(x.startswith('._')):   
        return True
    else:
        return False
    
def is_img2(x):
    if x.endswith('.png') and not(x.startswith('._')): 
        return True
    else:
        return False

def _np2Tensor(img):  
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float() 
    return tensor


'''读取训练数据'''
class get_train(data.Dataset):
    def __init__(self, rgb_path, sp_path, gt_path, patch_size, transforms, isTrain):
        self.isTrain = isTrain
        self.transforms = transforms
        self.patch_size = patch_size   
        self.rgb_path = rgb_path 
        self.sp_path  = sp_path 
        self.gt_path  = gt_path 
        self._set_filesystem(self.rgb_path, self.sp_path, self.gt_path)   
        self.images_rgb, self.images_sp, self.images_gt = self._scan()            
        self.repeat = 1
        
    '''打印路径'''        
    def _set_filesystem(self, dir_rgb, dir_sp, dir_gt):
        self.dir_rgb = dir_rgb
        self.dir_sp  = dir_sp
        self.dir_gt  = dir_gt
        self.ext = '.png'                   
        print('********* {}: dir_rgb and dir_sp and dir_gt ******'.format(self.isTrain))
        print(self.dir_rgb)
        print(self.dir_sp)
        print(self.dir_gt)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_rgb = sorted([os.path.join(self.dir_rgb, x) for x in os.listdir(self.dir_rgb) if is_img(x)])  
        random.shuffle(list_rgb)
        list_sp = [os.path.splitext(x)[0]+'.png' for x in list_rgb]         
        list_sp = [os.path.join(self.dir_sp, os.path.split(x)[-1]) for x in list_sp]  
        list_gt = [os.path.join(self.dir_gt, os.path.split(x)[-1]) for x in list_sp]  
        return list_rgb, list_sp, list_gt                                             

    def __getitem__(self, idx):
        img_rgb, img_sp, img_gt, filename_rgb, filename_sp, filename_gt = self._load_file(idx)                 
        if self.isTrain:                                            
            x = random.randint(0, img_rgb.size(1) - self.patch_size)  
            y = random.randint(0, img_rgb.size(2) - self.patch_size)
            img_rgb = img_rgb[:, x : x+self.patch_size, y : y+self.patch_size]   
            img_sp  = img_sp[:, x : x+self.patch_size, y : y+self.patch_size]
            img_gt  = img_gt[:, x : x+self.patch_size, y : y+self.patch_size]
        return img_rgb, img_sp, img_gt

    def __len__(self):
        if self.isTrain:
            return len(self.images_rgb) * self.repeat
        else:
            return len(self.images_rgb)
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        if self.isTrain:
            return idx % len(self.images_rgb)   
        else:
            return idx

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    
        file_rgb = self.images_rgb[idx]   
        file_sp  = self.images_sp[idx]   
        file_gt  = self.images_gt[idx]   
        img_rgb = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_rgb),   cv2.COLOR_BGR2RGB)))
        img_sp  = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_sp),   cv2.COLOR_BGR2GRAY)))
        img_gt  = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_gt),   cv2.COLOR_BGR2GRAY)))
        filename_rgb = os.path.splitext(os.path.split(file_rgb)[-1])[0]   
        filename_sp = os.path.splitext(os.path.split(file_sp)[-1])[0]     
        filename_gt = os.path.splitext(os.path.split(file_gt)[-1])[0]     
        return img_rgb, img_sp, img_gt, filename_rgb, filename_sp, filename_gt  

'''读取测试数据'''
class get_val(data.Dataset):
    def __init__(self, rgb_path, sp_path, gt_path, patch_size, transforms, isTrain):
        self.isTrain = isTrain
        self.transforms = transforms
        self.patch_size = patch_size   
        self.rgb_path = rgb_path 
        self.sp_path  = sp_path 
        self.gt_path  = gt_path 
        self._set_filesystem(self.rgb_path, self.sp_path, self.gt_path)   
        self.images_rgb, self.images_sp, self.images_gt = self._scan()           
        self.repeat = 1
        
    '''打印路径'''        
    def _set_filesystem(self, dir_rgb, dir_sp, dir_gt):
        self.dir_rgb = dir_rgb
        self.dir_sp  = dir_sp
        self.dir_gt  = dir_gt
        self.ext = '.png'                     
        print('********* {}: dir_rgb and dir_sp and dir_gt ******'.format(self.isTrain))
        print(self.dir_rgb)
        print(self.dir_sp)
        print(self.dir_gt)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_rgb = sorted([os.path.join(self.dir_rgb, x) for x in os.listdir(self.dir_rgb) if is_img(x)])  
        random.shuffle(list_rgb)
        list_sp = [os.path.splitext(x)[0]+'.png' for x in list_rgb]          
        list_sp = [os.path.join(self.dir_sp, os.path.split(x)[-1]) for x in list_sp]  
        list_gt = [os.path.join(self.dir_gt, os.path.split(x)[-1]) for x in list_sp]  
        return list_rgb, list_sp, list_gt                                            

    def __getitem__(self, idx):
        img_rgb, img_sp, img_gt, filename_rgb, filename_sp, filename_gt = self._load_file(idx)               
        if self.isTrain:                                            # 
            x = random.randint(0, img_rgb.size(1) - self.patch_size)  # 
            y = random.randint(0, img_rgb.size(2) - self.patch_size)
            img_rgb = img_rgb[:, x : x+self.patch_size, y : y+self.patch_size]   # 随机裁剪
            img_sp  = img_sp[:, x : x+self.patch_size, y : y+self.patch_size]
            img_gt  = img_gt[:, x : x+self.patch_size, y : y+self.patch_size]
        return img_rgb, img_sp, img_gt

    def __len__(self):
        if self.isTrain:
            return len(self.images_rgb) * self.repeat
        else:
            return len(self.images_rgb)
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        if self.isTrain:
            return idx % len(self.images_rgb)   # 余数
        else:
            return idx

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    
        file_rgb = self.images_rgb[idx]   
        file_sp  = self.images_sp[idx]   
        file_gt  = self.images_gt[idx]   
        img_rgb = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_rgb),   cv2.COLOR_BGR2RGB)))
#        img_sp  = self.transforms(imread(file_sp))
#        img_gt  = self.transforms(imread(file_gt))
        img_sp  = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_sp),   cv2.COLOR_BGR2GRAY)))
        img_gt  = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_gt),   cv2.COLOR_BGR2GRAY)))
        filename_rgb = os.path.splitext(os.path.split(file_rgb)[-1])[0]    
        filename_sp = os.path.splitext(os.path.split(file_sp)[-1])[0]      
        filename_gt = os.path.splitext(os.path.split(file_gt)[-1])[0]      
        return img_rgb, img_sp, img_gt, filename_rgb, filename_sp, filename_gt  

    
    

