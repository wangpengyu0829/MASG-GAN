import os
import cv2
import glob
import torch
import random
from PIL import Image


class get_train(torch.utils.data.Dataset):
    def __init__(self, clean_root, haze_root, transforms):
        self.clean_root = clean_root
        self.haze_root = haze_root
         self.image_name_list = glob.glob(os.path.join(self.clean_root, '*.png'))  # 
        self.file_list = []
        self.get_image_pair_list()                                                
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        clean_image_name, haze_image_name = self.file_list[item]
        clean_image   = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(clean_image_name),   cv2.COLOR_BGR2RGB)))
        haze_image    = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(haze_image_name),    cv2.COLOR_BGR2RGB)))
        return clean_image, haze_image
    
    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            key = image.split(self.clean_root)[-1]                                # 
            self.file_list.append([self.clean_root+key, self.haze_root+key])
        random.shuffle(self.file_list)



class get_test(torch.utils.data.Dataset):
    def __init__(self, clean_root, haze_root, transforms):
        self.clean_root = clean_root
        self.haze_root = haze_root
        self.image_name_list = glob.glob(os.path.join(self.clean_root, '*.png'))  
        self.file_list = []
        self.get_image_pair_list()                                                
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        clean_image_name, haze_image_name = self.file_list[item]
        clean_image   = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(clean_image_name),   cv2.COLOR_BGR2RGB)))
        haze_image    = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(haze_image_name),    cv2.COLOR_BGR2RGB)))
        return haze_image, clean_image
    
    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            key = image.split(self.clean_root)[-1]                                
            self.file_list.append([self.clean_root+key, self.haze_root+key])



class get_val(torch.utils.data.Dataset):
    def __init__(self, val_rgb_path, val_sp_root, transforms):
        self.val_rgb_path = val_rgb_path
        self.val_sp_root = val_sp_root
        self.image_name_list = glob.glob(os.path.join(self.val_rgb_path, '*.png')) 
        self.file_list = []
        self.get_image_pair_list()                                                
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        val_rgb_image_name, val_sp_image_name = self.file_list[item]
        val_rgb_image   = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(val_rgb_image_name),   cv2.COLOR_BGR2RGB)))

        return val_sp_image_name, val_rgb_image
    
    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            key = image.split(self.val_rgb_path)[-1]                                # 示例 \4_3.png
            self.file_list.append([self.val_rgb_path+key, self.val_sp_root+key])





