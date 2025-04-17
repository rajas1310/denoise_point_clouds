import open3d as o3d
import cv2

import numpy as np
import torch
import os, glob
from noise import pnoise2  # 2D noise
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import albumentations as A
from albumentations.pytorch import ToTensorV2

# data_sample(5) : rgb, depth_map, noisy_depth_map, noise, segmentation_mask

class ImageLoader(Dataset):
    def __init__(self, data, img_sz, transform=None):
        # self.data_dir = data_dir
        self.transform = transform
        self.data = data
        self.img_sz = img_sz

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_generator = SamAutomaticMaskGenerator(sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth").to(device))

    def get_segmentation_mask(self, image):
        # Generate masks
        masks = self.mask_generator.generate(image)

        # Initialize label map (H x W) with 0s
        segmentation_map = np.zeros(image.shape[:2], dtype=np.uint16)

        # Assign a unique label to each mask
        for i, mask in enumerate(masks):
            segmentation_map[mask["segmentation"]] = i + 1  # Label starts from 1

        return segmentation_map

    def get_noisy_depth_map(self, depth_map, seed=0):
        scale = np.random.uniform(0.02, 0.1)  # Scale of the noise
        amplitude = np.random.uniform(0.5, 1.0)  # Amplitude of the wave
        # print("Scale:", scale, "Amplitude:", amplitude)
        height, width = depth_map.shape
        noise_map = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                noise_val = pnoise2(x * scale, y * scale, repeatx=width, repeaty=height, base=seed)
                noise_map[y, x] = amplitude * noise_val
        
        return (noise_map + depth_map).astype(np.float32), noise_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        # rgb = Image.open(data['rgb']).convert("RGB")
        rgb = cv2.imread(data['rgb'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_map = cv2.imread(data['depth_map'], cv2.IMREAD_UNCHANGED) # should be a float32 array
        depth_map = depth_map.astype(np.float32)
        
        segmentation_mask = self.get_segmentation_mask(rgb)

        noisy_depth_map, noise = self.get_noisy_depth_map(depth_map)

        if self.transform: # todo transformation should be applied to the whole RGB, depth, segmap
            self.transforms_all = A.Compose([
                A.Resize(self.img_sz, self.img_sz),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ], additional_targets={'depth': 'image', 'noisy_depth': 'image', 'noise' : 'image', 'mask': 'mask'})

            # self.transforms_inputs_only = A.Compose([
            #     A.Normalize(),
            # ], additional_targets={'noisy_depth': 'image', 'mask': 'mask'})
            
            augmented_1 = self.transforms_all(image=rgb, depth=depth_map, noisy_depth=noisy_depth_map, noise=noise, mask=segmentation_mask)
            rgb = augmented_1['image']
            depth_map = augmented_1['depth']
            noisy_depth_map = augmented_1['noisy_depth']
            noise = augmented_1['noise']
            segmentation_mask = augmented_1['mask']

            # augmented_2 = self.transforms_inputs_only(image=rgb, depth=depth_map, noisy_depth=noisy_depth_map, noise=noise, mask=segmentation_mask)
            # rgb = augmented_2['image']
            # noisy_depth_map = augmented_2['noisy_depth']
            # segmentation_mask = augmented_2['mask']
        
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float()   # Ensure float type if needed
        if isinstance(noisy_depth_map, np.ndarray):
            noisy_depth_map = torch.from_numpy(noisy_depth_map).float()
        if isinstance(segmentation_mask, np.ndarray):
            segmentation_mask = torch.from_numpy(segmentation_mask).to(torch.uint16)

        # Convert RGB from HWC to CHW if needed.
        if rgb.dim() == 3 and rgb.shape[0] != 3:
            # Assuming shape is HWC: [H, W, 3]
            rgb = rgb.permute(2, 0, 1)

        # Add a channel dimension to depth maps if needed.
        if noisy_depth_map.dim() == 2:
            noisy_depth_map = noisy_depth_map.unsqueeze(0)
        if segmentation_mask.dim() == 2:
            segmentation_mask = segmentation_mask.unsqueeze(0)

        #concat rgb noisy_depth_map segmentation_mask
        input_tensor = torch.cat((rgb, noisy_depth_map, segmentation_mask), dim=0)
        return input_tensor, {'depth': depth_map, 'noise':noise, 'noisy_depth': noisy_depth_map}
    
class DPCDataset():
    def __init__(self, data_dir : str, img_sz, transform : bool = True):
        self.data_dir = data_dir
        self.img_sz = img_sz
        self.rgb_list, self.depth_list = self.get_rgbd_list()
        self.transform = transform
        self.data = [{'rgb': rgb_img, 'depth_map': depth_img} for rgb_img, depth_img in zip(self.rgb_list, self.depth_list)]

    def get_rgbd_list(self):
        rgb_list = glob.glob(os.path.join(self.data_dir, 'imgs/*/*-color.png'))
        depth_list = glob.glob(os.path.join(self.data_dir, 'imgs/*/*-depth.png'))
        rgb_list.sort()
        depth_list.sort()
        # print(len(rgb_list), "\n", rgb_list[:10])
        # print(len(depth_list), "\n", depth_list[:10])
        return rgb_list, depth_list
        
    
    def get_dataset(self):
        return ImageLoader(self.data, self.img_sz, transform=self.transform)

# obj = DPCDataset("datasets/rgbd-scenes-v2")
# trainset = obj.get_dataset()


# print(trainset[0][0].shape)
# print(trainset[0][1]['depth'].shape)
# print(trainset[0][1]['noise'].shape)