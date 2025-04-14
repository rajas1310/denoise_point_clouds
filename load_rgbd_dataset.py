import open3d as o3d
import cv2

import numpy as np
import torch
import os
from noise import pnoise2  # 2D noise

pose_data = np.loadtxt("datasets/rgbd-scenes-v2/pc/01.pose")

def get_rt_matrix(pose : list):
    assert len(pose) == 7, "Pose should contain 7 elements"
    # Convert to a 4x4 transformation matrix
    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(pose[3:])
    rt_matrix[:3, 3] = pose[:3]
    return np.asarray(rt_matrix)


# # Load the PLY file
# pcd = o3d.io.read_point_cloud("datasets/rgbd-scenes-v2/pc/01.ply")
# # print(np.asarray(pcd.colors))

# # # Apply the pose transformation
# # transformed_pcd = pcd.transform(pose)

# print(dir(pcd))
# print(np.asarray(pcd.points).shape)
# print(pcd.dimension)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])

def noise_adder(depth_map):
    # Get dimensions
    height, width = depth_map.shape

    # Create meshgrid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Random parameters for sinusoidal wave
    freq = np.random.uniform(0.07, 0.1)  # Frequency
    amplitude = np.random.uniform(0.1, 0.9)  # Amplitude of the wave
    phase = np.random.uniform(0, 2 * np.pi)  # Phase shift

    # Create horizontal sinusoidal wave (you can try vertical or diagonal too)
    sin_noise = amplitude * abs(np.sin(2 * np.pi * freq * x_grid + phase) +  np.sin(2 * np.pi * freq * y_grid + phase))

    noisy_depth = depth_map + sin_noise

    return noisy_depth.astype(np.float32) , sin_noise

def ddpm_noise (x, t):
    noise_steps=1000
    beta_start=1e-4
    beta_end=0.02
    beta = torch.linspace(beta_start, beta_end, noise_steps)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)


    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])
    Ɛ = torch.randn_like(x)
    return np.array(sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ), Ɛ



def generate_perlin_noise_2d(depth_map, seed=0):
    scale = np.random.uniform(0.02, 0.1)  # Scale of the noise
    amplitude = np.random.uniform(0.5, 1.0)  # Amplitude of the wave
    print("Scale:", scale, "Amplitude:", amplitude)
    height, width = depth_map.shape
    noise_map = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            noise_val = pnoise2(x * scale, y * scale, repeatx=width, repeaty=height, base=seed)
            noise_map[y, x] = amplitude * noise_val
    
    return (noise_map + depth_map).astype(np.float32), noise_map

class RGBDFrameLoader:
    def __init__(self, color_video_path, depth_frames_path):
        self.color_video_path = color_video_path

        self.cap_color = cv2.VideoCapture(color_video_path)
        self.depth_frame_array = np.load(depth_frames_path)

        print("Color video FPS:", self.cap_color.get(cv2.CAP_PROP_FPS), "\tTotal frames:", self.cap_color.get(cv2.CAP_PROP_FRAME_COUNT))
        # Depth array should be uint16 or float32
        print("Depth array type:", self.depth_frame_array.dtype, "\tMin depth:", self.depth_frame_array.min(), "\tMax depth:", self.depth_frame_array.max())  # Check depth range

        self.frame_index = -1
    
    def load_next_frames(self, smooth_depth=True):
        self.frame_index += 1
        ret, color_frame = self.cap_color.read()
        if ret == False:
            return None, None
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        color_frame = o3d.geometry.Image(color_frame)

        
        depth_map = self.depth_frame_array[self.frame_index]
        if smooth_depth:
            depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 11, sigmaColor=75, sigmaSpace=75)
        depth_map = o3d.geometry.Image(depth_map)
        
        return color_frame, depth_map
    
    def get_current_frame_index(self):
        return self.frame_index
    
    def get_frame_count(self):
        return int(self.cap_color.get(cv2.CAP_PROP_FRAME_COUNT))

    def load_next_rgbd(self, smooth_depth=True):
        color_frame, depth_map = self.load_next_frames(smooth_depth)
        if color_frame is None:
            return None
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_frame, depth_map, convert_rgb_to_intensity=False)
        return rgbd_image

color_frame = "datasets/rgbd-scenes-v2/imgs/scene_01/00000-color.png"
depth_map = "datasets/rgbd-scenes-v2/imgs/scene_01/00000-depth.png"
color_frame = o3d.io.read_image(color_frame)
depth_map = o3d.io.read_image(depth_map)

depth_array = np.asarray(depth_map).astype(np.float32) / 1000
print("Depth map stats - Min:", depth_array.min(), "Max:", depth_array.max(), "Shape:", depth_array.shape)
depth_map = o3d.geometry.Image(depth_array)

depth_viz = np.clip(depth_map, 0, 255).astype(np.uint8)

cv2.imshow("Depth Map", np.asarray(depth_viz))

# noisy_depth, noise = noise_adder(depth_array)
# noisy_depth, noise = ddpm_noise(torch.tensor(noisy_depth), 15)
noisy_depth, noise = generate_perlin_noise_2d(depth_array)
# print(np.asarray(depth_map))

noisy_depth_viz = np.clip(noisy_depth, 0, 255).astype(np.uint8)
cv2.imshow("Noisy", np.asarray(noise))
cv2.imshow("Noisy Depth Map", np.asarray(noisy_depth_viz))
cv2.waitKey(0)

noisy_depth = o3d.geometry.Image(noisy_depth)

camera_pose = get_rt_matrix(pose_data[0])

# print("Camera pose:\n", camera_pose)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_frame, noisy_depth, convert_rgb_to_intensity=False)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, camera_pose)

flip_transform = [[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]]
pcd.transform(flip_transform)

# Visualize the RGBD image
o3d.visualization.draw_geometries([pcd])
