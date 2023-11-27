import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from PIL import Image
import numpy as np
import sys
sys.path.append('../../')
from Data_preparation.heatmaps import coord2Heatmap
from IPython.display import clear_output
import time


class LandmarkDataset(Dataset):
    def __init__(self, image_paths, cam_points, visibility_values, image_transform, heatmap_transform):
        self.image_paths = image_paths
        self.cam_points = cam_points
        self.visibility_values = visibility_values

        self.image_transform = image_transform
        self.heatmap_transform = heatmap_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        image = self.image_transform(image)

        heatmaps = coord2Heatmap(self.cam_points[idx], self.visibility_values[idx])
        heatmaps = torch.stack([self.heatmap_transform(Image.fromarray(heatmaps[i])) for i in range(heatmaps.shape[0])])

        visibility = torch.tensor(self.visibility_values[idx])

        return image, heatmaps, visibility
    
def select_data(dataset, n_landmarks=9):
    data = pd.read_csv(dataset)
    LDM_pos = []
    for i in range(n_landmarks):
        LDM_pos.append(f'LDM{i}x')
        LDM_pos.append(f'LDM{i}y')
        
    Vp = [f'VP{i}' for i in range(n_landmarks)]

    data = data[data['image_nr'] < 800]

    image_paths = data['image_path'].values
    cam_points = data[LDM_pos].values.reshape(-1, n_landmarks, 2)
    visibility = data[Vp].values

    return image_paths, cam_points, visibility

def load_data(dataset, n_landmarks=9, batch_size=4, image_width=512, image_height=512):
    for data in dataset:
        img, cam_p, vis = select_data(data, n_landmarks)
        if data == dataset[0]:
            image_path = img
            cam_points = cam_p
            visibility = vis
        else:
            image_paths = np.concatenate((image_path, img))
            cam_points = np.concatenate((cam_points, cam_p))
            visibility = np.concatenate((visibility, vis))

    input_transform = Compose([
        Resize((image_width, image_height)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    target_transform = Compose([
        Resize((image_width, image_height)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = LandmarkDataset(image_paths, cam_points, visibility, input_transform, target_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
