import cv2
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from guided_diffusion.guided_diffusion.image_datasets import _list_image_files_recursively


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform


class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_spatial: torch.Tensor, 
        y_data: torch.Tensor,
        X_spectral:torch.Tensor
    ): 
        self.X_spatial = X_spatial
        self.y_data = y_data
        self.X_spectral =X_spectral

    def __getitem__(self, index):
      return self.X_spatial[index], self.y_data[index],self.X_spectral[index]

    def __len__(self):
        return len(self.X_spatial)


class ImageLabelDataset(Dataset):
   # modify on JUne 21, seperating data_dir and label_dir, rgb and hyperspectral cubes in the same folder dir
   # while labels in another dir
    #modify on April 30, add train_data pathches
    ''' 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None,
        image_sort = True

    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)

        
        if (image_sort):
          self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]
      
        """
        # label will be in a different data dir 
        self.label_paths = [
          os.path.join(label_dir, os.path.basename(image_path).split('.')[0] + '.npy')
            for image_path in self.image_paths]

        """
        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths]
        
        #data_cube is dimension (64,64,144)
        self.data_cube_path = [
            '.'.join(image_path.split('.')[:-1] + ['npz'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        #pdb.set_trace()
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert pil_image.size[0] == pil_image.size[1], \
               f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        data_cube_path = self.data_cube_path[idx]
        with np.load(data_cube_path) as data:
          data_cube = data['arr1']

        tensor_label = torch.from_numpy(label)
        tensor_data_cube = torch.from_numpy(data_cube)
        return tensor_image, tensor_label,tensor_data_cube


class InMemoryImageLabelDataset(Dataset):
    ''' 

    Same as ImageLabelDataset but images and labels are already loaded into RAM.
    It handles DDPM/GAN-produced datasets and is used to train DeepLabV3. 

    :param images: np.array of image samples [num_images, H, W, 3].
    :param labels: np.array of correspoding masks [num_images, H, W].
    :param resolution: image and mask output resolusion.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
            self,
            images: np.ndarray, 
            labels: np.ndarray,
            resolution=256,
            transform=None
    ):
        super().__init__()
        assert  len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        assert image.size[0] == image.size[1], \
               f"Only square images are supported: ({image.size[0]}, {image.size[1]})"

        tensor_image = self.transform(image)
        label = self.labels[idx]
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label




