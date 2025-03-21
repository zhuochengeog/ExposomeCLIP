import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform  
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transform(resize_crop_size = 224,
                  mean = [0.4139, 0.4341, 0.3482, 0.5263],
                  std = [0.0010, 0.0010, 0.0013, 0.0013]
                  ):

    augmentation = A.Compose(
        [
            A.RandomResizedCrop(height=resize_crop_size, width=resize_crop_size),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GaussianBlur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    def transform(sample):
        image = sample["image"].numpy().transpose(1,2,0)
        point = sample["point"]

        image = augmentation(image=image)["image"]
        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_s2_train_transform(resize_crop_size = 224):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"] / 10000.0
        point = sample["point"]
        image = torch.tensor(image)
        image = augmentation(image)
        point = coordinate_jitter(point)
        return dict(image=image, point=point)

    return transform

def get_pretrained_s2_train_transform(resize_crop_size = 224):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"] / 10000.0
        point = sample["point"]

        # B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        # image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        image = torch.tensor(image)

        image = augmentation(image)

        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def coordinate_jitter(
        point,
        radius=0.01 # approximately 1 km
    ):
    return point + torch.rand(point.shape) * radius

def get_gsi_train_transform(resize_crop_size=224,
                        mean=[123.675, 116.28, 103.53], # ImageNet stats
                        std=[58.395, 57.12, 57.375]): # ImageNet stats
    augmentation = A.Compose(
        [
            A.RandomResizedCrop(height=resize_crop_size, width=resize_crop_size),
            A.RandomBrightnessContrast(),
            A.GaussianBlur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(), # changes the dimension ordering from (H, W, C) back to (C, H, W)
        ]
    )

    def transform(sample):
        image = sample["image"].transpose(1,2,0) # Albumentations expects image data in the (height, width, channels) format 
        point = sample["point"]

        image = augmentation(image=image)["image"]
        point = coordinate_jitter(point, radius=0.001)

        return dict(image=image, point=point)

    return transform

def get_pretrained_gsv_train_transform(image_processor):
    def transform(sample):
        image = sample["image"]  # image should be in [C, H, W] format
        point = sample["point"]  

        # Process image using CLIP processor
        image = image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)  # Shape: [3, 224, 224]

        # Optional: Jitter the coordinates
        point = coordinate_jitter(point, radius=0.0001)

        return dict(image=image, point=point)

    return transform