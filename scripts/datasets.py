import os
import cv2
import glob 
import pandas as pd
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset

# CIFAR10 and ImageNet normalization transform
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# Define the classification datasets
# MNIST_TRAIN = torchvision.datasets.MNIST(root="datasets", train=True, download=True, 
#                                    transform=transforms.Compose([
#                                              transforms.ToTensor(),
#                                              transforms.Normalize((0.1307,), (0.3081,))
#                                     ]))

# MNIST_TEST = torchvision.datasets.MNIST(root="datasets", train=False, download=True, 
#                                    transform=transforms.Compose([
#                                              transforms.ToTensor(),
#                                              transforms.Normalize((0.1307,), (0.3081,))
#                                     ]))

# FASHION_MNIST_TRAIN = torchvision.datasets.FashionMNIST(root="datasets", train=True, download=True,
#                                    transform=transforms.Compose([
#                                              transforms.ToTensor(),
#                                              transforms.Normalize((0.1307,), (0.3081,))
#                                     ]))

# FASHION_MNIST_TEST = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, 
#                                    transform=transforms.Compose([
#                                              transforms.ToTensor(),
#                                              transforms.Normalize((0.1307,), (0.3081,))
#                                     ]))

# SVHN_TRAIN = torchvision.datasets.SVHN(root="datasets/svhn", split="train", download=True,
#                                 transform=transforms.Compose([
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ]))

# SVHN_TEST = torchvision.datasets.SVHN(root="datasets/svhn", split="test", download=True,
#                                 transform=transforms.Compose([
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ])) 

# CIFAR10_TRAIN = torchvision.datasets.CIFAR10(root="datasets/", train=True, download=True,
#                                 transform=transforms.Compose([
#                                           transforms.RandomCrop(32, padding=4),
#                                           transforms.RandomHorizontalFlip(), 
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ]))

# CIFAR10_TEST = torchvision.datasets.CIFAR10(root="datasets/", train=False, download=True,
#                                 transform=transforms.Compose([
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ]))

# CIFAR100_TRAIN = torchvision.datasets.CIFAR100(root="datasets/", train=True, download=True,
#                                 transform=transforms.Compose([
#                                           transforms.RandomCrop(32, padding=4),
#                                           transforms.RandomHorizontalFlip(), 
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ]))

# CIFAR100_TEST = torchvision.datasets.CIFAR100(root="datasets/", train=False, download=True,
#                                 transform=transforms.Compose([
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ]))

# IMAGENET_TRAIN = torchvision.datasets.ImageNet(root="datasets/ImageNet", split="train",
#                                 transform=transforms.Compose([
#                                           transforms.Resize(256),
#                                           transforms.RandomResizedCrop(224),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.ToTensor(),
#                                           normalize]
#                                 ))

# IMAGENET_VAL = torchvision.datasets.ImageNet(root="datasets/ImageNet", split="val",
#                                 transform=transforms.Compose([
#                                           transforms.Resize(256),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           normalize
#                                 ]))


class PneumoniaDataset(Dataset):
    def __init__(self, root, annotations_path, transform):
        self.root = root
        annotations = pd.read_csv(annotations_path).to_dict()
        self.file_names = annotations["file_name"]
        self.class_ids = annotations["class_id"]
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        file_name = self.file_names[index]
        class_id = self.class_ids[index]
        image = Image.open(os.path.join(self.root, file_name)).convert("RGB")
        image = self.transform(image)
        return image, class_id


train_transform = v2.Compose([v2.ToImagePIL(),
                              v2.Resize(size=(256, 256), antialias=True),
                              v2.RandomResizedCrop(size=(224, 224), antialias=True),
                              v2.RandAugment(),
                              v2.AugMix(),
                              v2.ToTensor(),
                              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])


val_transform = v2.Compose([v2.ToImagePIL(),
                            v2.Resize(256, antialias=True),
                            v2.CenterCrop(224),
                            v2.ToTensor(),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

PNEUMONIA_TRAIN = PneumoniaDataset(root="datasets/Pneumonia/train_images", annotations_path="datasets/Pneumonia/train_labels.csv", transform=train_transform)
PNEUMONIA_VAL = PneumoniaDataset(root="datasets/Pneumonia/train_images", annotations_path="datasets/Pneumonia/val_labels.csv", transform=val_transform)


# Function to load the correct training and testing set
def get_dataset(dataset_id):
    DATASETS = dict(# MNIST=[MNIST_TRAIN, MNIST_TEST], 
    #             FashionMNIST=[FASHION_MNIST_TRAIN, FASHION_MNIST_TEST],
    #             CIFAR10=[CIFAR10_TRAIN, CIFAR10_TEST], 
    #             CIFAR100=[CIFAR100_TRAIN, CIFAR100_TEST],
    #             SVHN=[SVHN_TRAIN, SVHN_TEST],
    #             ILSVRC2012=[IMAGENET_TRAIN, IMAGENET_VAL],
                PNEUMONIA=[PNEUMONIA_TRAIN, PNEUMONIA_VAL])
    return DATASETS.get(dataset_id)
