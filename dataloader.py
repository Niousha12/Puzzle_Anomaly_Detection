import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.datasets import ImageFolder


def load_data(config):
    normal_class = config['normal_class']
    batch_size = config['batch_size']
    img_size = config['image_size']

    if config['dataset_name'] in ['cifar10']:
        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        os.makedirs("./train/CIFAR10", exist_ok=True)
        dataset = CIFAR10('./train/CIFAR10', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./test/CIFAR10", exist_ok=True)
        test_set = CIFAR10("./test/CIFAR10", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['mnist']:
        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        os.makedirs("./train/MNIST", exist_ok=True)
        dataset = MNIST('./train/MNIST', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./test/MNIST", exist_ok=True)
        test_set = MNIST("./test/MNIST", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['fashionmnist']:
        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        os.makedirs("./train/FashionMNIST", exist_ok=True)
        dataset = FashionMNIST('./train/FashionMNIST', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./test/FashionMNIST", exist_ok=True)
        test_set = FashionMNIST("./test/FashionMNIST", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['brain_tumor', 'head_ct']:
        img_transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        root_path = 'Dataset/medical/' + config['dataset_name']
        train_data_path = root_path + '/train'
        test_data_path = root_path + '/test'
        dataset = ImageFolder(root=train_data_path, transform=img_transform)
        load_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataset_array = next(iter(load_dataset))[0]
        my_dataset = TensorDataset(train_dataset_array)
        train_set, val_set = torch.utils.data.random_split(my_dataset, [train_dataset_array.shape[0] - 5, 5])

        test_set = ImageFolder(root=test_data_path, transform=img_transform)

    elif config['dataset_name'] in ['coil100']:
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        root_path = 'Dataset/coil100/' + config['dataset_name']
        train_data_path = root_path + '/train'
        test_data_path = root_path + '/test'
        dataset = ImageFolder(root=train_data_path, transform=img_transform)
        load_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataset_array = next(iter(load_dataset))[0]
        my_dataset = TensorDataset(train_dataset_array)
        train_set, val_set = torch.utils.data.random_split(my_dataset, [train_dataset_array.shape[0] - 5, 5])

        test_set = ImageFolder(root=test_data_path, transform=img_transform)

    elif config['dataset_name'] in ['MVTec']:
        data_path = 'Dataset/MVTec/' + normal_class + '/train'
        data_list = []

        orig_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        orig_dataset = ImageFolder(root=data_path, transform=orig_transform)

        train_orig, val_set = torch.utils.data.random_split(orig_dataset, [len(orig_dataset) - 25, 25])
        data_list.append(train_orig)

        for i in range(3):
            img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomAffine(0, scale=(1.05, 1.2)),
                transforms.ToTensor()])

            dataset = ImageFolder(root=data_path, transform=img_transform)
            data_list.append(dataset)

        dataset = ConcatDataset(data_list)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataset_array = next(iter(train_loader))[0]
        train_set = TensorDataset(train_dataset_array)

        test_data_path = '/content/' + normal_class + '/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloader, val_dataloader, test_dataloader
