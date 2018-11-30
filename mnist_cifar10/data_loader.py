import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler


def get_mnist(batch_size, train, path, augmentation=False, std=0.0, shuffle=True, adversarial=False, subset=1000,
              workers=0):
    classes = np.arange(0, 10)

    if augmentation:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.Tensor(x.size()).normal_(mean=0.0, std=std)),  # add gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    dataset = torchvision.datasets.MNIST(root=path,
                                         train=train,
                                         download=True,
                                         transform=transform)

    if adversarial:
        np.random.seed(123)  # load always the same random subset
        indices = np.random.choice(np.arange(dataset.__len__()), subset)

        subset_sampler = SubsetSampler(indices)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=subset_sampler,
                                                 num_workers=workers
                                                 )

    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=workers
                                                 )
    return dataloader, dataset, classes


def get_cifar10(batch_size, train, path, augmentation=False, std=0.0, shuffle=True, adversarial=False, subset=1000,
                workers=0):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if augmentation:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.Tensor(x.size()).normal_(0.0, std)),  # add gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    dataset = torchvision.datasets.CIFAR10(root=path,
                                           train=train,
                                           download=True,
                                           transform=transform)

    if adversarial:
        np.random.seed(123)  # load always the same random subset
        indices = np.random.choice(np.arange(dataset.__len__()), subset)

        subset_sampler = SubsetSampler(indices)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=subset_sampler,
                                                 num_workers=workers
                                                 )
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=workers
                                                 )
    return dataloader, dataset, classes


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.shuffle = False

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
