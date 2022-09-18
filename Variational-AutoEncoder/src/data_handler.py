from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_training_data(batch_size=32, num_workers=0, pin_memory=False):
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader


def get_testing_data(batch_size=32, num_workers=0, pin_memory=False):
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader

