import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.fast_tensor_dataloader import FastTensorDataLoader


def get_mnist_dataloaders(batch_size=128, path_to_data='../data', warm_up=True, device=None):
    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    target_transform = lambda x: F.one_hot(torch.tensor(x), num_classes=10)
    train_data = datasets.MNIST(path_to_data, train=True, download=True, transform=data_transforms, target_transform=target_transform)
    test_data = datasets.MNIST(path_to_data, train=False, transform=data_transforms, target_transform=target_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    warm_up_loader = None
    if warm_up:
        warm_up_x, warm_up_y = WarmUpMNISTDataset(path_to_data, count=256, transform=data_transforms, target_transform=target_transform, device=device).get_tensors()
        warm_up_loader = FastTensorDataLoader(warm_up_x, warm_up_y, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, warm_up_loader


class WarmUpMNISTDataset(datasets.MNIST):
    def __init__(self, root, transform=None, target_transform=None, download=False, count=256, device=None):
        self.__class__.__name__ = 'MNIST'  # This is used in directory structure of datasets.MNIST
        super(WarmUpMNISTDataset, self).__init__(root, train=True, transform=transform, target_transform=target_transform, download=download)
        self.device = device
        self.count = count
        self.delete = set()
        self.mapping = list(set(range(count + len(self.delete))) - self.delete)

        self.save_all_images()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        translated_index = self.mapping[index]
        return super().__getitem__(translated_index)

    def get_tensors(self):
        x_shape, y_shape = self[0][0].shape, self[0][1].shape
        x, y = torch.zeros(self.count, *x_shape, device=self.device), torch.zeros(self.count, *y_shape, device=self.device)
        for i, (data, label) in enumerate(self):
            x[i], y[i] = data.to(self.device), label.to(self.device)
        return x, y

    def save_all_images(self):
        x_shape = self[0][0].shape
        all_images = torch.zeros(self.count, *x_shape)
        for i, (data, label) in enumerate(self):
            all_images[i] = data
        save_image(all_images, 'warm_up.png', nrow=(len(self) // 16))


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64', device=None, warm_up=True):
    data_transforms = transforms.Compose([
        # transforms.Resize(64),
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_kwargs = {
        'target_type': 'attr',
        'transform': data_transforms,
    }
    train_data = datasets.CelebA(path_to_data, split='train', download=True, **dataset_kwargs)
    test_data = datasets.CelebA(path_to_data, split='test', **dataset_kwargs)
    # warm_up_data = WarmUpCelebADataset(path_to_data, split='train', target_transform=target_transforms, **dataset_kwargs)

    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'pin_memory': device.type != 'cpu',
        # 'pin_memory': False,
        'num_workers': 0 if device.type == 'cpu' else 4,
    }
    train_loader = DataLoader(train_data, **dataloader_kwargs)
    test_loader = DataLoader(test_data, **dataloader_kwargs)
    # warm_up_loader = DataLoader(warm_up_data, **dataloader_kwargs)

    warm_up_loader = None
    if warm_up:
        # target_transforms = transforms.Compose([
        #     lambda x: x[celeba_good_columns],
        #     # lambda x: torch.flatten(F.one_hot(x, num_classes=2))
        #     my_celeba_target_transfrom
        # ])
        warm_up_x, warm_up_y = WarmUpCelebADataset(path_to_data, count=800, device=device, **dataset_kwargs).get_tensors()  # TODO: If it is good, make the class simpler
        warm_up_loader = FastTensorDataLoader(warm_up_x, warm_up_y, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, warm_up_loader


class WarmUpCelebADataset(datasets.CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, download=False,
                 count=256, device=None):
        super().__init__(root, split, target_type, transform, target_transform, download)
        self.count = count
        self.device = device
        # self.delete = {2, 36, 43, 66, 74, 96, 119, 148, 149, 162, 166, 168, 183, 188, 198}  # From 0 to 255+15
        # self.delete = {43, 74, 162, 183}  # From 0 to 299
        self.delete = set()
        self.mapping = list(set(range(count + len(self.delete))) - self.delete)

        self.labels = torch.tensor(np.genfromtxt('warm_up_labels.csv', delimiter=','), dtype=torch.float)

        # self.save_all_images()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # return super().__getitem__(index)
        translated_index = self.mapping[index]
        x, _ = super().__getitem__(translated_index)
        return x, self.labels[translated_index]

    def get_tensors(self):
        x_shape, y_shape = self[0][0].shape, self[0][1].shape
        x, y = torch.zeros(self.count, *x_shape, device=self.device), torch.zeros(self.count, *y_shape, device=self.device)
        for i, (data, label) in enumerate(self):
            x[i], y[i] = data.to(self.device), label.to(self.device)
        return x, y

    def save_all_images(self):
        x_shape = self[0][0].shape
        all_images = torch.zeros(self.count, *x_shape)
        for i, (data, label) in enumerate(self):
            all_images[i] = data
        save_image(all_images, 'warm_up.png', nrow=(len(self) // 16))


def get_dsprites_dataloader(batch_size=128, path_to_data='../dsprites/ndarray.npz', fraction=1., device=None, warm_up=False):
    dsprites_data = DSpritesDataset(path_to_data, fraction=fraction, device=device)
    # dsprites_loader = FastTensorDataLoader(*dsprites_data.get_tensors(), batch_size=batch_size, shuffle=True)  # Comment if you have memory limits, and uncomment the next line
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'pin_memory': device.type != 'cpu',
        'num_workers': 0 if device.type == 'cpu' else 4,
    }
    dsprites_loader = DataLoader(dsprites_data, **dataloader_kwargs)

    warm_up_loader = None
    if warm_up:
        warm_up_data = DSpritesWarmUpDataset(path_to_data, device=device)
        warm_up_loader = FastTensorDataLoader(*warm_up_data.get_tensors(), batch_size=batch_size, shuffle=True)

    return dsprites_loader, warm_up_loader


class DSpritesWarmUpDataset(Dataset):
    # Color[1], Shape[3], Scale, Orientation, PosX, PosY
    def __init__(self, path_to_data, size=10000, device=None):  # was 100, 737, 1000, 3686, 10000
        self.device = device
        data = np.load(path_to_data)
        indices = self.good_indices(size)
        self.imgs = np.expand_dims(data['imgs'][indices], axis=1)

        shape_value = data['latents_classes'][indices, 1]
        self.classes = np.zeros((size, 3))
        self.classes[np.arange(size), shape_value] = 1

        print(np.mean(self.classes, axis=0))

    def good_indices(self, size):
        # if size < 3 * 6 * 2 * 2 * 2:
        #     raise Exception('Too small!')
        indices = np.zeros(size, dtype=np.long)
        # [1, 3, 6, 40, 32, 32]
        module = np.array([737280, 245760, 40960, 1024, 32, 1])
        i = 0
        while True:
            for y_span in range(2):
                for x_span in range(2):
                    for orientation_span in range(2):
                        for scale in range(6):
                            for shape in range(3):
                                orientation = int(np.random.randint(0, 20, 1) + orientation_span * 20)
                                x = int(np.random.randint(0, 16, 1) + x_span * 16)
                                y = int(np.random.randint(0, 16, 1) + y_span * 16)
                                sample = np.array([0, shape, scale, orientation, x, y])
                                indices[i] = np.sum(sample * module)
                                i += 1
                                if i >= size:
                                    return indices

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.classes[idx]

    def get_tensors(self):
        return torch.tensor(self.imgs, dtype=torch.float, device=self.device), torch.tensor(self.classes, device=self.device)


class DSpritesDataset(Dataset):
    # Color[1], Shape[3], Scale, Orientation, PosX, PosY
    def __init__(self, path_to_data, fraction=1., device=None):
        self.device = device
        data = np.load(path_to_data)
        self.imgs = data['imgs']
        self.imgs = np.expand_dims(self.imgs, axis=1)
        self.classes = data['latents_classes']
        if fraction < 1:
            indices = np.random.choice(737280, size=int(fraction * 737280), replace=False)
            self.imgs = self.imgs[indices]
            self.classes = self.classes[indices]
        # self.attrs = data['latents_values'][indices]
        # self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # # Each image in the dataset has binary values so multiply by 255 to get
        # # pixel values
        # sample = self.imgs[idx] * 255
        # # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        # sample = sample.reshape(sample.shape + (1,))

        # if self.transform:
        #     sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        # return sample, (self.classes[idx], self.attrs[idx])

        # return torch.tensor(self.imgs[idx], dtype=torch.float, device=self.device), torch.tensor(self.classes[idx], device=self.device)
        return torch.tensor(self.imgs[idx], dtype=torch.float), torch.tensor(self.classes[idx])

    def get_tensors(self):
        return torch.tensor(self.imgs, dtype=torch.float, device=self.device), torch.tensor(self.classes, device=self.device)
