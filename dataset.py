import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


class CIFAR10(object):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    classes = None

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.loader_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': True}
        self.train_loader, self.test_loader = self.get_loaders()

    def get_train_loader(self):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)
        if self.classes is None:
            self.classes = {i: c for i, c in enumerate(train_data.classes)}
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **self.loader_kwargs)
        return self.train_loader

    def get_test_loader(self):
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, **self.loader_kwargs)
        return self.test_loader

    def get_loaders(self):
        return self.get_train_loader(), self.get_test_loader()

    @classmethod
    def denormalise(cls, tensor):
        for t, m, s in zip(tensor, cls.mean, cls.std):
            t.mul_(s).add_(m)
        return tensor

    def show_examples(self, figsize=None):
        batch_data, batch_label = next(iter(self.train_loader))

        _ = plt.figure(figsize=figsize)
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            plt.imshow(self.denormalise(batch_data[i]).permute(1, 2, 0))
            label = batch_label[i].item()
            if self.classes is not None:
                label = str(label) + ':' + self.classes[label]
            plt.title(label)
            plt.xticks([])
            plt.yticks([])
