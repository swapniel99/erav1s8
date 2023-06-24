import os
from matplotlib import pyplot as plt
from torchvision import transforms
from abc import abstractmethod


class DataSet(object):
    mean = None
    std = None
    classes = None

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.loader_kwargs = {'batch_size': batch_size, 'num_workers': os.cpu_count(), 'pin_memory': True}
        self.std_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.augment_transforms = None
        self.train_transforms = self.std_transforms
        self.test_transforms = self.std_transforms
        self.train_loader, self.test_loader = self.get_loaders()
        self.example_iter = iter(self.train_loader)

    @abstractmethod
    def get_train_loader(self):
        all_transforms = list()
        if self.augment_transforms is not None:
            all_transforms.append(self.augment_transforms)
        all_transforms.append(self.std_transforms)
        self.train_transforms = transforms.Compose(all_transforms)

    @abstractmethod
    def get_test_loader(self):
        pass

    def get_loaders(self):
        return self.get_train_loader(), self.get_test_loader()

    @classmethod
    def denormalise(cls, tensor):
        for t, m, s in zip(tensor, cls.mean, cls.std):
            t.mul_(s).add_(m)
        return tensor

    @abstractmethod
    def show_transform(self, img):
        pass

    def show_examples(self, figsize=None, denorm=True):
        batch_data, batch_label = next(self.example_iter)

        _ = plt.figure(figsize=figsize)
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            image = batch_data[i]
            if denorm:
                image = self.denormalise(image)
            plt.imshow(self.show_transform(image), cmap='gray')
            label = batch_label[i].item()
            if self.classes is not None:
                label = f'{label}:{self.classes[label]}'
            plt.title(str(label))
            plt.xticks([])
            plt.yticks([])
