import torch
import torchvision
from torchvision import datasets
from .data_transforms import torch_transforms
from utils import has_cuda, imshow


class DataEngine(object):
    
    classes = ["%s" % i for i in range(200)]

    def __init__(self, args):
        super(DataEngine, self).__init__()
        self.num_classes = 200
        self.batch_size_cuda = args.batch_size_cuda
        self.batch_size_cpu = args.batch_size_cpu
        self.num_workers = args.num_workers
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.load()

    def _transforms(self):
        train_transform = torch_transforms(train=True)
        test_transform = torch_transforms(train=False)
        return train_transform, test_transform
    
    def _dataset(self):
        train_transform, test_transform = self._transforms()

        train_set = datasets.ImageFolder(root=self.train_data_path, transform=train_transform)
        test_set = datasets.ImageFolder(root=self.test_data_path, transform=test_transform)

        return train_set, test_set
    
    def load(self):
        train_set, test_set = self._dataset()

        dataloader_args = dict(shuffle=True, batch_size=self.batch_size_cpu)

        if has_cuda():
            dataloader_args.update(batch_size=self.batch_size_cuda, num_workers=self.num_workers, pin_memory=True)

        self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
        self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)


    def show_samples(self):
        images, labels = next(iter(self.train_loader))
        idx = []
        num_im = min(len(self.classes), 10)

        for i in range(num_im):
            for j in range(len(labels)):
                if labels[j]==i:
                    idx.append(j)
                    break
        if len(idx)<num_im:
            for j in range(len(labels)):
                if len(idx)==num_im:
                    break
                if j not in idx:
                    idx.append(j)

        imshow(torchvision.utils.make_grid(images[idx], nrow=num_im, scale_each=True), "Sample train data")
