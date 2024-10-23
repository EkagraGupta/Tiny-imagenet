import torchvision.transforms as transforms

def torch_transforms(train=False):
    transforms_list = []
    if train:
        transforms_list.extend([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transforms_list.extend([
        transforms.ToTensor(),
    ])

    return transforms.Compose(transforms_list)