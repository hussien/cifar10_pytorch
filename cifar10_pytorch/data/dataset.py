import torch
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as T

def build_transforms(cfg, is_train=True):
    list_transforms = list()

    # Resizing. Some models must have ImageNet-size images as input
    if is_train and cfg.DATA.IMG_SIZE != 32:
        list_transforms.append(T.Resize((cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)))
    elif is_train == False and cfg.TEST.IMG_SIZE != 32:
        list_transforms.append(T.Resize((cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)))

    if is_train:
        if cfg.DATA.RANDOMCROP:
            list_transforms.append(T.RandomCrop(cfg.DATA.IMG_SIZE, padding=4))
        if cfg.DATA.LRFLIP:
            list_transforms.append(T.RandomHorizontalFlip())

    list_transforms.extend([
        T.ToTensor(),
        T.Normalize(cfg.DATA.NORMALIZE_MEAN, cfg.DATA.NORMALIZE_STD),
    ])

    transforms = T.Compose(list_transforms)

    return transforms

def prepare_cifar10_dataset(cfg):
    """ prepare CIFAR10 dataset based on configuration"""

    transform_train = build_transforms(cfg, is_train=True)
    transform_test = build_transforms(cfg, is_train=False)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # dataset_train = torchvision.datasets.CIFAR10(
    #     root='./data',
    #     train=True,
    #     download=True,
    #     transform=transform_train
    # )
    dataset_train = datasets.ImageFolder(
            '/content/CINIC-10-Filtered_1K/train',
            transform=transform_train
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.SOLVER.BATCHSIZE,
        shuffle=True,
        num_workers=cfg.SOLVER.NUM_WORKERS
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.BATCHSIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.NUM_WORKERS
    )

    return dataloader_train, dataloader_test