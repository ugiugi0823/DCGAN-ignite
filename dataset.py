try:
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torchvision.utils as vutils

except ImportError:
    raise ModuleNotFoundError(
        "Please install torchvision to run this example, for example "
        "via conda by running 'conda install -c pytorch torchvision'. "
    )



def check_dataset(dataset, dataroot):
    """

    Args:
        dataset (str): Name of the dataset to use. See CLI help for details
        dataroot (str): root directory where the dataset will be stored.

    Returns:
        dataset (data.Dataset): torchvision Dataset object

    """
    resize = transforms.Resize(64)
    crop = transforms.CenterCrop(64)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if dataset in {"imagenet", "folder", "lfw"}:
        dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([resize, crop, to_tensor, normalize]))
        nc = 3

    elif dataset == "lsun":
        dataset = dset.LSUN(
            root=dataroot, classes=["bedroom_train"], transform=transforms.Compose([resize, crop, to_tensor, normalize])
        )
        nc = 3

    elif dataset == "cifar10":
        dataset = dset.CIFAR10(
            root=dataroot, download=True, transform=transforms.Compose([resize, to_tensor, normalize])
        )
        nc = 3

    elif dataset == "mnist":
        dataset = dset.MNIST(root=dataroot, download=True, transform=transforms.Compose([resize, to_tensor, normalize]))
        nc = 1

    elif dataset == "fake":
        dataset = dset.FakeData(size=256, image_size=(3, 64, 64), transform=to_tensor)
        nc = 3

    else:
        raise RuntimeError(f"Invalid dataset name: {dataset}")

    return dataset, nc
