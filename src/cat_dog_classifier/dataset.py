from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(data_dir, batch_size=32):

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
    ])
    

    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])
    
    train_dataset = datasets.ImageFolder(
        root =f"{data_dir}/train",
        transform= train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=val_transform
    )

    test_dataset = datasets.ImageFolder(
            root=f"{data_dir}/test",
            transform=val_transform
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle= True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle= False
    )

    test_loader = DataLoader(
            test_dataset,
            batch_size= batch_size,
            shuffle= False
        )
    return train_loader, val_loader, test_loader