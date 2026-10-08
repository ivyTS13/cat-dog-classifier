from cat_dog_classifier.dataset import create_dataloaders


def main():
    train_loader, val_loader = create_dataloaders(
        "data",
        batch_size=32
    )

    print("Classes:")
    print(train_loader.dataset.classes)

    print("\nTraining classes:")
    print(train_loader.dataset.targets.count(0))
    print(train_loader.dataset.targets.count(1))

    print("\nValidation classes:")
    print(val_loader.dataset.targets.count(0))
    print(val_loader.dataset.targets.count(1))


if __name__ == "__main__":
    main()